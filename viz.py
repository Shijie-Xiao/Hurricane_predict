"""Visualization: SHAP analysis, Integrated-Gradients saliency and PCMCI causal graphs."""

import re
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import matplotlib.patheffects as pe
import networkx as nx

from scipy.ndimage import gaussian_filter
from mpl_toolkits.basemap import Basemap

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import shap

from models import build_model, UNet
from dataset import PatchDataset, HeatmapDataset


# -- geo helpers --------------------------------------------------------------

def _geo_grid(cfg, H, W, ds=2):
    """Geographic bounds and meshgrid from config region."""
    lat0 = cfg["data"]["region"]["lat"][0]
    lon0 = cfg["data"]["region"]["lon"][0]
    lat_max = 90.0 - (lat0 + 0.5)
    lat_min = 90.0 - (lat0 + H - 0.5)
    lon_min = -180.0 + (lon0 + 0.5)
    lon_max = -180.0 + (lon0 + W - 0.5)
    lats = np.linspace(lat_max, lat_min, max(1, H // ds))
    lons = np.linspace(lon_min, lon_max, max(1, W // ds))
    lon2d, lat2d = np.meshgrid(lons, lats)
    return (lat_min, lat_max, lon_min, lon_max), (lon2d, lat2d)


def _plot_map(path, data, title, bounds, grids, levels):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    m = Basemap(projection="cyl",
                llcrnrlat=bounds[0], urcrnrlat=bounds[1],
                llcrnrlon=bounds[2], urcrnrlon=bounds[3],
                resolution="i", ax=ax)
    ax.set_facecolor("white")
    m.drawcoastlines(color="black", linewidth=0.3, zorder=2)
    cf = m.contourf(grids[0], grids[1], data, levels=levels,
                    cmap="RdBu_r", extend="both", alpha=0.9, zorder=1)
    ax.set_title(title, fontsize=12); ax.axis("off")
    fig.colorbar(cf, ax=ax, orientation="vertical", fraction=0.04, pad=0.02)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -- SHAP analysis ------------------------------------------------------------

class _MeanScalarWrapper(nn.Module):
    """Wrap any model that outputs (B, ...) into a scalar (B, 1) for SHAP."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        out = self.base(x)
        return out.reshape(out.shape[0], -1).mean(dim=1, keepdim=True)


def _pick_shap_samples(dataset, n_bg, n_ex, seed=0):
    """Select background and explanation samples from a PyTorch Dataset.

    Works for both PatchDataset (returns x,y,...) and HeatmapDataset (returns x,y).
    Returns (bg_tensor, ex_tensor) with the input x stacked.
    """
    total = min(len(dataset), n_bg + n_ex)
    n_bg = min(n_bg, total - 1)
    n_ex = total - n_bg
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), total, replace=False)

    xs = []
    for i in indices:
        item = dataset[i]
        xs.append(item[0])  # x tensor

    stacked = torch.stack(xs)
    return stacked[:n_bg], stacked[n_bg:]


def shap_analysis(cfg, ckpt_path=None):
    """Run SHAP GradientExplainer on both Timesformer and UNet.

    Produces per-channel SHAP importance (beeswarm plot) and saves raw
    SHAP values as .npy files.  Model structures are unchanged; only the
    wrapper adds a mean-pooling layer so the output is scalar for SHAP.
    """
    sc = cfg["shap"]
    mc = cfg["model"]
    dc = cfg["data"]
    tc = cfg["train"]
    n_bg = sc["n_background"]
    n_ex = sc["n_explain"]
    out_dir = Path(sc["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_ch = mc["in_ch"]          # 9 real ERA5 channels
    channel_names = dc["channel_names"]
    ph, pw = mc["patch"]
    seed = tc["seed"]

    cache = Path(dc["cache_dir"])
    env_path = str(cache / "region_env.npy")
    hurr_path = str(cache / "region_hurr.npy")

    # ── Timesformer SHAP ─────────────────────────────────────────────
    mtype = mc["type"]
    model = build_model(cfg).to(device)
    if ckpt_path is None:
        ckpt_path = str(Path(tc["out_dir"]) / mtype / "run_01" / "best.pt")
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"[shap] Loaded Timesformer ({mtype}): {ckpt_path}")

    tf_ds = PatchDataset(
        env_path, hurr_path,
        seq_len=mc["seq_len"], stride=tc["stride"],
        patch_h=ph, patch_w=pw,
    )
    bg_tf, ex_tf = _pick_shap_samples(tf_ds, n_bg, n_ex, seed=seed)
    # PatchDataset returns x as (T, C, H, W); model expects (B, T, C, H, W)
    print(f"[shap] Timesformer background: {bg_tf.shape}, explain: {ex_tf.shape}")

    wrapper_tf = _MeanScalarWrapper(model).to(device)
    wrapper_tf.eval()

    bg_dev = bg_tf.to(device)
    ex_dev = ex_tf.to(device)
    explainer_tf = shap.GradientExplainer(wrapper_tf, bg_dev)
    sv_tf = explainer_tf.shap_values(ex_dev)
    if isinstance(sv_tf, list):
        sv_tf = sv_tf[0]
    sv_tf = np.array(sv_tf)
    if sv_tf.ndim == ex_tf.ndim + 1:
        sv_tf = sv_tf.squeeze(-1)
    print(f"[shap] Timesformer SHAP values shape: {sv_tf.shape}")

    np.save(str(out_dir / "shap_timesformer.npy"), sv_tf)

    # Channel aggregation: sv_tf is (B, T, C_total, H, W)
    # Only take the first `raw_ch` channels (skip positional lat/lon)
    sv_real = sv_tf[:, :, :raw_ch, :, :]  # (B, T, 9, H, W)
    ex_real = ex_tf.numpy()[:, :, :raw_ch, :, :]

    ch_shap_tf = np.abs(sv_real).mean(axis=(1, 3, 4))  # (B, 9)
    ch_data_tf = ex_real.mean(axis=(1, 3, 4))           # (B, 9)

    explanation_tf = shap.Explanation(
        values=ch_shap_tf,
        data=ch_data_tf,
        feature_names=channel_names,
    )
    fig_tf = plt.figure()
    shap.plots.beeswarm(explanation_tf, show=False)
    fig_tf.savefig(str(out_dir / "shap_timesformer_beeswarm.png"),
                   dpi=200, bbox_inches="tight")
    plt.close(fig_tf)
    print(f"[shap] Saved {out_dir / 'shap_timesformer_beeswarm.png'}")

    # ── UNet SHAP ────────────────────────────────────────────────────
    unet_ckpt = Path(cfg["unet"]["out_ckpt"])
    if unet_ckpt.exists():
        unet_in_ch = raw_ch + 2  # env + patch-level positional grid
        unet_model = UNet(in_ch=unet_in_ch, ph=ph, pw=pw).to(device)
        unet_model.load_state_dict(
            torch.load(str(unet_ckpt), map_location=device, weights_only=True)
        )
        unet_model.eval()
        print(f"[shap] Loaded UNet: {unet_ckpt}")

        unet_ds = HeatmapDataset(
            env_path, hurr_path, ph, pw,
            sigma=cfg["unet"]["sigma"],
        )
        bg_un, ex_un = _pick_shap_samples(unet_ds, n_bg, n_ex, seed=seed)
        print(f"[shap] UNet background: {bg_un.shape}, explain: {ex_un.shape}")

        wrapper_un = _MeanScalarWrapper(unet_model).to(device)
        wrapper_un.eval()

        bg_un_dev = bg_un.to(device)
        ex_un_dev = ex_un.to(device)
        explainer_un = shap.GradientExplainer(wrapper_un, bg_un_dev)
        sv_un = explainer_un.shap_values(ex_un_dev)
        if isinstance(sv_un, list):
            sv_un = sv_un[0]
        sv_un = np.array(sv_un)
        if sv_un.ndim == ex_un.ndim + 1:
            sv_un = sv_un.squeeze(-1)
        print(f"[shap] UNet SHAP values shape: {sv_un.shape}")

        np.save(str(out_dir / "shap_unet.npy"), sv_un)

        # UNet input is (C+2, ph, pw); only take first `raw_ch` channels
        sv_un_real = sv_un[:, :raw_ch, :, :]
        ex_un_real = ex_un.numpy()[:, :raw_ch, :, :]
        ch_shap_un = np.abs(sv_un_real).mean(axis=(2, 3))  # (B, 9)
        ch_data_un = ex_un_real.mean(axis=(2, 3))

        explanation_un = shap.Explanation(
            values=ch_shap_un,
            data=ch_data_un,
            feature_names=channel_names,
        )
        fig_un = plt.figure()
        shap.plots.beeswarm(explanation_un, show=False)
        fig_un.savefig(str(out_dir / "shap_unet_beeswarm.png"),
                       dpi=200, bbox_inches="tight")
        plt.close(fig_un)
        print(f"[shap] Saved {out_dir / 'shap_unet_beeswarm.png'}")
    else:
        print(f"[shap] UNet checkpoint not found ({unet_ckpt}), skipping UNet SHAP.")

    print("[shap] Done.")


# -- Integrated Gradients saliency -------------------------------------------

def saliency(cfg, ckpt_path=None):
    """Compute per-channel global + per-patch IG saliency maps."""
    vc = cfg["viz"]["saliency"]
    mc = cfg["model"]
    dc = cfg["data"]
    ig_steps = vc["ig_steps"]
    ds = vc["downsample"]
    out_root = Path(vc["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["model"]["type"] = "hierarchical"
    model = build_model(cfg).to(device)

    if ckpt_path is None:
        ckpt_path = str(
            Path(cfg["train"]["out_dir"]) / "hierarchical" / "run_01" / "best.pt"
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"[saliency] Loaded checkpoint: {ckpt_path}")

    cache = Path(dc["cache_dir"])
    ph, pw = mc["patch"]
    full_ds = PatchDataset(
        str(cache / "region_env.npy"), str(cache / "region_hurr.npy"),
        seq_len=mc["seq_len"], stride=cfg["train"]["stride"],
        patch_h=ph, patch_w=pw,
    )
    n_val = max(1, int(len(full_ds) * cfg["train"]["val_split"]))
    _, val_ds = random_split(
        full_ds, [len(full_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(cfg["train"]["seed"]),
    )
    loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    sample_x = next(iter(loader))[0]
    Hs, Ws = sample_x.shape[-2:]
    with torch.no_grad():
        _, _, nh, nw = model(sample_x[:1].to(device)).shape

    bounds, grids = _geo_grid(cfg, Hs, Ws, ds=ds)
    levels = np.linspace(-1, 1, 21)
    alphas = torch.linspace(0, 1, ig_steps, device=device)
    channels = dc["channel_names"]

    for c, name in enumerate(channels):
        safe = re.sub(r"[^\w_]", "_", name)
        print(f"[saliency] Channel {c}: {name}")
        cdir = out_root / safe
        cdir.mkdir(parents=True, exist_ok=True)

        patch_sum = {(i, j): np.zeros((Hs, Ws), dtype=np.float32)
                     for i in range(nh) for j in range(nw)}
        patch_cnt = {k: 0 for k in patch_sum}
        g_sum = np.zeros((Hs, Ws), dtype=np.float32)
        g_cnt = 0

        for xb, *_ in loader:
            B = xb.shape[0]
            xb = xb.to(device)
            base = torch.zeros_like(xb)
            diff = xb - base

            # global IG
            grad_acc = torch.zeros_like(xb)
            for a in alphas:
                inp = (base + a * diff).requires_grad_(True)
                model.zero_grad()
                model(inp).sum().backward()
                grad_acc += inp.grad
            ig = diff * grad_acc / ig_steps
            sal = ig[:, :, c, :, :].abs().mean(dim=1).detach().cpu().numpy()
            g_sum += sal.sum(axis=0)
            g_cnt += B

            # per-patch IG
            for i in range(nh):
                for j in range(nw):
                    grad_p = torch.zeros_like(xb)
                    for a in alphas:
                        inp = (base + a * diff).requires_grad_(True)
                        model.zero_grad()
                        model(inp)[:, :, i, j].sum().backward()
                        grad_p += inp.grad
                    igp = diff * grad_p / ig_steps
                    sp = igp[:, :, c, :, :].abs().mean(dim=1).detach().cpu().numpy()
                    patch_sum[(i, j)] += sp.sum(axis=0)
                    patch_cnt[(i, j)] += B

        for (i, j), sm in patch_sum.items():
            if patch_cnt[(i, j)] == 0:
                continue
            mp = gaussian_filter(sm / patch_cnt[(i, j)], sigma=1.0)
            dd = mp[::ds, ::ds]
            mx = float(np.max(np.abs(dd))) or 1.0
            normed = dd / mx
            np.save(cdir / f"patch_{i}_{j}.npy", normed)
            _plot_map(str(cdir / f"patch_{i}_{j}.png"), normed,
                      f"{name} - Patch ({i},{j})", bounds, grids, levels)

        if g_cnt > 0:
            mg = gaussian_filter(g_sum / g_cnt, sigma=1.0)
            dg = mg[::ds, ::ds]
            mx = float(np.max(np.abs(dg))) or 1.0
            normed = dg / mx
            np.save(cdir / "global_saliency.npy", normed)
            _plot_map(str(cdir / "global_saliency.png"), normed,
                      name, bounds, grids, levels)

    print("[saliency] Done.")


# -- PCMCI causal graph -------------------------------------------------------

def _xcorr(x, y, lag):
    if lag >= len(x):
        return 0.0
    a, b = x[lag:], y[:-lag]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return 0.0
    a = (a[m] - a[m].mean()) / (a[m].std() + 1e-12)
    b = (b[m] - b[m].mean()) / (b[m].std() + 1e-12)
    return float(np.corrcoef(a, b)[0, 1])


def _prescreen(X, r0, tau_max):
    K = X.shape[1]
    sel = {j: [] for j in range(K)}
    for j in range(K):
        for i in range(K):
            if i == j:
                continue
            for lag in range(1, tau_max + 1):
                if abs(_xcorr(X[:, j], X[:, i], lag)) >= r0:
                    sel[j].append((i, -lag))
    return sel


def _make_links(sel, K, tau_max):
    la = {j: {} for j in range(K)}
    for j in range(K):
        for i, tau in sel.get(j, []):
            if i != j and -tau_max <= tau <= -1:
                la[j][(i, tau)] = "-?>"
    return la


def _max_acorr(x, tau_max=5):
    vals = []
    for lag in range(1, tau_max + 1):
        if lag >= len(x):
            break
        a, b = x[lag:], x[:-lag]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 5:
            continue
        aa = (a[m] - a[m].mean()) / (a[m].std() + 1e-12)
        bb = (b[m] - b[m].mean()) / (b[m].std() + 1e-12)
        vals.append(np.corrcoef(aa, bb)[0, 1])
    return float(np.nanmax(np.abs(vals))) if vals else 0.0


def pcmci(cfg, saliency_dir=None):
    """Run PCMCI causal analysis using saliency-guided PCA1 features."""
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    from sklearn.decomposition import PCA
    from matplotlib import colormaps

    pc = cfg["viz"]["pcmci"]
    dc = cfg["data"]
    mc = cfg["model"]
    cache = Path(dc["cache_dir"])

    env = np.load(str(cache / "region_env.npy"))
    hurr = np.load(str(cache / "region_hurr.npy"))
    T, H, W, C = env.shape
    channels = dc["channel_names"]

    if saliency_dir is None:
        saliency_dir = cfg["viz"]["saliency"]["out_dir"]
    saliency_dir = Path(saliency_dir)

    sp_h, sp_w = mc.get("small_patch", [2, 2])
    nph, npw = H // sp_h, W // sp_w

    pcs, kept = [], []
    for ch, name in enumerate(channels):
        safe = re.sub(r"[^\w_]", "_", name)
        sal_path = saliency_dir / safe / "global_saliency.npy"
        if not sal_path.exists():
            print(f"[pcmci] WARN: missing saliency for {name}, skipping.")
            continue
        sal = np.load(sal_path)
        mask = (sal > 0.75).ravel()
        if mask.sum() < 3:
            thr = np.quantile(sal.ravel(), 0.80)
            mask = sal.ravel() >= thr
        if mask.sum() < 3:
            idx = np.argsort(sal.ravel())[::-1]
            mask = np.zeros_like(sal.ravel(), dtype=bool)
            mask[idx[:3]] = True

        arr = env[..., ch].reshape(T, nph, sp_h, npw, sp_w).mean(axis=(2, 4))
        sel = arr.reshape(T, -1)[:, mask]
        sel_z = (sel - sel.mean(0)) / (sel.std(0) + 1e-12)
        pc1 = PCA(n_components=1, random_state=0).fit_transform(sel_z).ravel()
        pcs.append(pc1.astype("float32"))
        kept.append(name)

    if len(pcs) == 0:
        print("[pcmci] No channels with saliency found. Aborting.")
        return

    # pre-genesis masking
    rh = np.where(np.isfinite(hurr), hurr, 0.0)
    genesis = rh.reshape(T, -1).max(axis=1) > 0
    mask_t = np.zeros(T, dtype=bool)
    for t in np.where(genesis)[0]:
        s = max(0, t - pc["pre_days"])
        if s < t:
            mask_t[s:t] = True

    X_raw = np.stack(pcs, axis=1)
    mu = np.nanmean(np.where(mask_t[:, None], X_raw, np.nan), axis=0)
    std = np.nanstd(np.where(mask_t[:, None], X_raw, np.nan), axis=0) + 1e-12
    X_std = (X_raw - mu) / std
    X_daily = X_std.copy()
    X_daily[~mask_t] = np.nan

    df_mask = ~np.isfinite(X_daily)
    X_filled = np.where(df_mask, 0.0, X_daily)
    df = pp.DataFrame(X_filled, var_names=kept, mask=df_mask,
                      remove_missing_upto_maxlag=True)

    tau_max = pc["tau_max"]
    sel_links = _prescreen(X_daily, r0=pc["r0_base"], tau_max=tau_max)
    link_assump = _make_links(sel_links, K=len(kept), tau_max=tau_max)

    engine = PCMCI(dataframe=df, cond_ind_test=ParCorr(significance="analytic"),
                   verbosity=0)
    res = engine.run_pcmci(
        tau_min=1, tau_max=tau_max,
        pc_alpha=pc["pc_alpha"],
        max_conds_dim=pc["max_conds_dim"],
        max_combinations=pc["max_combinations"],
        link_assumptions=link_assump,
    )
    qvals = engine.get_corrected_pvalues(res["p_matrix"], fdr_method="fdr_bh")
    val_mat = res["val_matrix"]

    _plot_graph(kept, val_mat, qvals, X_daily, tau_max, pc)
    print("[pcmci] Done.")


def _plot_graph(var_names, val_mat, q_mat, X_daily, tau_max, pc):
    from matplotlib import colormaps

    alpha_lo = pc["alpha_loose"]
    alpha_hi = pc["alpha_strict"]
    min_abs = pc["min_abs"]
    prefix = pc["fig_prefix"]

    matplotlib.rcParams.update({"font.size": 16, "axes.titlesize": 20})
    node_ac = {n: _max_acorr(X_daily[:, j], tau_max) for j, n in enumerate(var_names)}
    ac_norm = Normalize(vmin=0.0, vmax=0.8)
    ac_cmap = colormaps.get_cmap("YlOrRd")
    node_colors = [ac_cmap(ac_norm(node_ac[n])) for n in var_names]

    EDGE_MAX = 0.8
    rho_norm = Normalize(vmin=-EDGE_MAX, vmax=EDGE_MAX)
    rho_cmap = get_cmap("coolwarm")
    name2i = {n: i for i, n in enumerate(var_names)}

    def _pick(alpha):
        K = len(var_names)
        edges = {}
        for s in range(K):
            for t in range(K):
                if s == t:
                    continue
                cands = []
                for lag in range(1, tau_max + 1):
                    q, r = float(q_mat[s, t, lag]), float(val_mat[s, t, lag])
                    if np.isfinite(q) and np.isfinite(r) and q < alpha and abs(r) >= min_abs:
                        cands.append((lag, q, r))
                if cands:
                    lag, q, r = min(cands, key=lambda x: x[1])
                    edges[(var_names[s], var_names[t])] = {"lag": lag, "weight": r}
        return edges

    edges_lo = _pick(alpha_lo)
    edges_hi = {k: v for k, v in edges_lo.items()
                if float(q_mat[name2i[k[0]], name2i[k[1]], v["lag"]]) < alpha_hi}

    G_lo = nx.DiGraph(); G_lo.add_nodes_from(var_names)
    for (u, v), d in edges_lo.items():
        G_lo.add_edge(u, v, **d)
    G_hi = nx.DiGraph(); G_hi.add_nodes_from(var_names)
    for (u, v), d in edges_hi.items():
        G_hi.add_edge(u, v, **d)

    G_union = nx.compose(G_lo, G_hi)
    if not G_union.nodes:
        G_union.add_nodes_from(var_names)
    pos = nx.circular_layout(G_union)

    def _wrap(text, mx=12):
        words, lines, cur = text.split(), [], ""
        for w in words:
            if len(cur) + (1 if cur else 0) + len(w) <= mx:
                cur = f"{cur} {w}".strip()
            else:
                lines.append(cur); cur = w
        if cur:
            lines.append(cur)
        return "\n".join(lines)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    for ax, G, aval in zip(axes, (G_lo, G_hi), (alpha_lo, alpha_hi)):
        nx.draw_networkx_nodes(G, pos, node_size=[520] * len(G.nodes),
                               node_color=node_colors, linewidths=1.6,
                               edgecolors="white", ax=ax)
        if G.edges:
            ec = [rho_cmap(rho_norm(d["weight"])) for *_, d in G.edges(data=True)]
            ew = [3 + 6 * abs(d["weight"]) / EDGE_MAX for *_, d in G.edges(data=True)]
            nx.draw_networkx_edges(G, pos, edge_color=ec, width=ew, arrowsize=14,
                                   connectionstyle="arc3,rad=0.12", ax=ax)
        for n in G.nodes:
            x0, y0 = pos[n]
            txt = ax.text(x0, y0 - 0.18, _wrap(n), fontsize=11, ha="center",
                          va="center", zorder=6)
            txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
        for u, v, d in G.edges(data=True):
            ex, ey = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
            txt = ax.text(ex, ey, str(d["lag"]), fontsize=14, ha="center",
                          va="center", zorder=5)
            txt.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white")])
        ax.set_title(rf"$\alpha={aval}$", fontsize=20)
        ax.axis("off")

    fig.suptitle(rf"$\tau_{{\max}}={tau_max}$ days", fontsize=22, y=0.99)
    matplotlib.rcParams["svg.fonttype"] = "none"
    for ext in ("png", "svg"):
        out = f"{prefix}.{ext}"
        plt.savefig(out, dpi=600 if ext == "png" else None,
                    format=ext, bbox_inches="tight")
        print(f"[pcmci] Saved {out}")
    plt.close(fig)
