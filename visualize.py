#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utilities:
  - spatial_main           : quick spatial contour from a .npy array
  - pcmci_main             : PCMCI causal graph with pre-genesis masking and dual alpha thresholds
  - compute_and_save_saliency : Integrated Gradients saliency maps (global + per-patch) for all channels
  - saliency_entrypoint    : build a minimal val loader and run saliency (for run.py spatial integration)
"""

import os
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import networkx as nx

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from scipy.ndimage import gaussian_filter
from mpl_toolkits.basemap import Basemap

import torch
from torch.utils.data import DataLoader, random_split

import utils

# ---------------- Config defaults for saliency ----------------
IG_STEPS = 20
DOWNSAMPLE_FACTOR = 2

# Channel name lists (use these instead of utils defaults for saliency naming)
CHANNEL_NAMES_9 = [
    "Potential Intensity",
    "Specific Humidity (850mb)",
    "Vertical Shear",
    "Relative Vorticity (500mb)",
    "Relative Vorticity (850mb)",
    "Saturation Deficit (500mb)",
    "Surface Radiation",
    "Cloud Cover (500mb)",
    "Ocean Heat Content",
]

CHANNEL_NAMES_28 = [
    "1000 cloud cover","1000 Geopotential","1000 Vorticity (relative)","1000 Specific humidity",
    "850 cloud cover","850 Geopotential","850 Vorticity (relative)","850 Specific humidity",
    "850 Temperature","500 cloud cover","500 Vorticity (relative)","Vertical Shear",
    "Potential Intensity","Sea surface temperature","Surface pressure","Charnock (Wave Drag)",
    "Surface radiation","Convective precipitation","Total column rain water","500 Divergence",
    "850 Divergence","1000 Divergence","ENSO","MJO","Salinity","OHC",
    "850 Saturation Deficit","500 Saturation Deficit"
]


# ============================================================
# =============== Quick Spatial Contour (2D) =================
# ============================================================
def _save_contour(fig_path: str,
                  data_ds: np.ndarray,
                  title: str,
                  geo_bounds,
                  grids,
                  levels):
    """Save a contour figure with Basemap."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    m = Basemap(projection='cyl',
                llcrnrlat=geo_bounds[0], urcrnrlat=geo_bounds[1],
                llcrnrlon=geo_bounds[2], urcrnrlon=geo_bounds[3],
                resolution='i', ax=ax)
    ax.set_facecolor('white')
    m.drawcoastlines(color='black', linewidth=0.3, zorder=2)
    cf = m.contourf(grids[0], grids[1], data_ds, levels=levels,
                    cmap='RdBu_r', extend='both', alpha=0.9, zorder=1)
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    fig.colorbar(cf, ax=ax, orientation='vertical', fraction=0.04, pad=0.02)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def spatial_main(array_path: str,
                 title: str = "Spatial Map",
                 vmin: float = -1.0,
                 vmax: float = 1.0,
                 steps: int = 21,
                 out_path: str = "spatial.png",
                 lat0: int = 45, lat1: int = 85, lon0: int = 80, lon1: int = 180,
                 downsample: int = 2):
    """Plot a 2D .npy array as a quick contour map."""
    arr = np.load(array_path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    ds = arr[::downsample, ::downsample]
    levels = np.linspace(vmin, vmax, steps)

    H = lat1 - lat0
    W = lon1 - lon0
    bounds, grids = utils.geo_bounds_and_grid(H, W, region_lat0=lat0, region_lon0=lon0, ds_factor=downsample)

    _save_contour(
        fig_path=out_path,
        data_ds=ds,
        title=title,
        geo_bounds=bounds,
        grids=grids,
        levels=levels
    )
    print(f"[spatial] Saved -> {out_path}")


# ============================================================
# ======================= PCMCI Utils ========================
# ============================================================
def _fast_crosscorr(x, y, lag):
    if lag >= len(x):
        return 0.0
    a = x[lag:]; b = y[:-lag]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return 0.0
    a = (a[m]-a[m].mean())/(a[m].std()+1e-12)
    b = (b[m]-b[m].mean())/(b[m].std()+1e-12)
    return float(np.corrcoef(a, b)[0, 1])


def _prescreen_links(X, r0, tau_max):
    K = X.shape[1]
    sel = {j: [] for j in range(K)}
    for j in range(K):
        for i in range(K):
            if i == j:
                continue
            for lag in range(1, tau_max+1):
                r = _fast_crosscorr(X[:, j], X[:, i], lag)
                if abs(r) >= r0:
                    sel[j].append((i, -lag))
    return sel


def _estimate_pc_tests(selected_links, max_conds_dim, max_combinations):
    from math import comb
    total = 0
    for _, parents in selected_links.items():
        k = len(parents)
        if k <= 0:
            continue
        for _ in parents:
            for d in range(1, max_conds_dim+1):
                total += min(comb(max(k-1, 0), d), max_combinations)
    return int(total)


def _to_link_assumptions(selected_links_dict, K, tau_min=1, tau_max=1):
    la = {j: {} for j in range(K)}
    for j in range(K):
        parents = selected_links_dict.get(j, [])
        for (i, tau) in parents:
            if i == j:
                continue
            if not (-tau_max <= tau <= -tau_min):
                continue
            la[j][(i, tau)] = '-?>'
    return la


def _max_abs_autocorr(x, tau_max=5):
    vals = []
    for lag in range(1, tau_max+1):
        if lag >= len(x): break
        a = x[lag:]; b = x[:-lag]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 5: continue
        aa = (a[m]-a[m].mean())/(a[m].std()+1e-12)
        bb = (b[m]-b[m].mean())/(b[m].std()+1e-12)
        vals.append(np.corrcoef(aa, bb)[0, 1])
    return float(np.nanmax(np.abs(vals))) if len(vals)>0 else 0.0


# ============================================================
# ======================= PCMCI Main =========================
# ============================================================
def pcmci_main(env_path="region_env.npy",
               hurr_path="region_hurr.npy",
               saliency_dir="runs_FULL_9/run_01/maps_9Channels_3",
               sp_h=2, sp_w=2,
               th_sal=0.75, fallback_topq=0.80, min_exp=0.05,
               pre_days=7,
               tau_max=7, r0_base=0.05, pc_alpha_base=0.1,
               alpha_loose=0.05, alpha_strict=0.0001, min_abs=0.05,
               max_conds_dim=2, max_combinations=100000,
               fig_prefix="PCMCI_PRE7MASK"):
    """Minimal, configurable PCMCI pipeline."""
    # Load data
    env = np.load(env_path)    # (T,40,100,C)
    hurr = np.load(hurr_path)  # (T,40,100)
    T, H, W, C = env.shape
    n_lat_p = H // sp_h
    n_lon_p = W // sp_w

    # Channel names
    channel_names = utils.DEFAULT_SELECTED_9 if C == 9 else [f"Var{c}" for c in range(C)]

    # Build per-channel PCA1 features guided by saliency
    pcs_daily = []
    kept_names = []

    for ch_idx, name in enumerate(channel_names):
        safe = re.sub(r"[^\w_]", "_", name)
        sal_path = os.path.join(saliency_dir, safe, "global_saliency.npy")
        if not os.path.exists(sal_path):
            raise FileNotFoundError(f"Missing saliency for {name}: {sal_path}")

        sal_map = np.load(sal_path)  # expected (20,50) after downsampling
        mask_cells = (sal_map > th_sal).ravel()
        n_sel = int(mask_cells.sum()); used = "th"
        if n_sel < 3:
            thr = np.quantile(sal_map.ravel(), fallback_topq)
            mask_cells = (sal_map.ravel() >= thr)
            n_sel = int(mask_cells.sum()); used=f"top{int((1-fallback_topq)*100)}%"
        if n_sel < 3:
            idx_sorted = np.argsort(sal_map.ravel())[::-1]
            mask_cells = np.zeros_like(sal_map.ravel(), bool); mask_cells[idx_sorted[:3]] = True
            n_sel = 3; used="top3-cells"

        arr = env[..., ch_idx]  # (T,H,W)
        arr_patch = arr.reshape(T, n_lat_p, sp_h, n_lon_p, sp_w).mean(axis=(2,4))
        sel = arr_patch.reshape(T, -1)[:, mask_cells]  # (T,n_sel)
        sel_z = (sel - sel.mean(0)) / (sel.std(0) + 1e-12)

        # PCA1
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1, random_state=0)
        pc1 = pca.fit_transform(sel_z).ravel()
        pcs_daily.append(pc1.astype('float32'))
        kept_names.append(name)

    # === Pre-genesis mask (last `pre_days` prior to each genesis) ===
    rh = np.where(np.isfinite(hurr), hurr, 0.0)
    genesis_days = (rh.reshape(T,-1).max(axis=1) > 0).astype(bool)
    mask_time = np.zeros(T, dtype=bool)
    g_idx = np.where(genesis_days)[0]
    for t in g_idx:
        start = max(0, t - pre_days)
        if start < t:
            mask_time[start:t] = True

    # Standardize within mask
    X_raw = np.stack(pcs_daily, axis=1).astype('float32')  # (T,C)
    mu  = np.nanmean(np.where(mask_time[:,None], X_raw, np.nan), axis=0)
    std = np.nanstd (np.where(mask_time[:,None], X_raw, np.nan), axis=0) + 1e-12
    X_std = (X_raw - mu) / std
    X_daily = X_std.copy()
    X_daily[~mask_time, :] = np.nan
    var_names = kept_names

    # PCMCI dataframe with explicit mask
    df_mask   = ~np.isfinite(X_daily)
    X_filled  = np.where(df_mask, 0.0, X_daily)
    df_daily = pp.DataFrame(
        X_filled,
        var_names=var_names,
        mask=df_mask,
        remove_missing_upto_maxlag=True
    )

    # Prescreen & PCMCI
    tau_max_used = 7
    sel = _prescreen_links(X_daily, r0=r0_base, tau_max=tau_max_used)
    pc_total_est = _estimate_pc_tests(sel, max_conds_dim=max_conds_dim, max_combinations=max_combinations)
    link_assump  = _to_link_assumptions(sel, K=X_daily.shape[1], tau_min=1, tau_max=tau_max_used)

    pcmci = PCMCI(
        dataframe=df_daily,
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=0,
    )
    res = pcmci.run_pcmci(
        tau_min=1, tau_max=tau_max_used,
        pc_alpha=pc_alpha_base,
        max_conds_dim=max_conds_dim,
        max_combinations=max_combinations,
        link_assumptions=link_assump,
    )
    qvals = pcmci.get_corrected_pvalues(res["p_matrix"], fdr_method="fdr_bh")

    val_matrix = res["val_matrix"]
    q_matrix   = qvals

    # ===== Network drawing with dual alpha thresholds =====
    import matplotlib.patheffects as pe
    import matplotlib as mpl
    from matplotlib import colormaps

    mpl.rcParams.update({"font.size": 16, "axes.titlesize": 20})

    def max_abs_autocorr(x, tau_max=5):
        return _max_abs_autocorr(x, tau_max=tau_max)

    node_ac = {name: max_abs_autocorr(X_daily[:, j], tau_max=tau_max_used)
               for j, name in enumerate(var_names)}
    ac_norm = Normalize(vmin=0.0, vmax=0.8)
    ac_cmap = colormaps.get_cmap('YlOrRd')
    node_colors = [ac_cmap(ac_norm(node_ac[n])) for n in var_names]

    alpha_loose, alpha_strict = 0.05, 0.0001
    EDGE_ABS_MAX = 0.8
    rho_norm = Normalize(vmin=-EDGE_ABS_MAX, vmax=EDGE_ABS_MAX)
    rho_cmap = get_cmap('coolwarm')

    name_to_idx = {n:i for i,n in enumerate(var_names)}

    def pick_display_edges(var_names, q, val, tau_max, alpha_loose, min_abs, rule='min_q'):
        edges = {}
        K = len(var_names)
        for s in range(K):
            for t in range(K):
                if s == t: 
                    continue
                cands = []
                for lag in range(1, tau_max+1):
                    qv = float(q[s, t, lag])
                    rv = float(val[s, t, lag])
                    if (not np.isfinite(qv)) or (not np.isfinite(rv)): 
                        continue
                    if (qv < alpha_loose) and (abs(rv) >= min_abs):
                        cands.append((lag, qv, rv))
                if not cands:
                    continue
                if rule == 'max_abs_r':
                    lag_sel, q_sel, r_sel = max(cands, key=lambda x: abs(x[2]))
                else:
                    lag_sel, q_sel, r_sel = min(cands, key=lambda x: x[1])
                u = var_names[s]; v = var_names[t]
                edges[(u, v)] = {'lag': int(lag_sel), 'q': float(q_sel), 'weight': float(r_sel)}
        return edges

    display_edges = pick_display_edges(var_names, q_matrix, val_matrix,
                                       tau_max=tau_max_used, alpha_loose=alpha_loose,
                                       min_abs=0.05, rule='min_q')

    G_left = nx.DiGraph();  G_left.add_nodes_from(var_names)
    for (u, v), d in display_edges.items():
        G_left.add_edge(u, v, lag=d['lag'], weight=d['weight'])

    G_right = nx.DiGraph(); G_right.add_nodes_from(var_names)
    for (u, v), d in display_edges.items():
        s, t = name_to_idx[u], name_to_idx[v]
        lag  = d['lag']
        qv   = float(q_matrix[s, t, lag]); rv = float(val_matrix[s, t, lag])
        if (np.isfinite(qv) and np.isfinite(rv)
            and (qv < alpha_strict) and (abs(rv) >= 0.05)):
            G_right.add_edge(u, v, lag=lag, weight=rv)

    assert set(G_right.edges()).issubset(set(G_left.edges()))

    def soften(color_rgba, amt=0.55, alpha=0.98):
        rgb = np.array(color_rgba[:3])
        rgb_soft = 1.0 - amt * (1.0 - rgb)
        return (rgb_soft[0], rgb_soft[1], rgb_soft[2], alpha)

    def wrap_label(text: str, max_line_len=12):
        words = text.split(); lines, cur = [], ""
        for w in words:
            if len(cur) + (1 if cur else 0) + len(w) <= max_line_len:
                cur = f"{cur} {w}".strip()
            else:
                lines.append(cur); cur = w
        if cur: lines.append(cur)
        return "\n".join(lines)

    def autosize(text: str, base=18, min_size=14):
        L = len(text.replace("\n",""))
        if L <= 14: return base
        if L <= 20: return base - 1
        if L <= 28: return base - 2
        return max(min_size, base - 3)

    def compute_node_label_positions(pos, label_pos_push=0.16, min_dist=0.060, step=0.014, max_iter=100):
        xy = np.array(list(pos.values())); center = xy.mean(axis=0)
        used, out = [], {}
        for n, p in pos.items():
            p = np.array(p); vec = p - center
            nrm = np.linalg.norm(vec)
            vec = np.array([1.0,0.0]) if nrm<1e-12 else vec/nrm
            cur = p + label_pos_push * vec
            safe, iters = False, 0
            while not safe and iters < max_iter:
                safe = True
                for uxy in used:
                    if np.linalg.norm(cur - uxy) < min_dist:
                        cur = cur + step * vec; safe = False; break
                iters += 1
            out[n] = (float(cur[0]), float(cur[1])); used.append(cur)
        return out

    def compute_edge_label_positions(G, pos, label_pos=0.5, base_push=0.025, min_dist=0.055, step=0.014):
        xy = np.array(list(pos.values())); center = xy.mean(axis=0)
        used, out = [], {}
        for (u, v, d) in G.edges(data=True):
            p1 = np.array(pos[u]); p2 = np.array(pos[v])
            m  = p1 + label_pos * (p2 - p1)
            vec = m - center; nrm = np.linalg.norm(vec)
            vec = vec / (nrm + 1e-12)
            cur = m + base_push * vec
            safe = False; iters = 0
            while not safe and iters < 100:
                safe = True
                for uxy in used:
                    if np.linalg.norm(cur - uxy) < min_dist:
                        cur = cur + step * vec; safe = False; break
                iters += 1
            out[(u, v)] = (float(cur[0]), float(cur[1])); used.append(cur)
        return out

    G_union = nx.compose(G_left, G_right)
    if len(G_union.nodes) == 0: G_union.add_nodes_from(var_names)
    pos = nx.circular_layout(G_union)

    labels_full_wrapped = {n: wrap_label(n, max_line_len=12) for n in var_names}
    node_label_pos = compute_node_label_positions(pos)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300, constrained_layout=False)
    STROKE_W  = 1.2
    FONT_EDGE = 16

    for ax, G, title_alpha in zip(axes, (G_left, G_right), (alpha_loose, alpha_strict)):
        nx.draw_networkx_nodes(G, pos, node_size=[520]*len(G.nodes),
                               node_color=node_colors, linewidths=1.6, edgecolors='white', ax=ax)

        e_cols_raw = [rho_cmap(rho_norm(d['weight'])) for *_ , d in G.edges(data=True)]
        e_colors   = [soften(c, amt=0.55, alpha=0.98) for c in e_cols_raw]
        e_widths   = [3.0 + 6.0*abs(d['weight'])/EDGE_ABS_MAX for *_ , d in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, edge_color=e_colors, width=e_widths,
                               arrowsize=14, connectionstyle='arc3,rad=0.12', ax=ax)

        for n in G.nodes:
            x0, y0 = pos[n]; lx, ly = node_label_pos[n]
            label  = labels_full_wrapped[n]
            fs     = autosize(label)
            import matplotlib.patheffects as pe
            txt = ax.text(lx, ly, label, fontsize=fs, color='black', ha='center', va='center', zorder=6, linespacing=1.0)
            txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground='white')])
            ax.plot([x0, lx], [y0, ly], lw=1.2, color='gray', alpha=0.5, zorder=4)

        custom_pos = compute_edge_label_positions(G, pos)
        for (u, v, d) in G.edges(data=True):
            lx, ly = custom_pos[(u, v)]
            txt = ax.text(lx, ly, f"{d['lag']}", fontsize=FONT_EDGE, color='black', ha='center', va='center', zorder=5)
            txt.set_path_effects([pe.withStroke(linewidth=STROKE_W, foreground='white')])

        ax.set_title(rf"$\alpha={title_alpha}$", fontsize=20)
        ax.axis('off')

    fig.suptitle(rf"$\tau_{{\max}}={tau_max_used}$ days", fontsize=22, y=0.988)

    # Save
    import matplotlib as mpl
    mpl.rcParams['svg.fonttype'] = 'none'
    png_path = f"{fig_prefix}_NaNmask_STRICT_BIGFONT_refCB.png"
    svg_path = f"{fig_prefix}_NaNmask_STRICT_BIGFONT_refCB.svg"
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"[pcmci] Saved: {png_path} and {svg_path}")


# ============================================================
# ============== Integrated Gradients Saliency ===============
# ============================================================
def _make_geo_grid_from_hw(Hs: int, Ws: int, ds_factor: int = DOWNSAMPLE_FACTOR):
    """Infer NA region bounds from size and create grid."""
    # This assumes region is (40,100) from (lat0,lon0) = (45,80)
    lat0, lon0 = 45, 80
    bounds, grids = utils.geo_bounds_and_grid(Hs, Ws, region_lat0=lat0, region_lon0=lon0, ds_factor=ds_factor)
    return bounds, grids


def compute_and_save_saliency(run_dir: Path,
                              best_weight_path: Path,
                              val_loader: DataLoader,
                              device: torch.device,
                              in_ch: int,
                              channel_names=None):
    """
    Compute Integrated Gradients saliency maps (global + per-patch) for each channel.
    Saves both .png and .npy for:
      - global_saliency
      - patch_{i}_{j}
    Output root: run_dir / maps_<X>Channels_3/<ChannelNameSanitized>/
    """
    from model import RevisedHierarchicalPatchTimesformer

    # Rebuild model and load weights
    model = RevisedHierarchicalPatchTimesformer(in_ch=in_ch).to(device)
    model.load_state_dict(torch.load(best_weight_path, map_location=device))
    model.eval()

    # Probe H,W and n_lat,n_lon
    sample_x, y = next(iter(val_loader))
    Hs, Ws = sample_x.shape[-2:]
    with torch.no_grad():
        _, _, n_lat, n_lon = model(sample_x[:1].to(device)).shape

    geo_bounds, grids = _make_geo_grid_from_hw(Hs, Ws, ds_factor=DOWNSAMPLE_FACTOR)
    levels = np.linspace(-1, 1, 21)

    tag_ch = f"{in_ch}Channels_3"
    base_dir = run_dir / f"maps_{tag_ch}"
    base_dir.mkdir(parents=True, exist_ok=True)

    # IG setup
    alphas = torch.linspace(0.0, 1.0, IG_STEPS, device=device)

    # channel names
    if channel_names is None:
        channel_names = CHANNEL_NAMES_9 if in_ch == 9 else (CHANNEL_NAMES_28 if in_ch == 28 else [f"Ch{c}" for c in range(in_ch)])

    for c, name in enumerate(channel_names):
        safe_name = re.sub(r'[^\w_]', '_', name)
        print(f"[{run_dir.name}] Saliency - Channel {c}: {name}")

        patch_sum = {(i,j): np.zeros((Hs, Ws), dtype=np.float32) for i in range(n_lat) for j in range(n_lon)}
        patch_count = {(i,j): 0 for i in range(n_lat) for j in range(n_lon)}
        global_sum = np.zeros((Hs, Ws), dtype=np.float32)
        global_count = 0

        for x_batch, y_batch in val_loader:
            B = x_batch.shape[0]
            x_batch = x_batch.to(device)                      # (B,T,C,H,W)
            baseline = torch.zeros_like(x_batch)
            diff = x_batch - baseline

            # ----- global IG -----
            total_grad = torch.zeros_like(x_batch)
            for alpha in alphas:
                inp = baseline + alpha * diff
                inp.requires_grad_(True)
                logits = model(inp)                           # (B, T, n_lat, n_lon)
                score = logits.sum(dim=(1,2,3))               # (B,)
                model.zero_grad()
                score.sum().backward()
                total_grad += inp.grad
            ig_global = diff * total_grad / IG_STEPS
            sal_global = ig_global[:,:,c,:,:].abs().mean(dim=1).detach().cpu().numpy()  # (B,H,W)
            global_sum += sal_global.sum(axis=0)
            global_count += B

            # ----- patch-specific IG -----
            for i in range(n_lat):
                for j in range(n_lon):
                    total_grad_p = torch.zeros_like(x_batch)
                    for alpha in alphas:
                        inp = baseline + alpha * diff
                        inp.requires_grad_(True)
                        out = model(inp)[:,:,i,j].sum(dim=(1,))
                        model.zero_grad()
                        out.sum().backward()
                        total_grad_p += inp.grad
                    ig_patch = diff * total_grad_p / IG_STEPS
                    sal_patch = ig_patch[:,:,c,:,:].abs().mean(dim=1).detach().cpu().numpy()  # (B,H,W)
                    patch_sum[(i,j)] += sal_patch.sum(axis=0)
                    patch_count[(i,j)] += B

        # save patch-specific maps (image + npy)
        chan_dir = base_dir / safe_name
        chan_dir.mkdir(parents=True, exist_ok=True)

        for (i,j), sum_map in patch_sum.items():
            if patch_count[(i,j)] == 0:
                continue
            mean_patch = sum_map / patch_count[(i,j)]
            sm = gaussian_filter(mean_patch, sigma=1.0)
            ds = sm[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR]
            ma = float(np.max(np.abs(ds))) if np.any(np.isfinite(ds)) else 1.0
            if ma == 0: ma = 1.0
            norm_patch = ds / ma

            np.save(chan_dir / f'patch_{i}_{j}.npy', norm_patch)
            _save_contour(
                fig_path=str(chan_dir / f'patch_{i}_{j}.png'),
                data_ds=norm_patch,
                title=f'{name} – Patch ({i},{j})',
                geo_bounds=geo_bounds,
                grids=grids,
                levels=levels
            )

        # save global map (image + npy)
        if global_count > 0:
            mean_global = global_sum / global_count
            smg = gaussian_filter(mean_global, sigma=1.0)
            dsg = smg[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR]
            mg = float(np.max(np.abs(dsg))) if np.any(np.isfinite(dsg)) else 1.0
            if mg == 0: mg = 1.0
            norm_global = dsg / mg

            np.save(chan_dir / 'global_saliency.npy', norm_global)
            _save_contour(
                fig_path=str(chan_dir / 'global_saliency.png'),
                data_ds=norm_global,
                title=name,
                geo_bounds=geo_bounds,
                grids=grids,
                levels=levels
            )

        print(f"[{run_dir.name}] Saliency done for channel: {name}\n")


# ============================================================
# ============= Entry to run saliency from run.py ============
# ============================================================
def saliency_entrypoint(ckpt_path: str,
                        run_dir: str,
                        env_path: str,
                        hurr_path: str,
                        seq_len: int = 14,
                        stride: int = 3,
                        batch_size: int = 4,
                        in_ch: int = None):
    """
    Build a light validation loader from env/hurr, then run IG saliency for all channels.
    This is designed to be triggered from `run.py spatial` when --saliency_ckpt is provided.
    """
    from train import HurricanePatchDataset  # reuse dataset logic

    env = np.load(env_path)
    T, H, W, C = env.shape
    if in_ch is None:
        in_ch = C

    # Build dataset and a small validation loader (10% split)
    full_ds = HurricanePatchDataset(
        env_path=env_path, hurr_path=hurr_path,
        seq_len=seq_len, stride=stride,
        in_ch=in_ch, H=H, W=W, patch_h=20, patch_w=20
    )

    n_total = len(full_ds)
    n_train = max(1, int(n_total*0.9))
    n_val   = max(1, n_total - n_train)
    _, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(2024))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_and_save_saliency(
        run_dir=Path(run_dir),
        best_weight_path=Path(ckpt_path),
        val_loader=val_loader,
        device=device,
        in_ch=in_ch,
        channel_names=(CHANNEL_NAMES_9 if in_ch == 9 else (CHANNEL_NAMES_28 if in_ch == 28 else None))
    )
