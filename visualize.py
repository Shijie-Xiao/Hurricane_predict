#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utilities:
  - spatial_main : quick spatial contour from a .npy array
  - pcmci_main   : PCMCI causal graph with pre-genesis masking and dual alpha thresholds

All arguments have sensible defaults so this module can run without external configs.

Examples
--------
# 1) Spatial: draw contours for a 2D array (e.g., a saved global_saliency.npy)
python visualize.py --task spatial --array path/to/global_saliency.npy

# 2) PCMCI: run full pipeline on region arrays (no ENSO split by default)
python visualize.py --task pcmci --env region_env.npy --hurr region_hurr.npy
"""

import os
import re
import argparse
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable
import matplotlib.patheffects as pe
import networkx as nx

from scipy.ndimage import gaussian_filter

# Tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# Basemap (optional but preferred)
try:
    from mpl_toolkits.basemap import Basemap
    HAS_BASEMAP = True
except Exception:
    HAS_BASEMAP = False

import utils


# ---------- Spatial quick plot ----------
def spatial_main(array_path: str,
                 title: str = "Spatial Map",
                 vmin: float = -1.0,
                 vmax: float = 1.0,
                 steps: int = 21,
                 out_path: str = "spatial.png",
                 lat0: int = 45, lat1: int = 85, lon0: int = 80, lon1: int = 180,
                 downsample: int = 2):
    """Draw a quick spatial contour map for a 2D npy array."""
    arr = np.load(array_path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    # optional pre-smoothing to avoid salt-and-pepper visualization
    arr = gaussian_filter(arr.astype(float), sigma=0.8)

    ds = arr[::downsample, ::downsample]
    levels = np.linspace(vmin, vmax, steps)

    H = lat1 - lat0
    W = lon1 - lon0
    bounds, grids = utils.geo_bounds_and_grid(H, W, region_lat0=lat0, region_lon0=lon0, ds_factor=downsample)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=220)
    if HAS_BASEMAP:
        m = Basemap(projection='cyl',
                    llcrnrlat=bounds[0], urcrnrlat=bounds[1],
                    llcrnrlon=bounds[2], urcrnrlon=bounds[3],
                    resolution='l', ax=ax)
        ax.set_facecolor('white')
        m.drawcoastlines(color='black', linewidth=0.3, zorder=2)
        cf = m.contourf(grids[0], grids[1], ds, levels=levels, cmap='RdBu_r', extend='both', alpha=0.95, zorder=1)
    else:
        X, Y = grids
        cf = ax.contourf(X, Y, ds, levels=levels, cmap='RdBu_r', extend='both', alpha=0.95)

    ax.set_title(title, fontsize=12)
    ax.axis('off')
    fig.colorbar(cf, ax=ax, orientation='vertical', fraction=0.04, pad=0.02)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[spatial] Saved -> {out_path}")


# ---------- PCMCI helpers ----------
def _fast_crosscorr(x, y, lag):
    """NaN-safe corr(x_t, y_{t-lag}), lag>=1."""
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
    """For each target j, collect parents (i,-lag) with |corr|>=r0."""
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


def _wrap_label(text: str, max_line_len=12):
    words = text.split(); lines, cur = [], ""
    for w in words:
        if len(cur) + (1 if cur else 0) + len(w) <= max_line_len:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)
    return "\n".join(lines)


def _soften(color_rgba, amt=0.55, alpha=0.98):
    rgb = np.array(color_rgba[:3])
    rgb_soft = 1.0 - amt * (1.0 - rgb)
    return (rgb_soft[0], rgb_soft[1], rgb_soft[2], alpha)


# ---------- PCMCI main ----------
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
    """
    PCMCI pipeline with pre-genesis masking and dual alpha visualization.
    If 'saliency_dir' doesn't exist, PCA1 is computed on all patches (no saliency filtering).
    """
    # Load data
    env = np.load(env_path)    # (T,40,100,C)
    hurr = np.load(hurr_path)  # (T,40,100)
    T, H, W, C = env.shape
    n_lat_p = H // sp_h
    n_lon_p = W // sp_w

    # Channel names for labeling; fall back to generic
    channel_names = utils.DEFAULT_SELECTED_9
    if len(channel_names) != C:
        channel_names = [f"Var{c}" for c in range(C)]

    # -------- Build per-channel PCA1 features (optionally saliency-guided) --------
    pcs_daily, kept_names, notes = [], [], []
    use_saliency = saliency_dir and os.path.isdir(saliency_dir)

    for ch_idx, name in enumerate(channel_names):
        arr = env[..., ch_idx]  # (T,H,W)
        arr_patch = arr.reshape(T, n_lat_p, sp_h, n_lon_p, sp_w).mean(axis=(2,4))  # (T, 20, 50)
        flat = arr_patch.reshape(T, -1)                                            # (T, 1000)

        if use_saliency:
            safe = re.sub(r"[^\w_]", "_", name)
            sal_path = os.path.join(saliency_dir, safe, "global_saliency.npy")
            if not os.path.exists(sal_path):
                # Fallback to no-saliency for this channel
                sel_mask = np.ones(flat.shape[1], dtype=bool)
                used = "no-saliency"
            else:
                sal_map = np.load(sal_path).astype(float)  # expect (20,50)
                if sal_map.shape != (n_lat_p, n_lon_p):
                    # Attempt rough resize if shapes mismatch
                    sal_map = sal_map
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
                sel_mask = mask_cells
        else:
            sel_mask = np.ones(flat.shape[1], dtype=bool)
            used = "no-saliency"

        sel = flat[:, sel_mask]                                    # (T, n_sel)
        sel_z = (sel - sel.mean(0)) / (sel.std(0) + 1e-12)

        # PCA1 per channel
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1, random_state=0)
        pc1 = pca.fit_transform(sel_z).ravel().astype('float32')
        evr = float(pca.explained_variance_ratio_[0])
        if evr < min_exp:
            pc1 = sel.mean(axis=1).astype('float32')
            used += f" | fallback=mean(ev={evr:.3f})"
        else:
            used += f" | PCA1(ev={evr:.3f})"

        pcs_daily.append(pc1)
        kept_names.append(name)
        notes.append((name, used, int(sel_mask.sum())))

    X_raw = np.stack(pcs_daily, axis=1).astype('float32')  # (T, K)

    # -------- Pre-genesis time mask (union of the last 'pre_days' before each genesis) --------
    rh = np.where(np.isfinite(hurr), hurr, 0.0)
    genesis_days = (rh.reshape(T, -1).max(axis=1) > 0).astype(bool)
    mask_time = np.zeros(T, dtype=bool)
    g_idx = np.where(genesis_days)[0]
    for t in g_idx:
        start = max(0, t - pre_days)
        if start < t:
            mask_time[start:t] = True

    # Standardize within mask; outside becomes NaN
    mu  = np.nanmean(np.where(mask_time[:, None], X_raw, np.nan), axis=0)
    std = np.nanstd (np.where(mask_time[:, None], X_raw, np.nan), axis=0) + 1e-12
    X_std = (X_raw - mu) / std
    X_daily = X_std.copy()
    X_daily[~mask_time, :] = np.nan
    var_names = kept_names

    print(f"[pcmci] variables: {var_names}")
    print(f"[pcmci] mask days kept={int(mask_time.sum())}/{T} ({100.0*mask_time.mean():.1f}%)")
    for n, u, ns in notes:
        print(f" - {n}: {u}, n_sel={ns}")

    # -------- Build Tigramite DataFrame with explicit missing mask --------
    df_mask   = ~np.isfinite(X_daily)
    X_filled  = np.where(df_mask, 0.0, X_daily)      # value ignored where mask=True
    df_daily = pp.DataFrame(
        X_filled, var_names=var_names, mask=df_mask,
        remove_missing_upto_maxlag=True
    )

    # -------- Pre-screen + PCMCI --------
    sel = _prescreen_links(X_daily, r0=r0_base, tau_max=tau_max)
    pc_total_est = _estimate_pc_tests(sel, max_conds_dim=max_conds_dim, max_combinations=max_combinations)
    link_assump = _to_link_assumptions(sel, K=X_daily.shape[1], tau_min=1, tau_max=tau_max)

    print(f"[pcmci] estimated PC tests: {pc_total_est:,}")
    pcmci = PCMCI(dataframe=df_daily, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)

    # Monkey-patch a progress bar over PCMCI condition iterations
    _orig_iter = PCMCI._iter_conditions
    pbar = None
    try:
        from tqdm.auto import tqdm as _tqdm
        pbar = _tqdm(total=int(pc_total_est), desc="PCMCI", unit="test")
        def iter_with_tqdm(self, parent, conds_dim, all_parents):
            for cond in _orig_iter(self, parent, conds_dim, all_parents):
                pbar.update(1)
                yield cond
        pcmci._iter_conditions = iter_with_tqdm.__get__(pcmci, PCMCI)
    except Exception:
        pass

    res = pcmci.run_pcmci(
        tau_min=1, tau_max=tau_max,
        pc_alpha=pc_alpha_base,
        max_conds_dim=max_conds_dim,
        max_combinations=max_combinations,
        link_assumptions=link_assump,
    )
    qvals = pcmci.get_corrected_pvalues(res["p_matrix"], fdr_method="fdr_bh")
    if pbar is not None:
        pbar.close()

    val_matrix = res["val_matrix"]
    q_matrix   = qvals

    # -------- Build two graphs (loose / strict) --------
    EDGE_ABS_MAX = 0.8
    rho_norm = Normalize(vmin=-EDGE_ABS_MAX, vmax=EDGE_ABS_MAX)
    rho_cmap = get_cmap('coolwarm')

    # node color: auto-corr
    node_ac  = {name: _max_abs_autocorr(X_daily[:, j], tau_max=tau_max) for j, name in enumerate(var_names)}
    ac_norm  = Normalize(vmin=0.0, vmax=0.8)
    ac_cmap  = get_cmap('YlOrRd')
    node_cols = [ac_cmap(ac_norm(node_ac[n])) for n in var_names]

    # pick edges per (alpha, min_abs)
    def _pick_edges(var_names, q, val, tau_max, alpha, min_abs):
        edges = {}
        K = len(var_names)
        for s in range(K):
            for t in range(K):
                if s == t: 
                    continue
                best = None
                for lag in range(1, tau_max+1):
                    qv = float(q[s, t, lag]); rv = float(val[s, t, lag])
                    if not (np.isfinite(qv) and np.isfinite(rv)):
                        continue
                    if qv < alpha and abs(rv) >= min_abs:
                        if best is None or qv < best[1]:
                            best = (lag, qv, rv)
                if best is not None:
                    edges[(var_names[s], var_names[t])] = {
                        'lag': int(best[0]), 'q': float(best[1]), 'weight': float(best[2])
                    }
        return edges

    edges_loose  = _pick_edges(var_names, q_matrix, val_matrix, tau_max, alpha_loose,  min_abs)
    edges_strict = _pick_edges(var_names, q_matrix, val_matrix, tau_max, alpha_strict, min_abs)

    G_left = nx.DiGraph();  G_left.add_nodes_from(var_names)
    for (u, v), d in edges_loose.items():   G_left.add_edge(u, v, **d)
    G_right = nx.DiGraph(); G_right.add_nodes_from(var_names)
    for (u, v), d in edges_strict.items():  G_right.add_edge(u, v, **d)

    # -------- Draw two-panel figure --------
    mpl = matplotlib
    mpl.rcParams.update({"font.size": 14, "axes.titlesize": 18})

    pos = nx.circular_layout(G_left if len(G_right)==0 else nx.compose(G_left, G_right))

    def _edge_draw(G, ax):
        e_cols = [_soften(rho_cmap(rho_norm(d['weight'])), amt=0.55, alpha=0.98) for *_ , d in G.edges(data=True)]
        e_widths = [3.0 + 6.0*abs(d['weight'])/EDGE_ABS_MAX for *_ , d in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, edge_color=e_cols, width=e_widths,
                               arrowsize=14, connectionstyle='arc3,rad=0.12', ax=ax)
        # edge labels = lag
        for (u, v, d) in G.edges(data=True):
            (x1, y1) = pos[u]; (x2, y2) = pos[v]
            lx, ly = 0.5*(x1+x2), 0.5*(y1+y2)
            txt = ax.text(lx, ly, f"{d['lag']}", fontsize=12, color='black', ha='center', va='center', zorder=5)
            txt.set_path_effects([pe.withStroke(linewidth=1.6, foreground='white')])

    labels_wrapped = {n: _wrap_label(n, 12) for n in var_names}

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    for ax, G, name in zip(axes, (G_left, G_right), (f"α={alpha_loose}", f"α={alpha_strict}")):
        nx.draw_networkx_nodes(G, pos, node_size=[520]*len(G.nodes),
                               node_color=[ac_cmap(ac_norm(node_ac[n])) for n in G.nodes],
                               linewidths=1.6, edgecolors='white', ax=ax)
        _edge_draw(G, ax)
        nx.draw_networkx_labels(G, pos, labels=labels_wrapped, font_size=14, ax=ax)
        ax.set_title(name); ax.axis('off')

    fig.suptitle(rf"$\tau_{{\max}}={tau_max}$ days", fontsize=20, y=0.98)

    # colorbars
    sm_edges = ScalarMappable(cmap=rho_cmap, norm=rho_norm); sm_edges.set_array([])
    sm_nodes = ScalarMappable(cmap=ac_cmap,  norm=ac_norm);  sm_nodes.set_array([])
    cax1 = fig.add_axes([0.25, 0.08, 0.5, 0.025])
    cax2 = fig.add_axes([0.25, 0.03, 0.5, 0.025])
    cb1 = fig.colorbar(sm_edges, cax=cax1, orientation='horizontal'); cb1.set_label("Conditional Cross-correlation")
    cb2 = fig.colorbar(sm_nodes, cax=cax2, orientation='horizontal'); cb2.set_label("Autocorrelation")

    # save
    out_png = f"{fig_prefix}_NaNmask_STRICT.png"
    out_svg = f"{fig_prefix}_NaNmask_STRICT.svg"
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"[pcmci] Saved -> {out_png} and {out_svg}")


# ---------- CLI ----------
def _build_parser():
    p = argparse.ArgumentParser(description="Visualization utilities (spatial / pcmci).")
    p.add_argument("--task", type=str, choices=["spatial", "pcmci"], required=True)

    # spatial
    p.add_argument("--array", type=str, help="2D .npy array for spatial plot")
    p.add_argument("--title", type=str, default="Spatial Map")
    p.add_argument("--vmin", type=float, default=-1.0)
    p.add_argument("--vmax", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=21)
    p.add_argument("--out", type=str, default="spatial.png")
    p.add_argument("--lat0", type=int, default=45)
    p.add_argument("--lat1", type=int, default=85)
    p.add_argument("--lon0", type=int, default=80)
    p.add_argument("--lon1", type=int, default=180)
    p.add_argument("--downsample", type=int, default=2)

    # pcmci
    p.add_argument("--env", type=str, default="region_env.npy")
    p.add_argument("--hurr", type=str, default="region_hurr.npy")
    p.add_argument("--saliency_dir", type=str, default="runs_FULL_9/run_01/maps_9Channels_3")
    p.add_argument("--sp_h", type=int, default=2)
    p.add_argument("--sp_w", type=int, default=2)
    p.add_argument("--th_sal", type=float, default=0.75)
    p.add_argument("--fallback_topq", type=float, default=0.80)
    p.add_argument("--min_exp", type=float, default=0.05)
    p.add_argument("--pre_days", type=int, default=7)
    p.add_argument("--tau_max", type=int, default=7)
    p.add_argument("--r0_base", type=float, default=0.05)
    p.add_argument("--pc_alpha_base", type=float, default=0.1)
    p.add_argument("--alpha_loose", type=float, default=0.05)
    p.add_argument("--alpha_strict", type=float, default=0.0001)
    p.add_argument("--min_abs", type=float, default=0.05)
    p.add_argument("--max_conds_dim", type=int, default=2)
    p.add_argument("--max_combinations", type=int, default=100000)
    p.add_argument("--fig_prefix", type=str, default="PCMCI_PRE7MASK")
    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.task == "spatial":
        if not args.array:
            raise SystemExit("--array is required for task=spatial")
        spatial_main(
            array_path=args.array, title=args.title, vmin=args.vmin, vmax=args.vmax,
            steps=args.steps, out_path=args.out, lat0=args.lat0, lat1=args.lat1,
            lon0=args.lon0, lon1=args.lon1, downsample=args.downsample
        )
    else:
        pcmci_main(
            env_path=args.env, hurr_path=args.hurr, saliency_dir=args.saliency_dir,
            sp_h=args.sp_h, sp_w=args.sp_w, th_sal=args.th_sal, fallback_topq=args.fallback_topq,
            min_exp=args.min_exp, pre_days=args.pre_days, tau_max=args.tau_max,
            r0_base=args.r0_base, pc_alpha_base=args.pc_alpha_base,
            alpha_loose=args.alpha_loose, alpha_strict=args.alpha_strict,
            min_abs=args.min_abs, max_conds_dim=args.max_conds_dim,
            max_combinations=args.max_combinations, fig_prefix=args.fig_prefix
        )


if __name__ == "__main__":
    main()
