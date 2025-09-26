#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame-level heatmap modeling with U-Net on 20x20 patches.

Provides:
  - make_patches_20x20(...)             : generate ENV/HURR patches from region files
  - HurricaneFrameHeatmapDataset        : dataset (per-frame, per-patch) with gaussian-smoothed targets
  - UNetFrameLevel_20x20                : U-Net backbone
  - train_heatmap(...)                  : training loop
  - eval_heatmap_frames(...)            : evaluation & multi-radius frame visualization

All functions have sensible defaults so they run even without extra configs.
"""

from pathlib import Path
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from datetime import datetime, timedelta


# =========================
# ========== Patches ======
# =========================
def make_patches_20x20(
    region_env_path: str = "region_env.npy",
    region_hurr_path: str = "region_hurr.npy",
    out_env_patches: str = "ENV_PATCHES_20x20.npy",
    out_hurr_patches: str = "HURR_PATCHES_20x20.npy",
    patch_h: int = 20,
    patch_w: int = 20,
):
    """
    From region arrays: region_hurr (T,40,100), region_env (T,40,100,C)
    Produce:
      ENV_PATCHES_20x20.npy:  (T, P, 20, 20, C)
      HURR_PATCHES_20x20.npy: (T, P, 20, 20)
    """
    hurr = np.load(region_hurr_path)   # (T,40,100)
    env  = np.load(region_env_path)    # (T,40,100,C)
    assert hurr.ndim == 3 and env.ndim == 4, f"Unexpected shapes: {hurr.shape}, {env.shape}"
    T, H, W = hurr.shape
    _, H2, W2, C = env.shape
    assert (H, W) == (H2, W2), "HURR/ENV spatial mismatch"

    n_lat = H // patch_h     # 2
    n_lon = W // patch_w     # 5
    P = n_lat * n_lon

    # HURR -> (T,P,20,20)
    hurr_patches = (
        hurr
        .reshape(T, n_lat, patch_h, n_lon, patch_w)
        .transpose(0, 1, 3, 2, 4)
        .reshape(T, P, patch_h, patch_w)
    )

    # ENV -> (T,P,20,20,C)
    env_patches = (
        env
        .reshape(T, n_lat, patch_h, n_lon, patch_w, C)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(T, P, patch_h, patch_w, C)
    )

    np.save(out_hurr_patches, hurr_patches)
    np.save(out_env_patches,  env_patches)

    print("[patches] Saved:")
    print("  ", out_hurr_patches, "<-", hurr_patches.shape)
    print("  ", out_env_patches,  "<-", env_patches.shape)
    return out_env_patches, out_hurr_patches


# =========================
# ========== Dataset ======
# =========================
class HurricaneFrameHeatmapDataset(Dataset):
    """
    (frame, patch) → (C+2, 20, 20), y: gaussian-smoothed heatmap in [0,1]
    Adds 2D positional channels (x,y) normalized to [0,1].
    Keeps only frames where the target patch contains at least one hurricane pixel.
    """
    def __init__(self, env_path: str, labels_path: str, sigma: float = 1.0):
        env_all = np.nan_to_num(np.load(env_path),   nan=0.0, posinf=0.0, neginf=0.0)  # (T,P,20,20,C)
        lab_all = np.nan_to_num(np.load(labels_path),nan=0.0, posinf=0.0, neginf=0.0)  # (T,P,20,20)
        assert env_all.ndim == 5 and lab_all.ndim == 4, "Shapes must be (T,P,20,20,C) and (T,P,20,20)"
        T, P, H, W, C = env_all.shape
        assert lab_all.shape == (T, P, H, W)

        self.env = env_all.astype(np.float32)
        self.labels = lab_all.astype(np.float32)
        self.sigma = float(sigma)
        self.indices = []
        for t in range(T):
            for p in range(P):
                if self.labels[t, p].sum() > 0:
                    self.indices.append((t, p))

        # precompute pos grid
        yy, xx = np.meshgrid(
            np.linspace(0, 1, H),
            np.linspace(0, 1, W),
            indexing="ij"
        )
        self.pos = np.stack([xx, yy], axis=0).astype(np.float32)  # (2,H,W)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t, p = self.indices[idx]
        x = self.env[t, p].transpose(2, 0, 1)         # (C,20,20)
        orig = self.labels[t, p]                       # (20,20) binary
        heatmap = gaussian_filter(orig, sigma=self.sigma)
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max()).astype(np.float32)
        else:
            heatmap = heatmap.astype(np.float32)

        x = np.concatenate([x, self.pos], axis=0)      # (C+2,20,20)
        return torch.from_numpy(x), torch.from_numpy(heatmap)


# =========================
# =========== Model =======
# =========================
class ConvBlockDeepBN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(6):
            layers += [
                nn.Conv2d(ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            ch = out_ch
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetFrameLevel_20x20(nn.Module):
    """
    U-Net for small 20x20 patches. Uses encoder bottleneck with MLP for added capacity.
    """
    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlockDeepBN(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)  # 20->10

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.Conv2d(128,128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.Conv2d(128,128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.Conv2d(128,256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(inplace=True),
            nn.Conv2d(256,256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(inplace=True),
            nn.Conv2d(256,256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(inplace=True),
        )
        self.dropout2 = nn.Dropout(0.2)
        self.pool2 = nn.MaxPool2d(2)  # 10->5

        # Bottleneck MLP
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(5*5*256, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 5*5*512)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # 5->10
        self.dec1 = nn.Sequential(
            nn.Conv2d(512+256,256,3,padding=1), nn.GroupNorm(8,256), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),     nn.GroupNorm(8,256), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),     nn.GroupNorm(8,256), nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,padding=1),     nn.GroupNorm(8,128), nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),     nn.GroupNorm(8,128), nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),     nn.GroupNorm(8,128), nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 10->20
        self.dec2 = nn.Sequential(
            nn.Conv2d(64+64,64,3,padding=1), nn.GroupNorm(8,64), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),    nn.GroupNorm(8,64), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),    nn.GroupNorm(8,64), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),    nn.GroupNorm(8,64), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),    nn.GroupNorm(8,64), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),    nn.GroupNorm(8,64), nn.ReLU(inplace=True),
        )

        self.dropout_final = nn.Dropout(0.2)
        self.out_conv = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x)     # (B,64,20,20)
        p1 = self.pool1(c1)   # (B,64,10,10)
        c2 = self.enc2(p1)    # (B,256,10,10)
        c2 = self.dropout2(c2)
        p2 = self.pool2(c2)   # (B,256,5,5)

        b = self.flatten(p2)
        b = self.linear1(b)
        b = self.linear2(b)
        b = self.linear3(b)
        b = b.view(b.size(0), 512, 5, 5)

        u1 = self.up1(b)                  # (B,512,10,10)
        u1 = torch.cat([u1, c2], dim=1)
        d1 = self.dec1(u1)                # (B,128,10,10)

        u2 = self.up2(d1)                 # (B,64,20,20)
        u2 = torch.cat([u2, c1], dim=1)
        d2 = self.dec2(u2)                # (B,64,20,20)

        out = self.dropout_final(d2)
        out = self.out_conv(out)          # (B,1,20,20)
        return out.squeeze(1)             # (B,20,20)


# =========================
# ======== Training =======
# =========================
def train_heatmap(
    env_patches_path: str = "ENV_PATCHES_20x20.npy",
    hurr_patches_path: str = "HURR_PATCHES_20x20.npy",
    sigma: float = 1.5,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-4,
    val_split: float = 0.1,
    out_ckpt: str = "best_unet_heatmap.pt",
    seed: int = 1337,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = HurricaneFrameHeatmapDataset(env_patches_path, hurr_patches_path, sigma=sigma)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = max(1, len(dataset) - val_size)
    train_set, val_set = random_split(dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Infer channels
    sample_x, _ = next(iter(train_loader))
    in_ch = sample_x.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetFrameLevel_20x20(in_ch=in_ch, out_ch=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val = float("inf")
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_ckpt)

        print(f"[heatmap][Epoch {epoch:03d}] train={train_loss:.6f}  val={val_loss:.6f}  best={best_val:.6f}")

    print(f"[heatmap] Best checkpoint saved -> {out_ckpt}")
    return out_ckpt


# =========================
# ========== Eval =========
# =========================
def eval_heatmap_frames(
    best_patch_timesformer_ckpt: str,   # patch Timesformer ckpt (produces patch probs)
    best_unet_heatmap_ckpt: str,        # frame-level UNet ckpt (refines within predicted patch)
    val_loader_timesformer,             # (x_batch, _, y_patch_batch, start_idx_batch)
    seq_len: int = 7,
    region_lat0: int = 45,
    region_lon0: int = 80,
    n_lat_full: int = 90,
    n_lon_full: int = 180,
    out_dir: str = "frame_maps_with_global_radii",
    sample_frames: int = 10,
):
    """
    Reproduce your multi-radius evaluation visualization:
      - Compute min distances between predicted peak and all GT points
      - Report percentile radii & success rate
      - Draw Basemap panels with GT (red dots), Pred (black X), and colored radius rings
    """
    from model import RevisedHierarchicalPatchTimesformer

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build Timesformer (infer in_ch from loader)
    x0, _, _, _ = next(iter(val_loader_timesformer))
    in_ch_tf = x0.shape[2]
    model_tf = RevisedHierarchicalPatchTimesformer(in_ch=in_ch_tf).to(device)
    model_tf.load_state_dict(torch.load(best_patch_timesformer_ckpt, map_location=device))
    model_tf.eval()

    # Build UNet (infer in_ch from cropped patch + 2)
    # We'll form a sample patch to detect channel count
    _, _, _, _ = next(iter(val_loader_timesformer))
    # NOTE: in eval loop we assemble a patch crop on-the-fly, adding 2 positional channels
    # so here we just initialize with an upper bound; we will re-init if needed
    model_unet = None

    # constants
    PATCH_H, PATCH_W = 20, 20
    N_LAT_PATCHES, N_LON_PATCHES = 2, 5
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 1.00]
    colors     = ['gray', 'yellow', 'lime', 'orange', 'magenta', 'cyan', 'red']
    percents   = [int(th * 100) for th in thresholds]

    # helpers
    def normalize01(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    # distance accumulation
    min_dists = []

    start_date = datetime(2010, 1, 1)
    with torch.no_grad():
        for x_batch, _, y_patch_batch, _ in val_loader_timesformer:
            x_batch = x_batch.to(device)
            logits  = model_tf(x_batch)                       # (B,T,2,5)
            probs   = torch.sigmoid(logits).cpu().numpy()

            B, T = probs.shape[0], probs.shape[1]
            for bi in range(B):
                for t in range(seq_len):
                    flat = probs[bi, t].flatten() > 0.5
                    idxs = np.where(flat)[0]
                    if len(idxs) == 0:
                        continue
                    idx = idxs[0]
                    i_patch, j_patch = divmod(idx, N_LON_PATCHES)

                    # crop 20x20 patch & build (C+2,H,W)
                    patch_env = x_batch[bi, t,
                                       :,
                                       i_patch*PATCH_H:(i_patch+1)*PATCH_H,
                                       j_patch*PATCH_W:(j_patch+1)*PATCH_W
                                       ].detach().cpu().numpy()  # (C,20,20)

                    yy, xx = np.meshgrid(
                        np.linspace(0,1,PATCH_H),
                        np.linspace(0,1,PATCH_W),
                        indexing='ij'
                    )
                    inp = np.concatenate([patch_env, xx[None], yy[None]], axis=0)
                    if model_unet is None:
                        in_ch_unet = inp.shape[0]
                        model_unet = UNetFrameLevel_20x20(in_ch=in_ch_unet, out_ch=1).to(device)
                        model_unet.load_state_dict(torch.load(best_unet_heatmap_ckpt, map_location=device))
                        model_unet.eval()

                    xh   = torch.from_numpy(inp).unsqueeze(0).float().to(device)
                    hmap = model_unet(xh).squeeze(0).detach().cpu().numpy()
                    hmap = normalize01(hmap)
                    dy, dx = np.unravel_index(hmap.argmax(), hmap.shape)

                    gi_pred = region_lat0 + i_patch*PATCH_H + dy
                    gj_pred = region_lon0 + j_patch*PATCH_W + dx

                    # collect GT points across all patches at this time
                    gt_pts = []
                    for i in range(N_LAT_PATCHES):
                        for j in range(N_LON_PATCHES):
                            ys, xs = np.nonzero(y_patch_batch[bi][t, i, j].numpy())
                            for dy0, dx0 in zip(ys, xs):
                                gi = region_lat0 + i*PATCH_H + dy0
                                gj = region_lon0 + j*PATCH_W + dx0
                                gt_pts.append((gi, gj))
                    if not gt_pts:
                        continue

                    arr = np.array(gt_pts)
                    dists = np.hypot(arr[:,0] - gi_pred, arr[:,1] - gj_pred)
                    min_dists.append(dists.min())

    min_dists = np.array(min_dists)
    total_pts = len(min_dists)

    # pixel to degrees (approx)
    deg_i = 90.0  / n_lat_full
    deg_j = 180.0 / n_lon_full
    radii_px  = np.percentile(min_dists, percents)
    radii_deg = (radii_px * deg_i + radii_px * deg_j) / 2

    print(f"Total hurricane points evaluated: {total_pts}")
    print("Threshold | Radius (px) | Radius (deg) | Success (/total)")
    for pct, px, deg in zip(percents, radii_px, radii_deg):
        succ = (min_dists <= px).sum()
        print(f"{pct:3d}%      {px:8.2f}     {deg:10.4f}       {succ}/{total_pts}")

    # ===== Visualize a few frames with rings =====
    os.makedirs(out_dir, exist_ok=True)
    sample_count = 0
    for x_batch, _, y_patch_batch, start_idx_batch in val_loader_timesformer:
        if sample_count >= sample_frames:
            break
        x_batch = x_batch.to(device)
        with torch.no_grad():
            logits = model_tf(x_batch)
            probs  = torch.sigmoid(logits).cpu().numpy()

        B = x_batch.size(0)
        for bi in range(B):
            if sample_count >= sample_frames:
                break

            y_patch = y_patch_batch[bi].numpy()
            day0    = start_idx_batch[bi].item()
            win_start = start_date + timedelta(days=day0)
            pred_grid = (probs[bi] > 0.5).astype(int)  # (T,2,5)

            ncols = min(4, pred_grid.shape[0])
            nrows = (pred_grid.shape[0] + ncols - 1)//ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows),
                                     dpi=200, subplot_kw={'xticks':[], 'yticks':[]})
            axes = axes.flatten()

            for t in range(pred_grid.shape[0]):
                ax = axes[t]
                # region bounds
                lat_max = 90.0 - (region_lat0 + 0.5)*(90.0/n_lat_full)
                lat_min = 90.0 - (region_lat0 + 40 - 0.5)*(90.0/n_lat_full)
                lon_min = -180.0 + (region_lon0 + 0.5)*(180.0/n_lon_full)
                lon_max = -180.0 + (region_lon0 + 100 - 0.5)*(180.0/n_lon_full)

                m = Basemap(projection='cyl',
                            llcrnrlat=lat_min, urcrnrlat=lat_max,
                            llcrnrlon=lon_min, urcrnrlon=lon_max,
                            resolution='i', ax=ax)
                m.drawcoastlines(linewidth=0.5)
                m.fillcontinents(color='lightgray', lake_color='lightblue')
                m.drawmapboundary(fill_color='lightblue')
                ax.set_title((win_start+timedelta(days=t)).strftime("%Y-%m-%d"))

                # GT points
                for i in range(2):
                    for j in range(5):
                        ys, xs = np.nonzero(y_patch[t, i, j])
                        for dy0, dx0 in zip(ys, xs):
                            gi = region_lat0 + i*PATCH_H + dy0
                            gj = region_lon0 + j*PATCH_W + dx0
                            lat = 90.0  - (gi + 0.5)*(90.0/n_lat_full)
                            lon = -180.0 + (gj + 0.5)*(180.0/n_lon_full)
                            m.scatter(lon, lat, latlon=True, s=10, color='red',
                                      edgecolor='k', alpha=0.7, zorder=9)

                # Pred patch & UNet peak
                idxs = np.where(pred_grid[t].flatten()==1)[0]
                if len(idxs) == 0:
                    continue
                idx = idxs[0]
                i_patch, j_patch = divmod(idx, 5)

                patch_env = x_batch[bi, t,
                                     :,
                                     i_patch*PATCH_H:(i_patch+1)*PATCH_H,
                                     j_patch*PATCH_W:(j_patch+1)*PATCH_W].detach().cpu().numpy()
                yy, xx = np.meshgrid(
                    np.linspace(0,1,PATCH_H),
                    np.linspace(0,1,PATCH_W),
                    indexing='ij'
                )
                inp = np.concatenate([patch_env, xx[None], yy[None]], axis=0)
                if model_unet is None:
                    in_ch_unet = inp.shape[0]
                    model_unet = UNetFrameLevel_20x20(in_ch=in_ch_unet, out_ch=1).to(device)
                    model_unet.load_state_dict(torch.load(best_unet_heatmap_ckpt, map_location=device))
                    model_unet.eval()
                xh  = torch.from_numpy(inp).unsqueeze(0).float().to(device)
                hmap = model_unet(xh).squeeze(0).detach().cpu().numpy()
                hmap = (hmap - hmap.min())/(hmap.max()-hmap.min()+1e-8)
                dy, dx = np.unravel_index(hmap.argmax(), hmap.shape)

                gi_pred = region_lat0 + i_patch*PATCH_H + dy
                gj_pred = region_lon0 + j_patch*PATCH_W + dx
                lat_pred= 90.0 - (gi_pred + 0.5)*(90.0/n_lat_full)
                lon_pred= -180.0 + (gj_pred + 0.5)*(180.0/n_lon_full)
                m.scatter(lon_pred, lat_pred, latlon=True, s=15, color='k', marker='x', zorder=10)

                # ring legend
                deg_i = 90.0  / n_lat_full
                deg_j = 180.0 / n_lon_full
                radii_px  = np.percentile(min_dists, percents) if len(min_dists)>0 else np.zeros(len(percents))
                radii_deg = (radii_px * deg_i + radii_px * deg_j) / 2
                for px, col in zip(radii_deg, colors):
                    circ = Circle((lon_pred, lat_pred), px, facecolor=col, edgecolor=None,
                                  alpha=0.3, transform=ax.transData, zorder=11)
                    ax.add_patch(circ)

            for ax in axes[pred_grid.shape[0]:]:
                ax.axis('off')

            handles = [
                Line2D([0],[0], marker='o', color='w', label='GT',
                       markerfacecolor='red', markersize=6, markeredgecolor='k'),
                Line2D([0],[0], marker='x', color='k', label='Pred',
                       markersize=8, linestyle='None')
            ] + [
                Line2D([0],[0], marker='s', color='w',
                       markerfacecolor=col, markersize=10, alpha=0.3,
                       label=f"{pct}%")
                for pct, col in zip(percents, colors)
            ]
            fig.legend(handles=handles, loc='lower center',
                       ncol=len(handles), frameon=False, fontsize=8)

            plt.tight_layout(rect=[0,0.05,1,1])
            fout = os.path.join(out_dir, f"{win_start.strftime('%Y%m%d')}.png")
            fig.savefig(fout)
            plt.close(fig)
            sample_count += 1

    print(f"[heatmap][eval] Figures saved under: {out_dir}")
