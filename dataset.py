"""
Data preparation and PyTorch Dataset / DataLoader factories.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.ndimage import gaussian_filter


# ── data preparation ─────────────────────────────────────────────────
def prepare_data(cfg):
    """Load raw ERA5 arrays, transpose, crop to region, and cache as .npy.

    Returns ``(hurr_path, env_path)`` pointing to the cached files.
    """
    dc = cfg["data"]
    cache = Path(dc["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    hurr_out = cache / "region_hurr.npy"
    env_out = cache / "region_env.npy"

    if hurr_out.exists() and env_out.exists():
        h = np.load(hurr_out, mmap_mode="r")
        e = np.load(env_out, mmap_mode="r")
        print(f"[prepare] Using cached data: hurr {h.shape}, env {e.shape}")
        return str(hurr_out), str(env_out)

    print("[prepare] Loading raw arrays (this may take a minute) ...")
    hurr_raw = np.load(dc["hurr_path"])
    env_raw = np.load(dc["env_path"])

    # optional transpose (e.g. (C,T,H,W) -> (T,H,W,C))
    if dc.get("env_transpose"):
        env_raw = np.transpose(env_raw, dc["env_transpose"])

    # align time dimension
    min_days = min(hurr_raw.shape[0], env_raw.shape[0])
    hurr_raw = hurr_raw[:min_days]
    env_raw = env_raw[:min_days]

    lat0, lat1 = dc["region"]["lat"]
    lon0, lon1 = dc["region"]["lon"]
    region_hurr = hurr_raw[:, lat0:lat1, lon0:lon1]
    region_env = env_raw[:, lat0:lat1, lon0:lon1, :]

    np.save(hurr_out, region_hurr.astype(np.float32))
    np.save(env_out, region_env.astype(np.float32))
    print(f"[prepare] Saved: hurr {region_hurr.shape}, env {region_env.shape}")
    return str(hurr_out), str(env_out)


# ── Timesformer dataset ──────────────────────────────────────────────
class PatchDataset(Dataset):
    """Sliding-window sequences with patch-level binary labels.

    Returns ``(x, y, y_patch, start_idx)`` where
    * x: ``(T, C, H, W)``
    * y: ``(T, n_lat, n_lon)`` binary patch labels
    * y_patch: ``(T, n_lat, n_lon, ph, pw)`` raw pixel masks per patch
    * start_idx: int – first day index of the window
    """

    def __init__(self, env_path, hurr_path, seq_len, stride, patch_h, patch_w):
        env_all = np.nan_to_num(np.load(env_path), nan=0.0, posinf=0.0, neginf=0.0)
        hurr_all = np.nan_to_num(np.load(hurr_path), nan=0.0, posinf=0.0, neginf=0.0)
        assert env_all.ndim == 4 and hurr_all.ndim == 3

        self.env = torch.from_numpy(env_all).float()
        self.hurr = torch.from_numpy(hurr_all).float()
        self.seq_len = seq_len
        self.patch_h, self.patch_w = patch_h, patch_w
        D, H, W = hurr_all.shape
        self.n_lat = H // patch_h
        self.n_lon = W // patch_w

        all_starts = list(range(0, D - seq_len + 1, stride))
        self.starts = [
            s for s in all_starts
            if (hurr_all[s : s + seq_len].reshape(seq_len, -1).sum(axis=1) > 0).any()
        ] or all_starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        T = self.seq_len
        ph, pw = self.patch_h, self.patch_w
        nl, ml = self.n_lat, self.n_lon

        x = self.env[s : s + T].permute(0, 3, 1, 2)        # (T, C, H, W)
        h = self.hurr[s : s + T]                             # (T, H, W)

        y_patch = (
            h.reshape(T, nl, ph, ml, pw).permute(0, 1, 3, 2, 4)
        )                                                     # (T, nl, ml, ph, pw)
        y = (y_patch.reshape(T, nl, ml, ph * pw).sum(dim=-1) > 0).long()
        return x, y, y_patch, s


# ── UNet heatmap dataset ─────────────────────────────────────────────
class HeatmapDataset(Dataset):
    """Per-frame, per-patch samples with Gaussian-smoothed target.

    Input: ``(C + 2, ph, pw)`` (env channels + 2 positional channels).
    Target: ``(ph, pw)`` float heatmap in [0, 1].
    """

    def __init__(self, env_path, hurr_path, patch_h, patch_w, sigma=1.5):
        env_all = np.nan_to_num(np.load(env_path), nan=0.0).astype(np.float32)
        hurr_all = np.nan_to_num(np.load(hurr_path), nan=0.0).astype(np.float32)
        D, H, W, C = env_all.shape
        nl, ml = H // patch_h, W // patch_w
        P = nl * ml

        # reshape to patches
        self.env_p = (
            env_all.reshape(D, nl, patch_h, ml, patch_w, C)
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(D, P, patch_h, patch_w, C)
        )
        self.hurr_p = (
            hurr_all.reshape(D, nl, patch_h, ml, patch_w)
            .transpose(0, 1, 3, 2, 4)
            .reshape(D, P, patch_h, patch_w)
        )
        self.sigma = sigma

        self.indices = [
            (t, p)
            for t in range(D)
            for p in range(P)
            if self.hurr_p[t, p].sum() > 0
        ]

        yy, xx = np.meshgrid(
            np.linspace(0, 1, patch_h), np.linspace(0, 1, patch_w), indexing="ij"
        )
        self.pos = np.stack([xx, yy], axis=0).astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t, p = self.indices[idx]
        x = self.env_p[t, p].transpose(2, 0, 1)             # (C, ph, pw)
        x = np.concatenate([x, self.pos], axis=0)            # (C+2, ph, pw)
        mask = self.hurr_p[t, p]
        hm = gaussian_filter(mask, sigma=self.sigma)
        mx = hm.max()
        hm = (hm / mx).astype(np.float32) if mx > 0 else hm.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(hm)


# ── loader factories ─────────────────────────────────────────────────
def build_loaders(cfg, *, for_unet=False):
    """Return ``(train_loader, val_loader)`` ready for training.

    When *for_unet* is True, builds ``HeatmapDataset``; otherwise
    ``PatchDataset``.
    """
    dc = cfg["data"]
    cache = Path(dc["cache_dir"])
    env_path = str(cache / "region_env.npy")
    hurr_path = str(cache / "region_hurr.npy")
    ph, pw = cfg["model"]["patch"]
    seed = cfg["train"]["seed"]

    if for_unet:
        ds = HeatmapDataset(env_path, hurr_path, ph, pw, sigma=cfg["unet"]["sigma"])
        bs = cfg["unet"]["batch_size"]
    else:
        ds = PatchDataset(
            env_path, hurr_path,
            seq_len=cfg["model"]["seq_len"],
            stride=cfg["train"]["stride"],
            patch_h=ph, patch_w=pw,
        )
        bs = cfg["train"]["batch_size"]

    n_val = max(1, int(len(ds) * cfg["train"]["val_split"]))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False)
    print(f"[data] {'UNet' if for_unet else 'Timesformer'}: "
          f"train={n_train}, val={n_val}, batch={bs}")
    return train_loader, val_loader
