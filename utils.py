#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions:
  - Data preparation: crop 90x180 -> region 40x100, optional channel subset, optional ENSO split.
  - Channel-name mapping helpers.
  - Seeding, path helpers, simple geo utilities.
"""

from pathlib import Path
import json
import numpy as np
import random
import os


# ---- Canonical 28-channel names (example order, edit to match your data) ----
CHANNEL_NAMES_28 = [
    "1000 cloud cover","1000 Geopotential","1000 Vorticity (relative)","1000 Specific humidity",
    "850 cloud cover","850 Geopotential","Relative Vorticity (850mb)","Specific Humidity (850mb)",
    "850 Temperature","Cloud Cover (500mb)","Relative Vorticity (500mb)","Vertical Shear",
    "Potential Intensity","Sea surface temperature","Surface pressure","Charnock (Wave Drag)",
    "Surface Radiation","Convective precipitation","Total column rain water","500 Divergence",
    "850 Divergence","1000 Divergence","ENSO","MJO","Salinity","Ocean Heat Content",
    "850 Saturation Deficit","Saturation Deficit (500mb)"
]

# ---- Default 9-channel subset by name ----
DEFAULT_SELECTED_9 = [
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


def set_seed(seed: int):
    """Deterministic seeding for Python/NumPy (Torch is handled inside train)."""
    random.seed(seed)
    np.random.seed(seed)


def _resolve_indices(full_names, desired_names):
    name_to_idx = {nm: i for i, nm in enumerate(full_names)}
    kept_idx, kept_names, missing = [], [], []
    for nm in (desired_names or []):
        if nm in name_to_idx:
            kept_idx.append(name_to_idx[nm]); kept_names.append(nm)
        else:
            missing.append(nm)
    return kept_idx, kept_names, missing


def prepare_data(
    hurr_file: str,
    env_file: str,
    out_dir: str = ".",
    lat_slice=(45, 85),
    lon_slice=(80, 180),
    selected_channel_names=None,
    do_enso_split=False,
    th_nino=0.5,
    th_nina=-0.5,
    enso_name="ENSO",
):
    """
    Prepare region arrays (40x100) and optionally create ENSO splits.
    Returns dict of output paths.
    """
    out = {}
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] Load raw arrays...")
    hurr = np.load(hurr_file)  # (T,90,180)
    env  = np.load(env_file)   # (T,90,180,C)
    assert hurr.ndim == 3 and env.ndim == 4, f"Unexpected shapes: {hurr.shape}, {env.shape}"
    T, H, W = hurr.shape
    T2, H2, W2, C = env.shape
    assert (T, H, W) == (T2, H2, W2), "Time/space mismatch between HURR and ENV"

    lat0, lat1 = lat_slice
    lon0, lon1 = lon_slice

    # Crop to region (North Atlantic default 45:85, 80:180) -> (T,40,100,C)
    print(f"[prepare] Crop region lat[{lat0}:{lat1}] lon[{lon0}:{lon1}] ...")
    region_hurr = hurr[:, lat0:lat1, lon0:lon1]
    region_env  = env[:,  lat0:lat1, lon0:lon1, :]
    print(f"[prepare] Region shapes: hurr {region_hurr.shape}, env {region_env.shape}")

    # Optional channel subset by name
    kept_idx, kept_names, missing = [], [], []
    if selected_channel_names:
        kept_idx, kept_names, missing = _resolve_indices(CHANNEL_NAMES_28, selected_channel_names)
        if missing:
            print("[prepare][WARN] Missing channel names (skipped):", missing)
        region_env = region_env[..., kept_idx]
        print(f"[prepare] Selected channels ({len(kept_idx)}): {kept_names}")

    # Save region files
    fp_region_hurr = outp / "region_hurr.npy"
    fp_region_env  = outp / "region_env.npy"
    np.save(fp_region_hurr, region_hurr)
    np.save(fp_region_env, region_env)
    out["region_hurr"] = str(fp_region_hurr)
    out["region_env"]  = str(fp_region_env)

    # Optional ENSO split on CROPPED region (daily regional mean)
    if do_enso_split:
        if enso_name not in CHANNEL_NAMES_28:
            raise ValueError(f"Channel list does not contain '{enso_name}'.")
        enso_idx = CHANNEL_NAMES_28.index(enso_name)
        # Note: If subset was applied and ENSO is not inside kept_names, compute from original env.
        enso_daily = env[:, lat0:lat1, lon0:lon1, enso_idx].mean(axis=(1, 2))  # (T,)

        idx_nino = np.where(enso_daily >= th_nino)[0]
        idx_nina = np.where(enso_daily <= th_nina)[0]
        idx_net  = np.where((enso_daily > th_nina) & (enso_daily < th_nino))[0]
        print(f"[prepare] ENSO counts  NINO={len(idx_nino)}  NINA={len(idx_nina)}  NET={len(idx_net)}")

        def _save_split(tag, idx):
            X = region_env[idx]
            Y = region_hurr[idx]
            fx = outp / f"X_{tag}.npy"
            fy = outp / f"Y_{tag}.npy"
            np.save(fx, X); np.save(fy, Y)
            out[f"X_{tag}"] = str(fx)
            out[f"Y_{tag}"] = str(fy)

        _save_split("NINO", idx_nino)
        _save_split("NINA", idx_nina)
        _save_split("NET",  idx_net)

    # Save meta
    meta = {
        "input": {"hurr": hurr_file, "env": env_file, "T": int(T), "H": int(H), "W": int(W), "C": int(env.shape[-1])},
        "region": {"lat_slice": [lat0, lat1], "lon_slice": [lon0, lon1]},
        "selected_channels": kept_names,
        "missing_channels": missing,
        "outputs": out,
        "enso": {"enabled": bool(do_enso_split), "th_nino": th_nino, "th_nina": th_nina}
    }
    meta_fp = outp / "split_meta.json"
    with open(meta_fp, "w") as f:
        json.dump(meta, f, indent=2)
    out["meta"] = str(meta_fp)

    return out


# --- Simple geo helper for plotting grids (Basemap/Cartopy agnostic) ---
def geo_bounds_and_grid(H, W, n_lat_full=90, n_lon_full=180, region_lat0=45, region_lon0=80, ds_factor=2):
    lat_max =  90.0 - (region_lat0 + 0.5) * (90.0 / n_lat_full)
    lat_min =  90.0 - (region_lat0 + H   - 0.5) * (90.0 / n_lat_full)
    lon_min = -180.0 + (region_lon0 + 0.5) * (180.0 / n_lon_full)
    lon_max = -180.0 + (region_lon0 + W   - 0.5) * (180.0 / n_lon_full)
    lats = np.linspace(lat_max, lat_min, max(1, H//ds_factor))
    lons = np.linspace(lon_min, lon_max, max(1, W//ds_factor))
    lon2d, lat2d = np.meshgrid(lons, lats)
    return (lat_min, lat_max, lon_min, lon_max), (lon2d, lat2d)
