#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified CLI entry for the hurricane genesis codebase.

Subcommands:
  - prepare  : crop raw (90x180) to region (40x100), optional channel subset, optional ENSO split
  - train    : multi-run Timesformer training on FULL dataset by default (no ENSO split)
  - pcmci    : run PCMCI causal visualization with sensible defaults
  - spatial  : quick spatial contour maps (for arrays or saliency)
  - shap     : placeholder
  - predict  : placeholder (U-Net)

All commands have defaults so the script runs even without any external config.
"""

import argparse
from pathlib import Path

# Local modules
import utils
import visualize
import train as trainer  # ← now pointing to train.py


def cmd_prepare(args):
    """Crop raw arrays to region and optionally select channels / split ENSO."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = utils.prepare_data(
        hurr_file=args.hurr_raw,
        env_file=args.env_raw,
        out_dir=str(out_dir),
        lat_slice=(args.lat0, args.lat1),
        lon_slice=(args.lon0, args.lon1),
        selected_channel_names=utils.DEFAULT_SELECTED_9 if args.in_ch == 9 else None,
        do_enso_split=args.split_enso,
        th_nino=args.th_nino,
        th_nina=args.th_nina,
        enso_name="ENSO",
    )

    print("\n[prepare] Done.")
    for k, v in result.items():
        print(f"  {k}: {v}")


def cmd_train(args):
    """Train Timesformer on FULL dataset by default (no ENSO split)."""
    env  = args.env or "region_env.npy"
    hurr = args.hurr or "region_hurr.npy"

    print(f"[train] Using env={env}  hurr={hurr}")
    trainer.run_training(
        env_path=env,
        hurr_path=hurr,
        out_root=args.out_root or "runs_FULL_auto",
        runs=args.runs,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        in_ch=args.in_ch
    )


def cmd_pcmci(args):
    """Run PCMCI causal visualization."""
    visualize.pcmci_main(
        env_path=args.env or "region_env.npy",
        hurr_path=args.hurr or "region_hurr.npy",
        saliency_dir=args.saliency_dir or "runs_FULL_auto/run_01/maps_9Channels_3",
        sp_h=args.sp_h,
        sp_w=args.sp_w,
        th_sal=args.th_sal,
        fallback_topq=args.fallback_topq,
        min_exp=args.min_exp,
        pre_days=args.pre_days,
        tau_max=args.tau_max,
        r0_base=args.r0_base,
        pc_alpha_base=args.pc_alpha_base,
        alpha_loose=args.alpha_loose,
        alpha_strict=args.alpha_strict,
        min_abs=args.min_abs,
        max_conds_dim=args.max_conds_dim,
        max_combinations=args.max_combinations,
        fig_prefix=args.fig_prefix,
    )


def cmd_spatial(args):
    """Quick spatial plot for a .npy array (2D)."""
    visualize.spatial_main(
        array_path=args.array,
        title=args.title,
        vmin=args.vmin,
        vmax=args.vmax,
        steps=args.steps,
        out_path=args.out or "spatial.png",
        lat0=args.lat0, lat1=args.lat1, lon0=args.lon0, lon1=args.lon1,
        downsample=args.downsample
    )


def cmd_placeholder(_):
    print("This command is a placeholder (to be implemented by collaborators).")


def build_parser():
    p = argparse.ArgumentParser(description="Unified CLI for hurricane genesis workflows.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # prepare
    sp = sub.add_parser("prepare", help="Crop raw data (90x180) to region (40x100), optional channel subset & ENSO split")
    sp.add_argument("--hurr_raw", type=str, default="HURR_LOC_DAILY.npy")
    sp.add_argument("--env_raw",  type=str, default="X_28.npy")
    sp.add_argument("--out_dir",  type=str, default=".")
    sp.add_argument("--lat0", type=int, default=45)
    sp.add_argument("--lat1", type=int, default=85)
    sp.add_argument("--lon0", type=int, default=80)
    sp.add_argument("--lon1", type=int, default=180)
    sp.add_argument("--in_ch", type=int, default=9, help="If 9, select default 9-channel subset; else use all channels.")
    sp.add_argument("--split_enso", action="store_true", help="Also generate X_NINO/X_NINA/X_NET splits.")
    sp.add_argument("--th_nino", type=float, default=0.5)
    sp.add_argument("--th_nina", type=float, default=-0.5)
    sp.set_defaults(func=cmd_prepare)

    # train
    sp = sub.add_parser("train", help="Train Timesformer (FULL dataset by default).")
    sp.add_argument("--env", type=str, default="region_env.npy")
    sp.add_argument("--hurr", type=str, default="region_hurr.npy")
    sp.add_argument("--in_ch", type=int, default=None, help="If None, inferred from env last dim.")
    sp.add_argument("--seq_len", type=int, default=14)
    sp.add_argument("--stride",  type=int, default=3)
    sp.add_argument("--batch_size", type=int, default=8)
    sp.add_argument("--out_root", type=str, default="runs_FULL_auto")
    sp.add_argument("--runs", type=int, default=3)
    sp.add_argument("--epochs", type=int, default=60)
    sp.add_argument("--lr", type=float, default=1e-4)
    sp.add_argument("--weight_decay", type=float, default=1e-5)
    sp.add_argument("--seed", type=int, default=1337)
    sp.set_defaults(func=cmd_train)

    # pcmci
    sp = sub.add_parser("pcmci", help="Run PCMCI causal visualization.")
    sp.add_argument("--env", type=str, default="region_env.npy")
    sp.add_argument("--hurr", type=str, default="region_hurr.npy")
    sp.add_argument("--saliency_dir", type=str, default="runs_FULL_auto/run_01/maps_9Channels_3")
    sp.add_argument("--sp_h", type=int, default=2)
    sp.add_argument("--sp_w", type=int, default=2)
    sp.add_argument("--th_sal", type=float, default=0.75)
    sp.add_argument("--fallback_topq", type=float, default=0.80)
    sp.add_argument("--min_exp", type=float, default=0.05)
    sp.add_argument("--pre_days", type=int, default=7)
    sp.add_argument("--tau_max", type=int, default=7)
    sp.add_argument("--r0_base", type=float, default=0.05)
    sp.add_argument("--pc_alpha_base", type=float, default=0.1)
    sp.add_argument("--alpha_loose", type=float, default=0.05)
    sp.add_argument("--alpha_strict", type=float, default=0.0001)
    sp.add_argument("--min_abs", type=float, default=0.05)
    sp.add_argument("--max_conds_dim", type=int, default=2)
    sp.add_argument("--max_combinations", type=int, default=100000)
    sp.add_argument("--fig_prefix", type=str, default="PCMCI_PRE7MASK")
    sp.set_defaults(func=cmd_pcmci)

    # spatial
    sp = sub.add_parser("spatial", help="Quick spatial contour plot for a 2D array (.npy).")
    sp.add_argument("--array", type=str, required=True)
    sp.add_argument("--title", type=str, default="Spatial Map")
    sp.add_argument("--vmin", type=float, default=-1.0)
    sp.add_argument("--vmax", type=float, default=1.0)
    sp.add_argument("--steps", type=int, default=21)
    sp.add_argument("--out", type=str, default="spatial.png")
    sp.add_argument("--lat0", type=int, default=45)
    sp.add_argument("--lat1", type=int, default=85)
    sp.add_argument("--lon0", type=int, default=80)
    sp.add_argument("--lon1", type=int, default=180)
    sp.add_argument("--downsample", type=int, default=2)
    sp.set_defaults(func=cmd_spatial)

    # placeholders
    sub.add_parser("shap", help="Placeholder").set_defaults(func=cmd_placeholder)
    sub.add_parser("predict", help="Placeholder (U-Net)").set_defaults(func=cmd_placeholder)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
