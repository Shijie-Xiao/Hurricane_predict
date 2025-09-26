#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified CLI entry for the hurricane genesis codebase.

Subcommands:
  - prepare  : crop raw (90x180) to region (40x100), optional channel subset, optional ENSO split
  - train    : multi-run Timesformer training; supports FULL or ENSO splits (NINO/NINA/NET)
  - pcmci    : run PCMCI causal visualization
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
import train as trainer  # train.py
import heatmap

def cmd_patches(args):
    """Generate 20x20 patches from region files."""
    heatmap.make_patches_20x20(
        region_env_path=args.env,
        region_hurr_path=args.hurr,
        out_env_patches=args.out_env_patches,
        out_hurr_patches=args.out_hurr_patches,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
    )

def cmd_heatmap_train(args):
    """Train frame-level UNet heatmap model on patches."""
    heatmap.train_heatmap(
        env_patches_path=args.env_patches,
        hurr_patches_path=args.hurr_patches,
        sigma=args.sigma,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        out_ckpt=args.out_ckpt,
        seed=args.seed,
    )

def cmd_heatmap_eval(args):
    """
    Evaluate Timesformer + UNet heatmap.
    Requires a validation loader produced by your existing train_9.py / train.py.
    For simplicity here we import its dataset maker if available; otherwise, please pass a saved val loader.
    """
    try:
        from train_9 import build_loaders as build_timesformer_loaders
    except Exception:
        from train import build_loaders as build_timesformer_loaders

    loaders = build_timesformer_loaders(
        env_path=args.env, hurr_path=args.hurr,
        seq_len=args.seq_len, stride=args.stride,
        batch_size=args.batch_size, in_ch=args.in_ch,
        seed=args.seed
    )
    _, val_loader = loaders
    heatmap.eval_heatmap_frames(
        best_patch_timesformer_ckpt=args.patch_ckpt,
        best_unet_heatmap_ckpt=args.unet_ckpt,
        val_loader_timesformer=val_loader,
        seq_len=args.seq_len,
        region_lat0=args.lat0, region_lon0=args.lon0,
        n_lat_full=args.n_lat_full, n_lon_full=args.n_lon_full,
        out_dir=args.out_dir,
        sample_frames=args.sample_frames,
    )


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


def _resolve_paths_for_split(use_split: str, env: str, hurr: str):
    """If --use_split is set, map to X_<SPLIT>.npy / Y_<SPLIT>.npy, else keep passed paths."""
    if not use_split:
        return env, hurr, None
    split = use_split.upper()
    env_s, hurr_s = f"X_{split}.npy", f"Y_{split}.npy"
    return env_s, hurr_s, split


def cmd_train(args):
    """Train Timesformer; supports FULL or ENSO splits."""
    # If --use_split is provided, override env/hurr
    env, hurr, split = _resolve_paths_for_split(args.use_split, args.env, args.hurr)
    print(f"[train] Using env={env}  hurr={hurr}")

    # If out_root remains default, auto-append split/in_ch
    out_root = args.out_root
    if out_root == "runs_FULL_auto":
        tag = split if split else "FULL"
        ch  = args.in_ch if args.in_ch is not None else "auto"
        out_root = f"runs_{tag}_{ch}"

    trainer.run_training(
        env_path=env,
        hurr_path=hurr,
        out_root=out_root,
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
    """
    Quick spatial plot for a .npy array (2D).

    Additionally, if --saliency_ckpt is provided, this will also compute and save
    Integrated Gradients saliency maps for ALL channels, using the given checkpoint
    and env/hurr files, saving under <run_dir>/maps_<in_ch>Channels_3/<ChannelName>/...
    """
    # 1) Basic quick plot (if --array is given)
    if args.array:
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

    # 2) Optional: run IG saliency batch if checkpoint is provided
    if args.saliency_ckpt:
        if not args.env or not args.hurr:
            raise ValueError("When using --saliency_ckpt, please also provide --env and --hurr.")
        run_dir = args.run_dir or "runs_FULL_auto/run_01"
        print(f"[spatial] Running IG saliency from ckpt={args.saliency_ckpt} into {run_dir}")
        visualize.saliency_entrypoint(
            ckpt_path=args.saliency_ckpt,
            run_dir=run_dir,
            env_path=args.env,
            hurr_path=args.hurr,
            seq_len=args.seq_len,
            stride=args.stride,
            batch_size=args.batch_size,
            in_ch=args.in_ch
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
    sp.add_argument("--in_ch", type=int, default=9, help="If 9, select default 9-channel subset; else keep all channels.")
    sp.add_argument("--split_enso", action="store_true", help="Also generate X_NINO/X_NINA/X_NET splits.")
    sp.add_argument("--th_nino", type=float, default=0.5)
    sp.add_argument("--th_nina", type=float, default=-0.5)
    sp.set_defaults(func=cmd_prepare)

    # train
    sp = sub.add_parser("train", help="Train Timesformer (FULL by default; use --use_split to select NINO/NINA/NET).")
    sp.add_argument("--env", type=str, default="region_env.npy")
    sp.add_argument("--hurr", type=str, default="region_hurr.npy")
    sp.add_argument("--use_split", type=str, choices=["NINO","NINA","NET"], default=None,
                    help="If set, use X_<SPLIT>.npy / Y_<SPLIT>.npy instead of region_env/hurr.")
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
    sp = sub.add_parser("spatial", help="Quick spatial plot for a 2D array (.npy); optionally compute IG saliency for all channels.")
    sp.add_argument("--array", type=str, default=None, help="Path to a 2D .npy array to plot (optional).")
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

    # ↓↓↓ New options to trigger IG saliency batch generation ↓↓↓
    sp.add_argument("--saliency_ckpt", type=str, default=None,
                    help="Path to a trained checkpoint (.pt). If set, compute IG saliency for all channels.")
    sp.add_argument("--run_dir", type=str, default=None,
                    help="Where to save saliency outputs (default: runs_FULL_auto/run_01).")
    sp.add_argument("--env", type=str, default=None,
                    help="Env array (T,40,100,C) used to build validation loader for saliency.")
    sp.add_argument("--hurr", type=str, default=None,
                    help="Hurr mask (T,40,100) used to build validation loader for saliency.")
    sp.add_argument("--in_ch", type=int, default=None,
                    help="If None, inferred from env last dim.")
    sp.add_argument("--seq_len", type=int, default=14)
    sp.add_argument("--stride",  type=int, default=3)
    sp.add_argument("--batch_size", type=int, default=4)

    sp.set_defaults(func=cmd_spatial)

    # patches
    sp = sub.add_parser("patches", help="Generate 20x20 ENV/HURR patch arrays from region files.")
    sp.add_argument("--env", type=str, default="region_env.npy")
    sp.add_argument("--hurr", type=str, default="region_hurr.npy")
    sp.add_argument("--out_env_patches", type=str, default="ENV_PATCHES_20x20.npy")
    sp.add_argument("--out_hurr_patches", type=str, default="HURR_PATCHES_20x20.npy")
    sp.add_argument("--patch_h", type=int, default=20)
    sp.add_argument("--patch_w", type=int, default=20)
    sp.set_defaults(func=cmd_patches)

    # heatmap_train
    sp = sub.add_parser("heatmap_train", help="Train frame-level U-Net heatmap on 20x20 patches.")
    sp.add_argument("--env_patches", type=str, default="ENV_PATCHES_20x20.npy")
    sp.add_argument("--hurr_patches", type=str, default="HURR_PATCHES_20x20.npy")
    sp.add_argument("--sigma", type=float, default=1.5)
    sp.add_argument("--batch_size", type=int, default=32)
    sp.add_argument("--epochs", type=int, default=100)
    sp.add_argument("--lr", type=float, default=1e-4)
    sp.add_argument("--val_split", type=float, default=0.1)
    sp.add_argument("--out_ckpt", type=str, default="best_unet_heatmap.pt")
    sp.add_argument("--seed", type=int, default=1337)
    sp.set_defaults(func=cmd_heatmap_train)

    # heatmap_eval
    sp = sub.add_parser("heatmap_eval", help="Evaluate Timesformer+UNet and export frame maps with radius rings.")
    sp.add_argument("--patch_ckpt", type=str, default="best_patch_timesformer_by_f1.pt")
    sp.add_argument("--unet_ckpt", type=str, default="best_unet_heatmap.pt")
    sp.add_argument("--env", type=str, default="region_env.npy")
    sp.add_argument("--hurr", type=str, default="region_hurr.npy")
    sp.add_argument("--in_ch", type=int, default=None, help="Infer from env if None.")
    sp.add_argument("--seq_len", type=int, default=7)
    sp.add_argument("--stride", type=int, default=3)
    sp.add_argument("--batch_size", type=int, default=8)
    sp.add_argument("--seed", type=int, default=1337)
    sp.add_argument("--lat0", type=int, default=45)
    sp.add_argument("--lon0", type=int, default=80)
    sp.add_argument("--n_lat_full", type=int, default=90)
    sp.add_argument("--n_lon_full", type=int, default=180)
    sp.add_argument("--out_dir", type=str, default="frame_maps_with_global_radii")
    sp.add_argument("--sample_frames", type=int, default=10)
    sp.set_defaults(func=cmd_heatmap_eval)



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
