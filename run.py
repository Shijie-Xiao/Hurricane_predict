#!/usr/bin/env python3
"""
Hurricane Genesis CLI.

    python run.py prepare                 # crop raw data -> cache/
    python run.py train                   # train Timesformer
    python run.py train --unet            # train UNet heatmap
    python run.py saliency                # Integrated Gradients maps
    python run.py pcmci                   # PCMCI causal graph
"""

import argparse
import yaml

import dataset
import train
import viz


def load_cfg(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d[p]
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
                elif val.lower() == "null":
                    val = None
        d[parts[-1]] = val
    return cfg


def cmd_prepare(args):
    cfg = load_cfg(args.config, args.override)
    dataset.prepare_data(cfg)


def cmd_train(args):
    cfg = load_cfg(args.config, args.override)
    if args.unet:
        train.train_unet(cfg)
    else:
        train.train_cls(cfg)


def cmd_saliency(args):
    cfg = load_cfg(args.config, args.override)
    viz.saliency(cfg, ckpt_path=args.ckpt)


def cmd_pcmci(args):
    cfg = load_cfg(args.config, args.override)
    viz.pcmci(cfg, saliency_dir=args.saliency_dir)


def _add_common(sp):
    sp.add_argument("--config", default="config.yaml")
    sp.add_argument("--override", nargs="*", default=[], metavar="KEY=VAL",
                    help="Override config values, e.g. train.epochs=5")


def main():
    p = argparse.ArgumentParser(description="Hurricane Genesis CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    _add_common(sub.add_parser("prepare", help="Crop raw data and cache"))

    sp = sub.add_parser("train", help="Train model")
    sp.add_argument("--unet", action="store_true", help="Train UNet instead of Timesformer")
    _add_common(sp)

    sp = sub.add_parser("saliency", help="Compute IG saliency maps")
    sp.add_argument("--ckpt", default=None, help="Checkpoint path (default: auto)")
    _add_common(sp)

    sp = sub.add_parser("pcmci", help="Run PCMCI causal analysis")
    sp.add_argument("--saliency-dir", default=None, dest="saliency_dir")
    _add_common(sp)

    args = p.parse_args()
    {"prepare": cmd_prepare, "train": cmd_train,
     "saliency": cmd_saliency, "pcmci": cmd_pcmci}[args.cmd](args)


if __name__ == "__main__":
    main()
