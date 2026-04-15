#!/usr/bin/env python3
"""
Hurricane Genesis CLI – two automated pipelines.

    python run.py predict     # prepare → train Patch → train UNet → eval
    python run.py visualize   # prepare → train Hier → saliency → PCMCI

Individual steps are also available:

    python run.py prepare / train / eval / saliency / pcmci
"""

import argparse
import copy
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


# ── automated pipelines ──────────────────────────────────────────────

def cmd_predict(args):
    """Pipeline A: prepare → train PatchTimesformer → train UNet → eval."""
    cfg = load_cfg(args.config, args.override)
    cfg["model"]["type"] = "patch"

    print("=" * 60)
    print("[pipeline] PREDICT: prepare → train(patch) → train(unet) → eval")
    print("=" * 60)

    print("\n>>> Step 1/4: prepare data")
    dataset.prepare_data(cfg)

    print("\n>>> Step 2/4: train PatchTimesformer")
    results = train.train_cls(cfg)
    best_ckpt = max(results, key=lambda r: r[1])[0]

    print("\n>>> Step 3/4: train UNet (sub-patch localization)")
    train.train_unet(cfg)

    print("\n>>> Step 4/4: evaluate (classification + geographic distance)")
    train.eval_cls(cfg, ckpt_path=best_ckpt)

    print("\n" + "=" * 60)
    print("[pipeline] PREDICT complete.")
    print("=" * 60)


def cmd_visualize(args):
    """Pipeline B: prepare → train HierTimesformer → saliency → PCMCI."""
    cfg = load_cfg(args.config, args.override)

    print("=" * 60)
    print("[pipeline] VISUALIZE: prepare → train(hier) → saliency → PCMCI")
    print("=" * 60)

    print("\n>>> Step 1/4: prepare data")
    dataset.prepare_data(cfg)

    print("\n>>> Step 2/4: train HierTimesformer")
    cfg_hier = copy.deepcopy(cfg)
    cfg_hier["model"]["type"] = "hierarchical"
    results = train.train_cls(cfg_hier)
    best_ckpt = max(results, key=lambda r: r[1])[0]

    print("\n>>> Step 3/4: Integrated Gradients saliency")
    viz.saliency(cfg_hier, ckpt_path=best_ckpt)

    print("\n>>> Step 4/4: PCMCI causal analysis")
    viz.pcmci(cfg_hier)

    print("\n" + "=" * 60)
    print("[pipeline] VISUALIZE complete.")
    print("=" * 60)


# ── individual step commands ─────────────────────────────────────────

def cmd_prepare(args):
    cfg = load_cfg(args.config, args.override)
    dataset.prepare_data(cfg)


def cmd_train(args):
    cfg = load_cfg(args.config, args.override)
    if args.unet:
        train.train_unet(cfg)
    else:
        train.train_cls(cfg)


def cmd_eval(args):
    cfg = load_cfg(args.config, args.override)
    train.eval_cls(cfg, ckpt_path=args.ckpt)


def cmd_saliency(args):
    cfg = load_cfg(args.config, args.override)
    viz.saliency(cfg, ckpt_path=args.ckpt)


def cmd_pcmci(args):
    cfg = load_cfg(args.config, args.override)
    viz.pcmci(cfg, saliency_dir=args.saliency_dir)


# ── CLI ──────────────────────────────────────────────────────────────

def _add_common(sp):
    sp.add_argument("--config", default="config.yaml")
    sp.add_argument("--override", nargs="*", default=[], metavar="KEY=VAL",
                    help="Override config values, e.g. train.epochs=5")


def main():
    p = argparse.ArgumentParser(
        description="Hurricane Genesis – prediction & visualization pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    _add_common(sub.add_parser(
        "predict",
        help="Full pipeline: prepare → train Patch + UNet → eval"))
    _add_common(sub.add_parser(
        "visualize",
        help="Full pipeline: prepare → train Hier → saliency → PCMCI"))

    _add_common(sub.add_parser("prepare", help="Crop raw data and cache"))

    sp = sub.add_parser("train", help="Train a single model")
    sp.add_argument("--unet", action="store_true")
    _add_common(sp)

    sp = sub.add_parser("eval", help="Evaluate checkpoint")
    sp.add_argument("--ckpt", default=None)
    _add_common(sp)

    sp = sub.add_parser("saliency", help="Compute IG saliency maps")
    sp.add_argument("--ckpt", default=None)
    _add_common(sp)

    sp = sub.add_parser("pcmci", help="Run PCMCI causal analysis")
    sp.add_argument("--saliency-dir", default=None, dest="saliency_dir")
    _add_common(sp)

    args = p.parse_args()
    dispatch = {
        "predict": cmd_predict, "visualize": cmd_visualize,
        "prepare": cmd_prepare, "train": cmd_train, "eval": cmd_eval,
        "saliency": cmd_saliency, "pcmci": cmd_pcmci,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
