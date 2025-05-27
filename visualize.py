# visualize.py

import os
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

from torch.utils.data import DataLoader
from utils import SequenceDataset, collate_fn
from model import DETRModel

def nms_points(pts: np.ndarray, scores: np.ndarray, radius: float):
    """
    Simple NMS for 2D points.
    Args:
        pts: array of shape (N, 2), normalized coordinates in [0,1]
        scores: array of shape (N,), confidence scores
        radius: suppression radius (same units as pts, e.g. normalized)
    Returns:
        keep: list of indices to keep
    """
    order = np.argsort(-scores)
    keep = []
    for i in order:
        p = pts[i]
        if all(np.linalg.norm(p - pts[j]) > radius for j in keep):
            keep.append(i)
    return keep

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize DETRModel predictions vs. ground truth"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="model/best_model.pth",
        help="Path to saved model checkpoint"
    )
    parser.add_argument(
        "--segments", type=int, default=5,
        help="Number of random segments to visualize"
    )
    parser.add_argument(
        "--seg_len", type=int, default=10,
        help="Length of each segment (in frames)"
    )
    parser.add_argument(
        "--score_thresh", type=float, default=0.8,
        help="Objectness score threshold"
    )
    parser.add_argument(
        "--nms_radius", type=float, default=0.05,
        help="NMS suppression radius (normalized)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="vis_outputs",
        help="Directory to save visualization images"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # dataset parameters
    env_path    = cfg['data']['env_path']
    labels_path = cfg['data']['labels_path']
    seq_len     = cfg['data'].get('seq_len', 24)
    stride      = cfg['data'].get('stride', None)
    H = cfg['model'].get('img_size', 90)
    W = cfg['model'].get('img_size', 90) * 2  # if width != height, adjust here

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset & one batch
    dataset = SequenceDataset(
        env_path, labels_path,
        seq_len=seq_len, stride=stride
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn
    )
    env_seq, gt_seq = next(iter(loader))  # env_seq: (1,T,H,W,C), gt_seq: list of T tensors
    env_seq = env_seq.to(device)

    # load model
    model = DETRModel(
        in_channels=cfg['model']['in_channels'],
        hidden_dim=cfg['model']['hidden_dim'],
        num_layers=cfg['model']['num_layers'],
        num_heads=cfg['model']['num_heads'],
        num_queries=cfg['model']['num_queries'],
        seq_len=seq_len,
        img_size=cfg['model']['img_size']
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # forward pass
    with torch.no_grad():
        logits, coords = model(env_seq)  # logits: (1,T,Q,2), coords: (1,T,Q,2)
    logits = logits[0].cpu()
    coords = coords[0].cpu().numpy()
    gt_seq = gt_seq[0]  # list of T tensors

    # pick random segments
    max_start = seq_len - args.seg_len
    starts = random.sample(range(0, max_start + 1), args.segments)

    # visualize each segment and frame
    for start in starts:
        for t in range(start, start + args.seg_len):
            # compute objectness probabilities
            logit_t = logits[t]         # (Q,2)
            prob = torch.softmax(logit_t, dim=-1)[:, 1].numpy()  # (Q,)

            # threshold + NMS
            mask = prob > args.score_thresh
            pts  = coords[t][mask]      # (N_thresh,2)
            scores = prob[mask]
            keep = nms_points(pts, scores, args.nms_radius)
            pred_pts   = pts[keep]
            pred_scores= scores[keep]

            # convert to pixel coords
            px = pred_pts[:, 0] * (W - 1)
            py = pred_pts[:, 1] * (H - 1)

            # ground-truth points in pixel coords
            gt_norm = gt_seq[t].numpy()
            gt_px = gt_norm[:, 0] * (W - 1)
            gt_py = gt_norm[:, 1] * (H - 1)

            # plot
            plt.figure(figsize=(5, 5))
            ax = plt.gca()
            ax.scatter(gt_px, gt_py, marker='x', s=100,
                       c='red', label='Ground Truth')
            sc = ax.scatter(px, py, c=pred_scores,
                            cmap='viridis', s=80,
                            edgecolors='black', label='Prediction')
            ax.legend(loc='upper right')
            ax.set_title(f"Segment {start}, Frame {t}")
            ax.set_xlim(0, W-1)
            ax.set_ylim(H-1, 0)  # invert y-axis so origin is top-left
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")

            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Confidence")

            plt.tight_layout()
            out_path = os.path.join(
                args.output_dir,
                f"seg{start:02d}_frame{t:02d}.png"
            )
            plt.savefig(out_path, dpi=150)
            plt.close()

if __name__ == "__main__":
    main()
