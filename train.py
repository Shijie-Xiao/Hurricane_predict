# train.py

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from utils import SequenceDataset, collate_fn
from model import DETRModel, Matcher, Criterion

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DETRModel for storm location prediction"
    )
    # Data parameters
    parser.add_argument("--env_path", type=str,
                        default="data/env_data_normalized.npy",
                        help="Path to input features (.npy)")
    parser.add_argument("--labels_path", type=str,
                        default="data/HURR_LOCS.npy",
                        help="Path to labels/masks (.npy)")
    parser.add_argument("--seq_len", type=int, default=24,
                        help="Sequence length")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride for sliding window")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of data for validation")
    # Model parameters
    parser.add_argument("--in_channels", type=int, default=19)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_queries", type=int, default=15)
    parser.add_argument("--img_size", type=int, default=90)
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    parser.add_argument("--bbox_weight", type=float, default=5.0,
                        help="Weight for bbox regression loss")
    parser.add_argument("--cls_weight", type=float, default=1.0,
                        help="Weight for classification loss")
    parser.add_argument("--noobj_coef", type=float, default=0.1,
                        help="Coefficient for no-object loss")
    parser.add_argument("--pct_start", type=float, default=0.1,
                        help="pct_start for OneCycleLR scheduler")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu or cuda")
    parser.add_argument("--checkpoint_path", type=str,
                        default="model/best_model.pth",
                        help="Path to save best checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()

    # Reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Device setup
    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

    # Load dataset and split
    dataset = SequenceDataset(args.env_path,
                              args.labels_path,
                              seq_len=args.seq_len,
                              stride=args.stride)
    total = len(dataset)
    val_size = int(total * args.val_split)
    train_size = total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=collate_fn)

    # Build model, matcher, loss, optimizer, scheduler
    model = DETRModel(in_channels=args.in_channels,
                      hidden_dim=args.hidden_dim,
                      num_layers=args.num_layers,
                      num_heads=args.num_heads,
                      num_queries=args.num_queries,
                      seq_len=args.seq_len,
                      img_size=args.img_size).to(device)
    matcher = Matcher()
    criterion = Criterion(matcher,
                          bbox_weight=args.bbox_weight,
                          cls_weight=args.cls_weight,
                          noobj_coef=args.noobj_coef)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=args.pct_start
    )

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        # Training loop
        model.train()
        train_loss = 0.0
        for env_seq, coords_seq in train_loader:
            env_seq = env_seq.to(device)
            logits, coords = model(env_seq)
            loss = criterion(logits, coords, coords_seq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        avg_train = train_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for env_seq, coords_seq in val_loader:
                env_seq = env_seq.to(device)
                logits, coords = model(env_seq)
                val_loss += criterion(logits, coords, coords_seq).item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # Save best checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"Saved best model to {args.checkpoint_path} (val_loss={best_val_loss:.4f})")

if __name__ == "__main__":
    main()
