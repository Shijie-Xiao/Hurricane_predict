#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic trainer for RevisedHierarchicalPatchTimesformer.

- Supports 9 or 28 input channels (or any C inferred from env data).
- Defaults to training on FULL dataset (no ENSO split):
    env=region_env.npy, hurr=region_hurr.npy
- Patch-wise binary genesis prediction with BCEWithLogitsLoss.
- Saves best checkpoint per run by validation F1.

This script is self-contained (no external configs required).
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm

from model import RevisedHierarchicalPatchTimesformer


# ----------------- Dataset -----------------
class HurricanePatchDataset(Dataset):
    """
    Builds sequences and patch labels:
      - Input x: (T, C, H, W)
      - Label y: (T, n_lat_p, n_lon_p) where each patch is 1 if any genesis pixel exists
    """
    def __init__(self, env_path, hurr_path,
                 seq_len=14, stride=3,
                 in_ch=None,
                 H=40, W=100, patch_h=20, patch_w=20):
        env_all  = np.nan_to_num(np.load(env_path),  nan=0.0, posinf=0.0, neginf=0.0)  # (D,H,W,C)
        hurr_all = np.nan_to_num(np.load(hurr_path), nan=0.0, posinf=0.0, neginf=0.0)  # (D,H,W)
        assert env_all.ndim == 4 and hurr_all.ndim == 3, "Check shapes of env/hurr arrays."

        D, H0, W0, C = env_all.shape
        assert (H0, W0) == (H, W), f"Expected ({H},{W}), got ({H0},{W0})"
        if in_ch is None:
            in_ch = C
        assert C == in_ch, f"--in_ch={in_ch} but env has C={C}"

        self.env  = torch.from_numpy(env_all).float()   # (D,H,W,C)
        self.hurr = torch.from_numpy(hurr_all).float()  # (D,H,W)
        self.seq_len = seq_len
        self.stride  = stride
        self.in_ch   = in_ch
        self.H, self.W = H, W
        self.patch_h, self.patch_w = patch_h, patch_w
        self.n_lat_p = H // patch_h
        self.n_lon_p = W // patch_w

        # sequence starts; keep those that contain at least one positive frame
        all_starts = list(range(0, D - seq_len + 1, stride))
        keep = []
        for s in all_starts:
            window = self.hurr[s:s+seq_len].reshape(seq_len, -1).sum(dim=1)  # (T,)
            if (window > 0).any():
                keep.append(s)
        if len(keep) == 0:
            # allow empty; still produce sequences
            keep = all_starts
        self.starts = keep

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        x = self.env[s:s+self.seq_len].permute(0, 3, 1, 2)  # (T,C,H,W)
        h = self.hurr[s:s+self.seq_len]                     # (T,H,W)

        # binarize per (20x20) patch
        T = self.seq_len
        ph, pw = self.patch_h, self.patch_w
        nlp, nmp = self.n_lat_p, self.n_lon_p
        h_patches = (h.reshape(T, nlp, ph, nmp, pw).permute(0,1,3,2,4))  # (T, nlp, nmp, ph, pw)
        y = (h_patches.reshape(T, nlp, nmp, ph*pw).sum(dim=-1) > 0).long()  # (T, nlp, nmp)
        return x, y


# ----------------- Metrics -----------------
def acc_pr_re_f1(y_true, y_pred):
    """Compute accuracy, precision, recall, F1 for 0/1 arrays."""
    tp = int(((y_true==1)&(y_pred==1)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    tn = int(((y_true==0)&(y_pred==0)).sum())
    acc  = (tp+tn)/max(1,(tp+tn+fp+fn))
    prec = tp/max(1,(tp+fp))
    rec  = tp/max(1,(tp+fn))
    f1   = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    return acc,prec,rec,f1


# ----------------- One-run training -----------------
def train_one_run(run_dir, device, ds, seq_len, in_ch, H, W, patch_h, patch_w,
                  epochs=130, batch_size=8, lr=1e-4, weight_decay=1e-5, seed=1337):
    """Train a single run and save the best checkpoint by validation F1."""
    # split
    n_total = len(ds)
    n_train = max(1, int(n_total*0.9))
    n_val   = max(1, n_total - n_train)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    # model
    model = RevisedHierarchicalPatchTimesformer(
        seq_len=seq_len, in_ch=in_ch, H=H, W=W,
        small_patch_h=2, small_patch_w=2,
        patch_h=patch_h, patch_w=patch_w,
        embed_small=64, depth_small=2,
        embed_large=128, depth_large=4,
        num_heads=8
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    best_ckpt = run_dir / "best_patch_timesformer.pt"

    for ep in range(1, epochs+1):
        # train
        model.train(); tr_loss, ytr, ypr = 0.0, [], []
        for x, y in tqdm(train_loader, desc=f"[Train] Ep{ep}", leave=False):
            x = x.to(device)
            y = y.to(device).float()  # (B,T,nlat,nlon)

            logits = model(x)                 # (B,T,nlat,nlon)
            loss   = crit(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item()*x.size(0)
            preds = (torch.sigmoid(logits)>0.5).long().cpu().view(-1).numpy()
            ypr.extend(preds); ytr.extend(y.cpu().view(-1).numpy())

        tr_acc,tr_pre,tr_rec,tr_f1 = acc_pr_re_f1(np.array(ytr), np.array(ypr))

        # val
        model.eval(); va_loss, yva, ypv = 0.0, [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"[ Val ] Ep{ep}", leave=False):
                x = x.to(device); y = y.to(device).float()
                logits = model(x)
                loss   = crit(logits, y)
                va_loss += loss.item()*x.size(0)
                preds = (torch.sigmoid(logits)>0.5).long().cpu().view(-1).numpy()
                ypv.extend(preds); yva.extend(y.cpu().view(-1).numpy())

        va_acc,va_pre,va_rec,va_f1 = acc_pr_re_f1(np.array(yva), np.array(ypv))
        print(f"Ep{ep:03d} | TrLoss {tr_loss/len(train_ds):.4f} F1 {tr_f1:.4f} | "
              f"VaLoss {va_loss/len(val_ds):.4f} F1 {va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> Saved best: {best_ckpt} (F1={va_f1:.4f})")

    return best_ckpt, best_f1


# ----------------- Multi-run wrapper (for run.py) -----------------
def run_training(env_path="region_env.npy",
                 hurr_path="region_hurr.npy",
                 out_root="runs_FULL_auto",
                 runs=3, epochs=60, lr=1e-4, weight_decay=1e-5, seed=1337,
                 seq_len=14, stride=3, batch_size=8, in_ch=None):
    """Convenience entry callable by run.py with explicit arguments."""
    env = np.load(env_path)
    T, H, W, C = env.shape
    if in_ch is None:
        in_ch = C

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_training] Device: {device} | Env C={C} | Using in_ch={in_ch}")

    ds = HurricanePatchDataset(
        env_path=env_path, hurr_path=hurr_path,
        seq_len=seq_len, stride=stride,
        in_ch=in_ch, H=H, W=W, patch_h=20, patch_w=20
    )

    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    bests = []
    for r in range(1, runs+1):
        torch.manual_seed(seed + r)
        np.random.seed(seed + r)

        run_dir = out_root / f"run_{r:02d}"; run_dir.mkdir(parents=True, exist_ok=True)
        ckpt, f1 = train_one_run(
            run_dir, device, ds, seq_len, in_ch, H, W, 20, 20,
            epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, seed=seed + r
        )
        bests.append((str(ckpt), f1))

    print("\n[run_training] Best checkpoints:")
    for p, f1 in bests:
        print(f" - {p} | F1={f1:.4f}")


# ----------------- CLI main -----------------
def main():
    ap = argparse.ArgumentParser(description="Train Timesformer on FULL dataset (no ENSO split by default).")
    ap.add_argument("--env", type=str, default="region_env.npy", help="Path to env array (T,40,100,C)")
    ap.add_argument("--hurr", type=str, default="region_hurr.npy", help="Path to hurr mask (T,40,100)")
    ap.add_argument("--in_ch", type=int, default=None, help="If None, inferred from env last dim")
    ap.add_argument("--seq_len", type=int, default=14)
    ap.add_argument("--stride",  type=int, default=3)
    ap.add_argument("--runs",    type=int, default=3)
    ap.add_argument("--epochs",  type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr",      type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--seed",    type=int, default=1337)
    ap.add_argument("--out_root", type=str, default="runs_FULL_auto")
    args = ap.parse_args()

    run_training(
        env_path=args.env, hurr_path=args.hurr, out_root=args.out_root,
        runs=args.runs, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        seed=args.seed, seq_len=args.seq_len, stride=args.stride,
        batch_size=args.batch_size, in_ch=args.in_ch
    )


if __name__ == "__main__":
    main()
