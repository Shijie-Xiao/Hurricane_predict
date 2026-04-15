"""Training loops for classification (Timesformer) and heatmap (UNet) models."""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from models import build_model, UNet
from dataset import build_loaders


def _metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return acc, prec, rec, f1


def _run_one(run_dir, model, device, train_loader, val_loader, cfg):
    tc = cfg["train"]
    pw = tc.get("pos_weight")
    if pw is not None:
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pw)], device=device))
    else:
        crit = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])

    best_f1, best_path = 0.0, run_dir / "best.pt"
    for ep in range(1, tc["epochs"] + 1):
        model.train()
        loss_sum, yt, yp = 0.0, [], []
        for x, y, *_ in tqdm(train_loader, desc=f"[Train] Ep{ep}", leave=False):
            x, y = x.to(device), y.to(device).float()
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
            yp.extend((torch.sigmoid(logits) > 0.5).long().cpu().view(-1).numpy())
            yt.extend(y.cpu().view(-1).numpy())

        tr = _metrics(np.array(yt), np.array(yp))

        model.eval()
        vloss, yvt, yvp = 0.0, [], []
        with torch.no_grad():
            for x, y, *_ in tqdm(val_loader, desc=f"[ Val ] Ep{ep}", leave=False):
                x, y = x.to(device), y.to(device).float()
                logits = model(x)
                vloss += crit(logits, y).item() * x.size(0)
                yvp.extend((torch.sigmoid(logits) > 0.5).long().cpu().view(-1).numpy())
                yvt.extend(y.cpu().view(-1).numpy())

        va = _metrics(np.array(yvt), np.array(yvp))
        n_tr = len(train_loader.dataset)
        n_va = len(val_loader.dataset)
        print(f"Ep{ep:03d} | TrLoss {loss_sum/n_tr:.4f} F1 {tr[3]:.4f} | "
              f"VaLoss {vloss/n_va:.4f} F1 {va[3]:.4f}")

        if va[3] > best_f1:
            best_f1 = va[3]
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved {best_path} (F1={best_f1:.4f})")

    return best_path, best_f1


def train_cls(cfg):
    """Multi-run Timesformer classification training."""
    tc = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(cfg, for_unet=False)

    out = Path(tc["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    results = []
    for r in range(1, tc["runs"] + 1):
        torch.manual_seed(tc["seed"] + r)
        np.random.seed(tc["seed"] + r)
        run_dir = out / f"run_{r:02d}"; run_dir.mkdir(exist_ok=True)
        model = build_model(cfg).to(device)
        ckpt, f1 = _run_one(run_dir, model, device, train_loader, val_loader, cfg)
        results.append((str(ckpt), f1))

    print("\n[train] Best checkpoints:")
    for p, f1 in results:
        print(f"  {p}  F1={f1:.4f}")
    return results


def train_unet(cfg):
    """Train UNet heatmap model."""
    uc = cfg["unet"]
    seed = cfg["train"]["seed"]
    torch.manual_seed(seed); np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(cfg, for_unet=True)

    sample_x, _ = next(iter(train_loader))
    in_ch = sample_x.shape[1]
    model = UNet(in_ch=in_ch).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=uc["lr"])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    ckpt_path = Path(uc["out_ckpt"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for ep in range(1, uc["epochs"] + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += crit(model(xb), yb).item() * xb.size(0)
        va_loss /= len(val_loader.dataset)
        sched.step(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), ckpt_path)

        print(f"[unet] Ep{ep:03d} train={tr_loss:.6f} val={va_loss:.6f} best={best_val:.6f}")

    print(f"[unet] Best checkpoint -> {ckpt_path}")
    return str(ckpt_path)
