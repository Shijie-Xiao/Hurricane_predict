"""Training loops for classification (Timesformer) and heatmap (UNet) models."""

from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from models import build_model, UNet
from dataset import build_loaders, PatchDataset
from torch.utils.data import DataLoader, random_split


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

    best_f1, best_path = -1.0, run_dir / "best.pt"
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

    mtype = cfg["model"]["type"]
    out = Path(tc["out_dir"]) / mtype; out.mkdir(parents=True, exist_ok=True)
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


def _haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two (lat, lon) points in degrees."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def eval_cls(cfg, ckpt_path=None):
    """Evaluate Timesformer + UNet: classification metrics + geographic distance.

    For every true-positive patch, feeds the patch into UNet to get a pixel-level
    heatmap, then computes haversine distance between the heatmap peak (predicted
    genesis location) and the ground-truth hurricane mask centroid.
    """
    mc = cfg["model"]
    dc = cfg["data"]
    tc = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    mtype = mc["type"]
    if ckpt_path is None:
        ckpt_path = str(Path(tc["out_dir"]) / mtype / "run_01" / "best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"[eval] Loaded {mtype} checkpoint: {ckpt_path}")

    unet_ckpt = Path(cfg["unet"]["out_ckpt"])
    if not unet_ckpt.exists():
        raise FileNotFoundError(
            f"UNet checkpoint not found: {unet_ckpt}\n"
            "  Train UNet first: python run.py train --unet"
        )
    in_ch = mc["in_ch"] + 2
    unet_model = UNet(in_ch=in_ch).to(device)
    unet_model.load_state_dict(
        torch.load(str(unet_ckpt), map_location=device, weights_only=True)
    )
    unet_model.eval()
    print(f"[eval] Loaded UNet checkpoint: {unet_ckpt}")

    cache = Path(dc["cache_dir"])
    ph, pw = mc["patch"]
    full_ds = PatchDataset(
        str(cache / "region_env.npy"), str(cache / "region_hurr.npy"),
        seq_len=mc["seq_len"], stride=tc["stride"],
        patch_h=ph, patch_w=pw,
    )
    n_val = max(1, int(len(full_ds) * tc["val_split"]))
    _, val_ds = random_split(
        full_ds, [len(full_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(tc["seed"]),
    )
    loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False)

    lat0 = dc["region"]["lat"][0]
    lon0 = dc["region"]["lon"][0]

    yy_pos = np.linspace(0, 1, ph)
    xx_pos = np.linspace(0, 1, pw)
    pos_grid = np.stack(np.meshgrid(xx_pos, yy_pos, indexing="ij"), axis=0)
    pos_grid = pos_grid.transpose(0, 2, 1).astype(np.float32)

    all_true, all_pred = [], []
    dists = []

    with torch.no_grad():
        for x, y, y_patch, start_idx in tqdm(loader, desc="[eval]"):
            logits = model(x.to(device))
            pred = (torch.sigmoid(logits) > 0.5).long().cpu()

            all_true.append(y.numpy().reshape(-1))
            all_pred.append(pred.numpy().reshape(-1))

            B, T, nh, nw = y.shape
            for b in range(B):
                for t in range(T):
                    for i in range(nh):
                        for j in range(nw):
                            if y[b, t, i, j] != 1 or pred[b, t, i, j] != 1:
                                continue
                            mask = y_patch[b, t, i, j].numpy()
                            if mask.sum() == 0:
                                continue
                            rows, cols = np.where(mask > 0)
                            true_lat = 90.0 - (lat0 + i * ph + rows.mean())
                            true_lon = float(lon0 + j * pw + cols.mean())

                            env_patch = x[b, t, :,
                                          i * ph:(i + 1) * ph,
                                          j * pw:(j + 1) * pw].numpy()
                            inp = np.concatenate([env_patch, pos_grid], axis=0)
                            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
                            hm = torch.sigmoid(unet_model(inp_t)).cpu().numpy()[0]
                            peak = np.unravel_index(hm.argmax(), hm.shape)
                            pred_lat = 90.0 - (lat0 + i * ph + peak[0])
                            pred_lon = float(lon0 + j * pw + peak[1])
                            dists.append(
                                _haversine(pred_lat, pred_lon, true_lat, true_lon)
                            )

    yt = np.concatenate(all_true)
    yp = np.concatenate(all_pred)
    acc, prec, rec, f1 = _metrics(yt, yp)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())

    print(f"\n{'='*50}")
    print(f"[eval] Model: {mtype}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")

    if dists:
        d = np.array(dists)
        print(f"\n[eval] UNet localization distance (TP patches, n={len(d)}):")
        print(f"  Mean:   {d.mean():.1f} km")
        print(f"  Median: {np.median(d):.1f} km")
        print(f"  Std:    {d.std():.1f} km")
        print(f"  Max:    {d.max():.1f} km")
    else:
        print("\n[eval] No true positives — cannot compute distance.")
