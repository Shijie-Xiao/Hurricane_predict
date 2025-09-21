# Hurricane Genesis Modeling Codebase

End-to-end, configurable codebase for **tropical cyclone genesis (TCG)** modeling, saliency, and causal inference.  
Supports **FULL** dataset training (no ENSO split) and **ENSO-split** training on **NINO/NINA/NET** groups.  
All modules include **default arguments**; no external configs required.

---

## Repository Structure

```
.
├─ run.py               # Unified CLI (prepare/train/spatial/pcmci/…)
├─ train.py             # Generic trainer (9 or 28 channels; FULL or ENSO split)
├─ model.py             # RevisedHierarchicalPatchTimesformer
├─ visualize.py         # Spatial maps + PCMCI causal graphs
├─ utils.py             # Data prepare, channel mapping, geo helpers
├─ requirements.txt
├─ environment.yml
└─ README.md
```

---

## Installation

```bash
# (Recommended) Conda
conda env create -f environment.yml
conda activate hurrlab

# Or pip
pip install -r requirements.txt
```

---

## Data Format

- **Raw inputs (full grid 90×180)**:
  - `HURR_LOC_DAILY.npy` → shape `(T, 90, 180)`
  - `X_28.npy`           → shape `(T, 90, 180, C)` (usually `C=28`)

- **Prepared region (North Atlantic 40×100)**:
  - `region_hurr.npy` `(T, 40, 100)`
  - `region_env.npy`  `(T, 40, 100, Csel)` where `Csel=9` (default) or `28`
  - Optional **ENSO splits** (regional daily mean over cropped area):  
    `X_NINO.npy`, `Y_NINO.npy`, `X_NINA.npy`, `Y_NINA.npy`, `X_NET.npy`, `Y_NET.npy`

---

## Quick Start

### 1) Prepare (crop + optional ENSO split)

```bash
# Crop to North Atlantic (lat 45:85, lon 80:180); select default 9 channels; also export ENSO splits
python run.py prepare --split_enso   --hurr_raw HURR_LOC_DAILY.npy   --env_raw  X_28.npy   --out_dir  .   --in_ch 9   --th_nino 0.5 --th_nina -0.5
```

Outputs:
```
region_hurr.npy
region_env.npy
X_NINO.npy / Y_NINO.npy
X_NINA.npy / Y_NINA.npy
X_NET.npy  / Y_NET.npy
split_meta.json
```

> **28-channel**: set `--in_ch 28` (or any non-9 number), then we keep **all** channels (no name filtering).

---

### 2) Train (FULL dataset by default)

```bash
# Train on FULL dataset using region_env/region_hurr (9 channels if you prepared with --in_ch 9)
python run.py train --in_ch 9 --out_root runs_FULL_9

# Train on FULL dataset (28 channels)
python run.py train --in_ch 28 --out_root runs_FULL_28
```

- `--in_ch` is required to **match** the number of channels in `region_env.npy`.
  - If omitted, `train.py` infers from the file.
- Other important options:
  - `--seq_len` (default 14), `--stride` (default 3), `--batch_size` (default 8)
  - `--runs` (default 3), `--epochs` (default 60)
  - `--lr` (default 1e-4), `--weight_decay` (default 1e-5)
  - `--seed` (default 1337)

---

### 3) Train on ENSO Splits (NINO / NINA / NET)

Use `--use_split {NINO,NINA,NET}` to automatically switch to the corresponding `.npy` pairs:

```bash
# NINA group, 9 channels
python run.py train --use_split NINA --in_ch 9 --out_root runs_NINA_9

# NET group, 28 channels
python run.py train --use_split NET --in_ch 28 --out_root runs_NET_28
```

This maps to:
- `X_<SPLIT>.npy` as env
- `Y_<SPLIT>.npy` as hurr

If files are missing, run `prepare` with `--split_enso` first.

---

### 4) PCMCI Causal Graphs

```bash
# FULL dataset PCMCI
python run.py pcmci --env region_env.npy --hurr region_hurr.npy   --fig_prefix PCMCI_FULL

# NET group PCMCI
python run.py pcmci --env X_NET.npy --hurr Y_NET.npy   --fig_prefix PCMCI_NET
```

- Auto-handles pre-genesis mask (default last 7 days before each genesis).
- If you trained and exported IG saliency, set:
  `--saliency_dir runs_<TAG>_<CH>/run_05/maps_<CH>Channels_3`
- Outputs: high-res `.png` + `.svg` network graphs (SVG labels are editable).

---

## Model & Training

- **Model**: `RevisedHierarchicalPatchTimesformer` (hierarchical patch, factorized ST attention)
- **Target**: per-(20×20) patch binary genesis probability (`BCEWithLogitsLoss`)
- **Metrics**: Accuracy / Precision / Recall / F1 (validation best checkpoint is saved)

---

## Visualization

- `spatial`: quick spatial contour for 2D arrays (e.g., `global_saliency.npy`)
- `pcmci`: builds PCA1 features per channel, pre-genesis mask, pre-screening, PCMCI, and two-threshold graph

---

## Tips

- **OOM**: decrease `--batch_size`, `--seq_len`, or transformer depth/heads in `model.py`.
- **Channel mismatch**: ensure `--in_ch` matches the **last dim** of your chosen env file.
- **Basemap install**: if difficult, switch to Cartopy in `visualize.py` (minor edits).

---

## License

MIT License.
