# Hurricane Genesis Modeling Codebase

End-to-end codebase for **tropical cyclone genesis (TCG)** prediction, saliency analysis, and causal inference.  
Implements multi-run training with a **RevisedHierarchicalPatchTimesformer**, **Integrated Gradients (IG) saliency maps**, and visualization utilities.  
All modules include **default arguments**, so each script can run without external config files.

---

## Repository structure

```
.
├─ train_9.py           # Main multi-run training + saliency pipeline (9 channels default)
├─ run.py               # Unified CLI entry (optional, can run train_9.py directly)
├─ utils.py             # Helper functions (I/O, reproducibility, etc.)
├─ visualize.py         # Spatial visualization and PCMCI causal analysis
├─ data/                # Data storage (raw and processed)
├─ configs/             # Example config files (YAML) [optional]
└─ README.md
```

---

## Requirements

- Python 3.10+
- PyTorch >= 2.1
- NumPy, SciPy, scikit-learn
- tqdm
- matplotlib
- Basemap (or Cartopy if you prefer)
- tigramite (for PCMCI)
- networkx
- pyyaml

Install with:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
numpy
scipy
matplotlib
tqdm
torch>=2.1
scikit-learn
basemap
tigramite
networkx
pyyaml
```

---

## Data format

- **Raw input arrays**:
  - `HURR_LOC_DAILY.npy` → shape `(T, 90, 180)` (daily hurricane mask)
  - `X_28.npy` → shape `(T, 90, 180, C)` (environment fields, usually `C=28`)
- **Processed region (North Atlantic)**:
  - Cropped to `(T, 40, 100, Csel)` using `lat 45:85, lon 80:180`
  - ENSO-split into `X_NINO.npy`, `X_NINA.npy`, `X_NET.npy` and corresponding `Y_*`

If you want **9-channel runs**, slice the 28-channel arrays before training.  
Default `train_9.py` assumes `(T, 40, 100, 9)` environment input and `(T, 40, 100)` hurricane labels.

---

## Training (Timesformer)

The main entry is [`train_9.py`](train_9.py).  
It supports **multi-run training**, each run stored under `{out_root}/run_XX/`.

Usage:

```bash
# Default run (NINA split, 9 channels)
python train_9.py

# Custom dataset and hyperparameters
python train_9.py --env X_NET.npy --hurr Y_NET.npy --out_root runs_NET_9 --runs 5 --epochs 80 --lr 2e-4
```

Arguments:

| Flag            | Default       | Description |
|-----------------|---------------|-------------|
| `--env`         | `X_NINA.npy`  | Path to environment input |
| `--hurr`        | `Y_NINA.npy`  | Path to hurricane labels |
| `--out_root`    | `runs_NINA_9` | Root folder for run outputs |
| `--runs`        | `5`          | Number of independent runs |
| `--epochs`      | `130`         | Training epochs per run |
| `--lr`          | `1e-4`        | Learning rate |
| `--weight_decay`| `1e-5`        | Weight decay |
| `--seed`        | `1337`        | Base random seed (offset per run) |
| `--no_saliency` | *False*       | Skip saliency computation |

Outputs per run (`runs_*/run_XX/`):
- `best_patch_timesformer.pt` → best checkpoint
- `train_log.txt` → epoch-wise loss/metrics
- `maps_9Channels_3/<ChannelName>/...` → IG saliency maps (`.npy` + `.png`)

Metrics reported:
- Accuracy, Precision, Recall, F1

---

## Saliency maps (Integrated Gradients)

After each training run, **IG saliency maps** are computed automatically (unless `--no_saliency` is set).  
For each channel and each spatial patch:
- `patch_i_j.npy` and `patch_i_j.png`
- `global_saliency.npy` and `global_saliency.png`

Maps are normalized, smoothed with a Gaussian filter, and downsampled for plotting.  
Contour plots are drawn using Basemap.

---

## Visualization (PCMCI causal graphs)

See `visualize.py` for causal analysis:
- Pre-genesis mask (e.g., 7 days before genesis)
- NaN-masked features
- PCMCI with ParCorr test
- Dual α-threshold plots (loose vs strict)

Usage:

```bash
python visualize.py --env X_NET.npy --hurr Y_NET.npy --saliency_dir runs_NET_9/run_05/maps_9Channels_3
```

Outputs:
- `PCMCI_PRE7MASK_*.png` and `.svg` (SVG keeps editable text)

---

## Unified entry (optional)

If you want one CLI instead of multiple scripts, use `run.py`:

```bash
python run.py prepare    # Crop + ENSO split
python run.py train      # Train Timesformer + saliency
python run.py visualize  # Spatial maps
python run.py pcmci      # PCMCI causal graphs
python run.py shap       # (placeholder)
python run.py predict    # (U-Net placeholder)
```

---

## Reproducibility

- All scripts use deterministic seeds.
- Each run has a unique seed offset (`base + run_index`).
- Outputs are structured for experiment tracking.

---

## License

MIT License.
