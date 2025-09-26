# Hurricane Genesis Prediction Codebase

End-to-end, configurable codebase for **tropical cyclone genesis (TCG)** modeling, saliency, and causal inference.  
Supports both **FULL dataset training (no ENSO split)** and **ENSO-split training on NINO/NINA/NET groups**,  
with support for **9-channel subset** or **full 28-channel inputs**.

All modules include **default arguments** so scripts can run out-of-the-box.

---

## Repository Structure

```
.
├─ run.py               # Unified CLI (prepare/train/spatial/pcmci/…)
├─ train.py             # Generic trainer (9 or 28 channels; FULL or ENSO split)
├─ model.py             # RevisedHierarchicalPatchTimesformer
├─ visualize.py         # Spatial maps + PCMCI causal graphs
├─ utils.py             # Data preparation, channel mapping, geo helpers
├─ requirements.txt     # Python pip requirements
├─ environment.yml      # Conda environment specification
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

### Raw Data

- **Source**:  
  The data is derived from **ERA5 reanalysis** provided by **ECMWF (European Centre for Medium-Range Weather Forecasts)**, accessible through the **Copernicus Climate Data Store (CDS)**.  
  - ERA5 provides hourly and daily reanalysis of the global climate system.  
  - Download link: [CDS Portal](https://cds.climate.copernicus.eu/).  

- **License**:  
  ERA5 data is distributed under the **Copernicus License**, which permits free use provided attribution is given:  
  > *“We acknowledge the use of ERA5 data from the Copernicus Climate Change Service (C3S).”*

- **Raw input files**:
  - `HURR_LOC_DAILY.npy` → shape `(T, 90, 180)`  
    Daily hurricane occurrence mask (derived from IBTrACS or a tracking algorithm).  
  - `X_28.npy` → shape `(T, 90, 180, C)`  
    ERA5-derived environmental variables (e.g., Potential Intensity, SST, shear, vorticity, humidity, etc.), with `C=28` channels.  

### Prepared Data (after `prepare`)

- `region_hurr.npy` `(T, 40, 100)`  
  Cropped hurricane mask (North Atlantic).  

- `region_env.npy` `(T, 40, 100, Csel)`  
  Cropped environmental variables. `Csel=9` (default subset) or `28`.  


- **ENSO splits** (optional, by daily mean ENSO index in the cropped region):  
  - `X_NINO.npy`, `Y_NINO.npy`  
  - `X_NINA.npy`, `Y_NINA.npy`  
  - `X_NET.npy`,  `Y_NET.npy`  

Default thresholds: `≥ +0.5 → NINO`, `≤ -0.5 → NINA`, otherwise NET.

---


## Quick Start Workflows

### 1) Prepare data (crop + optional ENSO split)

```bash
# Crop to North Atlantic (lat 45:85, lon 80:180), select default 9 channels, also export ENSO splits
python run.py prepare --split_enso   --hurr_raw HURR_LOC_DAILY.npy   --env_raw  X_28.npy   --out_dir  .   --in_ch 9   --th_nino 0.5 --th_nina -0.5
```

Outputs include:
```
region_hurr.npy
region_env.npy
X_NINO.npy / Y_NINO.npy
X_NINA.npy / Y_NINA.npy
X_NET.npy  / Y_NET.npy
split_meta.json
```

- Use `--in_ch 28` to keep all 28 channels (no filtering).  
- Without `--split_enso`, only `region_env.npy` and `region_hurr.npy` are created.

---

### 2) Train Timesformer

#### Case A: FULL dataset (no ENSO split)

```bash
# Train with 9 channels (region_env.npy last dim=9)
python run.py train --in_ch 9 --out_root runs_FULL_9

# Train with 28 channels
python run.py train --in_ch 28 --out_root runs_FULL_28
```

#### Case B: ENSO splits (NINO / NINA / NET)

```bash
# Train on NINA split (9 channels)
python run.py train --use_split NINA --in_ch 9 --out_root runs_NINA_9

# Train on NET split (28 channels)
python run.py train --use_split NET --in_ch 28 --out_root runs_NET_28
```

- `--in_ch` must equal the channel count of the `.npy` file (9 if subset, 28 if full).  
- Common options:  
  - `--seq_len` (default 14), `--stride` (default 3)  
  - `--batch_size` (default 8), `--epochs` (default 60), `--runs` (default 3)  
  - `--lr` (default 1e-4), `--weight_decay` (default 1e-5), `--seed` (default 1337)

---

### 3) PCMCI Causal Graphs

Run on **FULL dataset**:

```bash
python run.py pcmci --env region_env.npy --hurr region_hurr.npy --fig_prefix PCMCI_FULL
```

Run on **ENSO split**:

```bash
python run.py pcmci --env X_NET.npy --hurr Y_NET.npy --fig_prefix PCMCI_NET
```

- Default: last 7 days pre-genesis mask.  
- Use `--saliency_dir` to point to IG saliency outputs (optional).  
- Output: `.png` and `.svg` graphs (SVG is editable in Illustrator/Inkscape).

---

### 4) Spatial Quick Visualization

```bash
# Plot any 2D array, e.g. saliency map
python run.py spatial --array runs_FULL_9/run_01/maps_9Channels_3/Potential_Intensity/global_saliency.npy   --title "Saliency PI" --out saliency_PI.png
```

Options:  
- `--vmin`, `--vmax`: color range (default -1, 1)  
- `--steps`: number of contour levels (default 21)  
- `--downsample`: reduce resolution for faster plotting (default 2)  
- `--lat0, --lat1, --lon0, --lon1`: region bounds (default 45–85, 80–180)  

---

## File Overview

- **`run.py`**  
  - **Function**: Main CLI, handling subcommands `prepare`, `train`, `pcmci`, `spatial`.  
  - **Implementation**: Uses argparse subparsers to dispatch commands to `utils.prepare_data`, `train.run_training`, `visualize.pcmci_main`, and `visualize.spatial_main`.  
  - **Usage**:  
    ```bash
    python run.py prepare --split_enso
    python run.py train --in_ch 9 --out_root runs_FULL_9
    python run.py pcmci --env X_NET.npy --hurr Y_NET.npy
    python run.py spatial --array some_array.npy
    ```  
  - **Key arguments**: `--use_split`, `--in_ch`, `--out_root`.

- **`train.py`**  
  - **Function**: Training loop for `RevisedHierarchicalPatchTimesformer`. Supports 9 or 28 channels.  
  - **Implementation**:  
    - Defines `HurricanePatchDataset`: creates `(seq_len, C, H, W)` input and `(seq_len, n_lat, n_lon)` labels.  
    - `train_one_run`: single-run training loop with BCEWithLogitsLoss.  
    - `run_training`: multiple runs (default 3), saves best checkpoint by validation F1.  
  - **Usage**:  
    ```bash
    python run.py train --in_ch 9
    python run.py train --use_split NINA --in_ch 28
    ```  
  - **Key parameters**: `--seq_len`, `--stride`, `--batch_size`, `--epochs`, `--lr`, `--runs`.

- **`model.py`**  
  - **Function**: Defines hierarchical patch Timesformer.  
  - **Components**:  
    - `FactorizedSTBlock`: factorized space-time attention + MLP.  
    - `RevisedHierarchicalPatchTimesformer`: two-stage (small patch → large patch) transformer with patch-level logits.  
  - **Usage**:  
    ```python
    from model import RevisedHierarchicalPatchTimesformer
    model = RevisedHierarchicalPatchTimesformer(seq_len=14, in_ch=9)
    logits = model(x)  # (B,T,n_lat,n_lon)
    ```

- **`visualize.py`**  
  - **Function**: Visualization tools for spatial maps and PCMCI causal graphs.  
  - **Components**:  
    - `spatial_main`: quick contour plotting for any 2D `.npy` array.  
    - `pcmci_main`: builds PCA1 features, applies pre-genesis masking, runs PCMCI, and draws graphs (loose vs strict thresholds).  
  - **Usage**:  
    ```bash
    python run.py spatial --array global_saliency.npy --title "Saliency"
    python run.py pcmci --env X_NET.npy --hurr Y_NET.npy
    ```

- **`utils.py`**  
  - **Function**: Data preparation and helper utilities.  
  - **Components**:  
    - `prepare_data`: crops `(90,180)` → `(40,100)`, optional 9-channel subset, optional ENSO split.  
    - Channel name tables: 28-channel full list, 9-channel default subset.  
    - `geo_bounds_and_grid`: generates geographic grid for plotting.  
  - **Usage**:  
    ```python
    import utils
    result = utils.prepare_data("HURR_LOC_DAILY.npy", "X_28.npy", out_dir=".")
    
    ```
- **heatmap.py** – Frame‑level heatmaps on 20×20 patches (U‑Net).
  - `make_patches_20x20(...)` – export patch arrays from region files.
  - `HurricaneFrameHeatmapDataset` – per‑patch dataset with Gaussian‑smoothed target.
  - `UNetFrameLevel_20x20` – U‑Net with MLP bottleneck for 20×20 inputs.
  - `train_heatmap(...)` – training loop with `BCEWithLogitsLoss` and plateau scheduler.
  - `eval_heatmap_frames(...)` – combine Timesformer patch predictions + U‑Net peaks, output multi‑radius maps.


- **`requirements.txt`**  
  - Pip package dependencies (numpy, torch, matplotlib, tigramite, basemap, etc.).  

- **`environment.yml`**  
  - Conda environment specification (recommended for stable Basemap/Tigramite installation).

---

## Parameter Setting Guidelines

- **Channels (`--in_ch`)**: Must match the last dim of env file. Use 9 for subset, 28 for full.  
- **Sequence length (`--seq_len`)**: Longer sequences capture more temporal dynamics but increase memory.  
- **Stride (`--stride`)**: Larger stride → fewer sequences, smaller dataset.  
- **Batch size (`--batch_size`)**: Adjust to GPU memory (e.g. 4–16).  
- **Learning rate (`--lr`)**: Default 1e-4 works for AdamW.  
- **Epochs / Runs**: Increase for stability. Multiple runs recommended.  
- **ENSO thresholds (`--th_nino`, `--th_nina`)**: Default ±0.5; adjust per ENSO definition.  
- **PCMCI (`--tau_max`, `--r0_base`, `--pc_alpha_base`)**: Stricter thresholds = fewer, stronger edges.

---

## Tips

- **GPU OOM**: lower `--batch_size`, shorten `--seq_len`, or reduce model depth in `model.py`.  
- **Basemap install issues**: replace with Cartopy in `visualize.py` if needed.  
- **Saliency not found**: run training with saliency enabled, or skip saliency (PCMCI will fallback).

---

## License

MIT License.
