# Hurricane Genesis

End-to-end tropical cyclone genesis (TCG) prediction, explainability, and causal inference.

Code for: *Physics-Based Machine Learning for Tropical Cyclone Genesis Comprehension in the North Atlantic Basin*

## Files

```
config.yaml   All paths, hyperparameters, and output settings
models.py     PatchTimesformer · HierTimesformer · UNet
dataset.py    Data preparation + Dataset / DataLoader
train.py      Training loops (classification & UNet) + evaluation
viz.py        SHAP · Integrated Gradients saliency · PCMCI causal graphs
run.py        CLI entry point
```

## Setup

```bash
conda env create -f environment.yml
conda activate hurricane
```

### Prediction

```bash
python run.py predict
```

prepare → train PatchTimesformer → train UNet → eval → SHAP

### Visualization

```bash
python run.py visualize
```

prepare → train HierTimesformer → saliency (IG) → PCMCI

### Individual steps

```bash
python run.py prepare
python run.py train [--unet]
python run.py eval [--ckpt path]
python run.py shap [--ckpt path]
python run.py saliency [--ckpt path]
python run.py pcmci [--saliency-dir path]
```

<!-- ## Output locations

| Output | Path | Description |
|--------|------|-------------|
| Cached data | `./cache/` | Cropped ERA5 region arrays |
| Timesformer checkpoints | `checkpoints/<type>/run_*/best.pt` | Best model per run |
| UNet checkpoint | `checkpoints/best_unet.pt` | Best UNet weights |
| SHAP values | `shap_outputs/shap_timesformer.npy` | Raw SHAP values (Timesformer) |
| SHAP values | `shap_outputs/shap_unet.npy` | Raw SHAP values (UNet) |
| SHAP plots | `shap_outputs/shap_*_beeswarm.png` | Per-channel importance beeswarm |
| Saliency maps | `saliency_maps/<channel>/` | IG maps (`.npy` + `.png`) per patch |
| PCMCI graph | `PCMCI.png` / `PCMCI.svg` | Causal graph visualization |

All output paths are configurable in `config.yaml`. -->
