# Hurricane Genesis

End-to-end tropical cyclone genesis (TCG) prediction, saliency analysis, and
causal inference. All parameters in a single `config.yaml` -- zero hard-coding.

## Files

```
config.yaml       Configuration (paths / model / training / viz)
models.py         PatchTimesformer · HierTimesformer · UNet
dataset.py        Data prep + Dataset / DataLoader
train.py          Training loops (classification & UNet)
viz.py            Integrated Gradients saliency · PCMCI causal graphs
run.py            CLI entry point
environment.yml   Conda environment spec
requirements.txt  Pip-only dependency list
```

## Environment setup

```bash
conda env create -f environment.yml
conda activate hurricane
```


## Reproduce results

```bash
# 1. Prepare data -- crop ERA5 to North Atlantic, cache to ./cache/
python run.py prepare

# 2. Train PatchTimesformer classifier (default, best prediction accuracy)
python run.py train

# 3. Train HierTimesformer (finer saliency maps, better for visualization)
python run.py train --override model.type=hierarchical

# 4. Train UNet heatmap model (pixel-level localization within a patch)
python run.py train --unet

# 5. Compute Integrated Gradients saliency maps
python run.py saliency

# 6. Run PCMCI causal analysis
python run.py pcmci
```

## Config overrides

Override any `config.yaml` value from the command line:

```bash
python run.py train --override model.type=hierarchical train.epochs=50 train.runs=1
```

## Config sections

| Section | Controls |
|---------|----------|
| `data`  | Raw file paths, transpose, region crop, channel names |
| `model` | Architecture (`patch` / `hierarchical`), dims, depth |
| `unet`  | UNet heatmap hyperparameters |
| `train` | Epochs, batch size, LR, multi-run |
| `viz`   | IG steps, PCMCI thresholds |

## Model architectures

**PatchTimesformer** (`model.type: patch`) -- single-stage factorized
space-time attention on 20x20 patches (10 patches). Best prediction accuracy.

**HierTimesformer** (`model.type: hierarchical`) -- two-stage: 2x2 fine
patches (1000 tokens) pooled to 20x20 (10 tokens) with skip connections.
Better for saliency visualization and causal analysis.

**UNet** -- pixel-level heatmap within a predicted 20x20 patch for precise
genesis location.

## Data

ERA5 reanalysis arrays (NERSC CFS):

| File | Shape |
|------|-------|
| `HURR_LOC_DAILY_GLOBAL.npy` | `(T, 180, 360)` |
| `NORMtraining_data_9VAR_1980-2023.npy` | `(9, T, 181, 360)` |

The `prepare` step transposes, aligns, and crops to a North Atlantic region
of shape `(T, 40, 100, 9)`.
