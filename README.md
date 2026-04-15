# Hurricane Genesis

End-to-end tropical cyclone genesis (TCG) prediction, saliency analysis, and
causal inference.  

Code for Paper：Physics-Based Machine Learning for Tropical Cyclone3
Genesis Comprehension in the North Atlantic Basin4

## Files

```
config.yaml       Configuration (paths / model / training / viz)
models.py         PatchTimesformer · HierTimesformer · UNet
dataset.py        Data prep + Dataset / DataLoader
train.py          Training loops (classification & UNet) + evaluation
viz.py            Integrated Gradients saliency · PCMCI causal graphs
run.py            CLI entry point (two automated pipelines + individual steps)
environment.yml   Conda environment spec
requirements.txt  Pip-only dependency list
```

## Environment setup

```bash
conda env create -f environment.yml
conda activate hurricane
```


## Prediction

```bash
python run.py predict
```

1. **prepare** — crop ERA5 data → `./cache/`
2. **train PatchTimesformer** — best classification accuracy
3. **train UNet** — sub-patch pixel-level localization
4. **eval** — classification metrics (F1, precision, recall) + UNet
   localization distance: haversine distance from UNet heatmap peak to
   actual genesis centroid (km)

## Visualization

```bash
python run.py visualize
```

1. **prepare** — crop ERA5 data → `./cache/`
2. **train HierTimesformer** — finer spatial gradients for saliency
3. **saliency** — Integrated Gradients maps (per-channel + per-patch)
4. **PCMCI** — causal inference graph between environmental variables

**Estimated runtime** (default config: 150 epochs × 3 runs, ig_steps=20).
The model is small and saliency is bottlenecked by sequential Python loops,
so GPU acceleration is modest (~2×) compared to large-model workloads.

| Step | CPU | Single A100 GPU |
|------|-----|----------------|
| train HierTimesformer | ~12 h | ~5 h |
| saliency (9ch × 20 IG steps × val set) | ~8 h | ~5 h |
| PCMCI (pure numpy, no GPU) | ~30 min | ~30 min |
| **Total** | **~20 h** | **~10 h** |

To shorten for a quick test:

```bash
python run.py visualize --override train.epochs=10 train.runs=1 viz.saliency.ig_steps=5
```