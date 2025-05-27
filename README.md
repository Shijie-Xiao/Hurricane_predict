# Hurricane_predict

## Introduction

Hurricane_predict is a spatio-temporal object detection toolkit for predicting hurricane center locations from multi-channel environmental data. It combines:

- a **3D convolutional stem** to extract short-term spatial features,  
- a **TimeSformer encoder** for long-range temporal modeling,  
- a **Transformer decoder (DETR-style)** for end-to-end point‐based detection.

**Author**: Shijie Xiao, Georgia Tech ECE

## Project Structure

```text
Hurricane_predict/
├── configs/
│ └── default.yaml
├── data/
│ ├── env_data_normalized.npy
│ └── HURR_LOCS.npy
├── model/
│ └── best_model.pth
├── utils.py
├── model.py
├── train.py
├── test.py
├── visualize.py
├── requirements.txt
└── README.md
```

## Prerequisites

- **Python** 3.8  
- **CUDA-capable GPU** with matching drivers  
- **Conda** (or any virtual environment manager)

## Environment Setup

```bash
conda create -n hurri-env python=3.8 -y
conda activate hurri-env
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```
## Configuration

```bash
python train.py --config configs/default.yaml --batch_size 8 --epochs 50
```

### train.py

| Argument            | Type    | Default                        | Description                                    |
| ------------------- | ------- | ------------------------------ | ---------------------------------------------- |
| `--env_path`        | `str`   | `data/env_data_normalized.npy` | Path to input feature `.npy`                   |
| `--labels_path`     | `str`   | `data/HURR_LOCS.npy`           | Path to label `.npy`                           |
| `--seq_len`         | `int`   | `24`                           | Sequence length (frames)                       |
| `--stride`          | `int`   | `None`                         | Sliding window stride (`seq_len//2` if `None`) |
| `--batch_size`      | `int`   | `4`                            | Batch size                                     |
| `--val_split`       | `float` | `0.2`                          | Fraction of data for validation                |
| `--epochs`          | `int`   | `100`                          | Number of training epochs                      |
| `--lr`              | `float` | `1e-4`                         | Learning rate                                  |
| `--weight_decay`    | `float` | `1e-5`                         | Weight decay for optimizer                     |
| `--bbox_weight`     | `float` | `5.0`                          | Loss weight for bbox regression                |
| `--cls_weight`      | `float` | `1.0`                          | Loss weight for classification                 |
| `--noobj_coef`      | `float` | `0.1`                          | Down-weight factor for no-object queries       |
| `--pct_start`       | `float` | `0.1`                          | `pct_start` for OneCycleLR scheduler           |
| `--seed`            | `int`   | `None`                         | Random seed                                    |
| `--device`          | `str`   | `None`                         | `cpu` or `cuda`                                |
| `--checkpoint_path` | `str`   | `model/best_model.pth`         | Path to save best checkpoint                   |

### visualize.py
| Argument         | Type    | Default                | Description                                     |
| ---------------- | ------- | ---------------------- | ----------------------------------------------- |
| `--config`       | `str`   | `configs/default.yaml` | Path to YAML config file                        |
| `--checkpoint`   | `str`   | `model/best_model.pth` | Path to saved model checkpoint                  |
| `--segments`     | `int`   | `5`                    | Number of random segments to visualize          |
| `--seg_len`      | `int`   | `10`                   | Length of each segment (in frames)              |
| `--score_thresh` | `float` | `0.8`                  | Objectness score threshold                      |
| `--nms_radius`   | `float` | `0.05`                 | NMS suppression radius (normalized coordinates) |
| `--output_dir`   | `str`   | `vis_outputs`          | Directory to save visualization images          |

python visualize.py \
  --config configs/default.yaml \
  --checkpoint model/best_model.pth \
  --segments 3 --seg_len 8



