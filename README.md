# Hurricane_predict

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
