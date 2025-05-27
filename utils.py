# utils.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    """
    Dataset for spatio-temporal storm location sequences.
    Expects:
      - env_path:   numpy file of shape (N, H, W, C) for input features
      - labels_path: numpy file of shape (N, H, W) for binary masks (1=storm)
    """
    def __init__(self, env_path, labels_path, seq_len=24, stride=None):
        assert os.path.isfile(env_path), f"{env_path} not found"
        assert os.path.isfile(labels_path), f"{labels_path} not found"
        env = np.load(env_path)              # (N, H, W, C)
        masks = np.load(labels_path)         # (N, H, W)
        self.seq_len = seq_len
        self.stride = stride or seq_len // 2
        self.env = torch.from_numpy(env).float()
        self.masks = masks
        self.H, self.W = env.shape[1], env.shape[2]
        self.sequences = self._build_sequences()

    def _build_sequences(self):
        """
        Build overlapping sequences of normalized point coordinates.
        """
        sequences = []
        for start in range(0, len(self.env) - self.seq_len + 1, self.stride):
            coords_seq = []
            for t in range(start, start + self.seq_len):
                ys, xs = np.where(self.masks[t] > 0)
                if len(xs) > 0:
                    pts = torch.stack([
                        torch.from_numpy(xs.astype(np.float32) / (self.W - 1)),
                        torch.from_numpy(ys.astype(np.float32) / (self.H - 1))
                    ], dim=1)  # (num_points, 2)
                else:
                    pts = torch.empty((0, 2), dtype=torch.float32)
                coords_seq.append(pts)
            sequences.append(coords_seq)
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns:
          - env_seq: tensor of shape (T, H, W, C)
          - coords_seq: list of T tensors with shape (n_t, 2)
        """
        env_seq = self.env[idx : idx + self.seq_len]
        coords_seq = self.sequences[idx]
        return env_seq, coords_seq

def collate_fn(batch):
    """
    Collate a batch of (env_seq, coords_seq) pairs.
    Returns:
      - envs: tensor (B, T, H, W, C)
      - coords: list of B lists, each containing T tensors (n_t, 2)
    """
    envs, coords = zip(*batch)
    envs = torch.stack(envs, dim=0)
    return envs, coords
