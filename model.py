# model.py

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np
from transformers import TimesformerConfig, TimesformerModel

class Matcher(nn.Module):
    """
    Perform matching between predicted points and ground-truth points
    using the Hungarian algorithm.
    """
    @torch.no_grad()
    def forward(self, preds, targets):
        """
        Args:
          - preds: (B, T, Q, 2) predicted coordinates
          - targets: list of B lists, each containing T tensors (n_t, 2)
        Returns:
          - indices: list of B lists, each containing T (pred_idx, tgt_idx) pairs
        """
        B, T, Q, _ = preds.shape
        all_indices = []
        for b in range(B):
            frame_indices = []
            for t in range(T):
                p = preds[b, t].detach().cpu()     # (Q,2)
                gt = targets[b][t].detach().cpu()  # (n_t,2)
                if gt.numel() == 0:
                    frame_indices.append((torch.empty(0, dtype=torch.int64),
                                           torch.empty(0, dtype=torch.int64)))
                else:
                    cost = torch.cdist(p, gt, p=1).numpy()  # (Q, n_t)
                    row_idx, col_idx = linear_sum_assignment(cost)
                    frame_indices.append((
                        torch.tensor(row_idx, dtype=torch.int64),
                        torch.tensor(col_idx, dtype=torch.int64)
                    ))
            all_indices.append(frame_indices)
        return all_indices

class Criterion(nn.Module):
    """
    Compute classification and bounding-box regression loss.
    """
    def __init__(self, matcher, bbox_weight=5.0, cls_weight=1.0, noobj_coef=0.1):
        super().__init__()
        self.matcher = matcher
        self.bbox_weight = bbox_weight
        self.cls_weight = cls_weight
        self.noobj_coef = noobj_coef
        self.l1_loss = nn.L1Loss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, coords, targets):
        """
        Args:
          - logits: (B, T, Q, 2) classification logits
          - coords: (B, T, Q, 2) normalized bbox centers
          - targets: list of B lists of T tensors (n_t, 2)
        Returns:
          - total loss (scalar)
        """
        B, T, Q, _ = coords.shape
        device = coords.device
        indices = self.matcher(coords, targets)
        cls_losses = []
        bbox_losses = []

        for b in range(B):
            for t in range(T):
                logit = logits[b, t]   # (Q,2)
                coord = coords[b, t]   # (Q,2)
                pred_idx, tgt_idx = indices[b][t]

                # classification labels: 1 for object, 0 for background
                labels = torch.zeros(Q, dtype=torch.long, device=device)
                if len(pred_idx) > 0:
                    labels[pred_idx] = 1

                # classification loss
                cls_loss = self.ce_loss(logit.view(-1, 2), labels)
                cls_loss = cls_loss * torch.where(
                    labels == 0,
                    self.noobj_coef,
                    torch.tensor(1.0, device=device)
                )
                cls_losses.append(cls_loss.mean())

                # bbox regression loss for matched points
                if len(pred_idx) > 0:
                    p_coords = coord[pred_idx]
                    gt_coords = targets[b][t][tgt_idx].to(device)
                    box_loss = self.l1_loss(p_coords, gt_coords).sum(dim=1).mean()
                    bbox_losses.append(box_loss)

        loss_cls = torch.stack(cls_losses).mean() * self.cls_weight
        loss_box = (torch.stack(bbox_losses).mean() * self.bbox_weight
                    if bbox_losses else torch.tensor(0., device=device))
        return loss_cls + loss_box

class DETRModel(nn.Module):
    """
    DETR-based model with a 3D conv stem and TimeSformer encoder.
    """
    def __init__(self,
                 in_channels=19,
                 hidden_dim=256,
                 num_layers=6,
                 num_heads=8,
                 num_queries=15,
                 seq_len=24,
                 img_size=90):
        super().__init__()
        self.seq_len = seq_len
        self.num_queries = num_queries

        # 3D convolutional backbone
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim//2, (3,3,3), (1,2,2), (1,1,1)),
            nn.BatchNorm3d(hidden_dim//2),
            nn.GELU(),
            nn.Conv3d(hidden_dim//2, hidden_dim, (3,3,3), (1,2,2), (1,1,1)),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU()
        )

        # TimeSformer encoder
        reduced = img_size // 4
        cfg = TimesformerConfig(
            image_size=reduced,
            patch_size=16,
            num_frames=seq_len,
            num_channels=hidden_dim,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim*4,
            attention_type="divided_space_time"
        )
        self.encoder = TimesformerModel(cfg)

        # query embeddings + transformer decoder
        self.query_embed = nn.Embedding(seq_len * num_queries, hidden_dim)
        dec_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # output heads
        self.class_head = nn.Linear(hidden_dim, 2)
        self.coord_head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        """
        Args:
          - x: tensor of shape (B, T, H, W, C)
        Returns:
          - logits: (B, T, Q, 2)
          - coords: (B, T, Q, 2) normalized in [0,1]
        """
        # reshape to (B, C, T, H, W)
        x = x.permute(0,4,1,2,3)
        features = self.stem(x)                   # (B, D, T, H/4, W/4)
        feat = features.permute(0,2,1,3,4)        # (B, T, D, H', W')
        enc = self.encoder(pixel_values=feat)
        tokens = enc.last_hidden_state[:,1:]      # drop CLS token
        memory = tokens.transpose(0,1)            # (P*T, B, D)

        q = self.query_embed.weight.unsqueeze(1).repeat(1, x.size(0), 1)
        out = self.decoder(q, memory)             # (P*T, B, D)
        out = out.transpose(0,1).view(x.size(0), self.seq_len,
                                      self.num_queries, -1)
        logits = self.class_head(out)
        coords = torch.sigmoid(self.coord_head(out))
        return logits, coords
