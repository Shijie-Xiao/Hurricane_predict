#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RevisedHierarchicalPatchTimesformer model definition.
Kept minimal and framework-agnostic so it can be imported by training/visualization code.
"""

import torch
import torch.nn as nn


class FactorizedSTBlock(nn.Module):
    """Factorized space-time attention block with MLP."""
    def __init__(self, E, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm_sp = nn.LayerNorm(E)
        self.attn_sp = nn.MultiheadAttention(E, num_heads, dropout=drop)
        self.norm_tp = nn.LayerNorm(E)
        self.attn_tp = nn.MultiheadAttention(E, num_heads, dropout=drop)
        self.norm_mlp = nn.LayerNorm(E)
        hid = int(E * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(E, hid),
            nn.GELU(),
            nn.Linear(hid, E),
        )

    def forward(self, x):
        # x: (B, T, P, E)
        B, T, Pn, E = x.shape
        # spatial attention
        sp = x.reshape(B*T, Pn, E).transpose(0, 1)  # (P, BT, E)
        o_sp, _ = self.attn_sp(sp, sp, sp, need_weights=False)
        o_sp = o_sp.transpose(0, 1).reshape(B, T, Pn, E)
        x = x + self.norm_sp(o_sp)
        # temporal attention
        tp = x.permute(2, 0, 1, 3).reshape(Pn*B, T, E).transpose(0, 1)  # (T, PB, E)
        o_tp, _ = self.attn_tp(tp, tp, tp, need_weights=False)
        o_tp = o_tp.transpose(0, 1).reshape(Pn, B, T, E).permute(1, 2, 0, 3)
        x = x + self.norm_tp(o_tp)
        # mlp
        x = x + self.mlp(self.norm_mlp(x))
        return x


class RevisedHierarchicalPatchTimesformer(nn.Module):
    """Two-stage hierarchical patch Timesformer for patch-wise binary genesis prediction."""
    def __init__(self,
                 seq_len=14,
                 in_ch=9,
                 H=40, W=100,
                 small_patch_h=2, small_patch_w=2,
                 patch_h=20, patch_w=20,
                 embed_small=64,
                 depth_small=2,
                 embed_large=128,
                 depth_large=4,
                 num_heads=8):
        super().__init__()
        self.seq_len = seq_len
        self.sh, self.sw = small_patch_h, small_patch_w
        self.ph, self.pw = patch_h, patch_w
        self.n_slat = H // small_patch_h
        self.n_slon = W // small_patch_w
        self.n_lat  = H // patch_h
        self.n_lon  = W // patch_w
        P_small = self.n_slat * self.n_slon
        P_large = self.n_lat  * self.n_lon

        # small-level embedding
        self.small_proj = nn.Linear(in_ch * self.sh * self.sw, embed_small)
        self.pos_sp_small = nn.Parameter(torch.zeros(1, 1, P_small, embed_small))
        self.pos_tp_small = nn.Parameter(torch.zeros(1, seq_len, 1, embed_small))
        self.blocks_small = nn.ModuleList([FactorizedSTBlock(embed_small, num_heads) for _ in range(depth_small)])
        self.norm_small   = nn.LayerNorm(embed_small)

        # large-level embedding (pooled from small)
        self.skip_proj  = nn.Linear(embed_small, embed_large)
        self.large_proj = nn.Linear(embed_small, embed_large)
        self.pos_sp_large = nn.Parameter(torch.zeros(1, 1, P_large, embed_large))
        self.pos_tp_large = nn.Parameter(torch.zeros(1, seq_len, 1, embed_large))
        self.blocks_large = nn.ModuleList([FactorizedSTBlock(embed_large, num_heads) for _ in range(depth_large)])
        self.norm_large   = nn.LayerNorm(embed_large)

        # prediction head
        self.head = nn.Linear(embed_large, 1)

        # init
        for p in [self.pos_sp_small, self.pos_tp_small, self.pos_sp_large, self.pos_tp_large]:
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        returns: logits (B, T, n_lat, n_lon)
        """
        B, T, C, Hh, Ww = x.shape
        # small patches
        x_s = x.view(B, T, C, self.n_slat, self.sh, self.n_slon, self.sw)
        x_s = x_s.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        x_s = x_s.view(B, T, self.n_slat*self.n_slon, C*self.sh*self.sw)

        z_s = self.small_proj(x_s)
        z_s = z_s + self.pos_sp_small + self.pos_tp_small
        for blk in self.blocks_small:
            z_s = blk(z_s)
        z_s = self.norm_small(z_s)

        # pool small -> large patches
        z_s2 = z_s.view(B, T, self.n_lat, self.ph//self.sh, self.n_lon, self.pw//self.sw, -1)
        z_pooled = z_s2.mean(dim=5).mean(dim=3)  # average within each large patch
        z_l_small = z_pooled.view(B, T, self.n_lat*self.n_lon, -1)

        skip = self.skip_proj(z_l_small)
        z_l  = self.large_proj(z_l_small) + skip
        z_l  = z_l + self.pos_sp_large + self.pos_tp_large
        for blk in self.blocks_large:
            z_l = blk(z_l)
        z_l = self.norm_large(z_l)

        logits = self.head(z_l).squeeze(-1).view(B, T, self.n_lat, self.n_lon)
        return logits
