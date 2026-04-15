"""Model definitions: PatchTimesformer, HierTimesformer, UNet, and build_model factory."""

import torch
import torch.nn as nn


class STBlock(nn.Module):
    """Factorized space-time attention + MLP."""

    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = nn.MultiheadAttention(dim, heads, dropout=drop)
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = nn.MultiheadAttention(dim, heads, dropout=drop)
        self.norm_ff = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, x):
        B, T, P, D = x.shape
        # spatial attention
        s = x.reshape(B * T, P, D).transpose(0, 1)
        o_s, _ = self.attn_s(s, s, s, need_weights=False)
        x = x + self.norm_s(o_s.transpose(0, 1).reshape(B, T, P, D))
        # temporal attention
        t = x.permute(2, 0, 1, 3).reshape(P * B, T, D).transpose(0, 1)
        o_t, _ = self.attn_t(t, t, t, need_weights=False)
        x = x + self.norm_t(o_t.transpose(0, 1).reshape(P, B, T, D).permute(1, 2, 0, 3))
        # feed-forward
        x = x + self.ff(self.norm_ff(x))
        return x


class PatchTimesformer(nn.Module):
    """Single-stage flat-patch Timesformer for TCG classification."""

    def __init__(self, seq_len, in_ch, H, W, ph, pw,
                 dim=128, depth=4, heads=8):
        super().__init__()
        self.seq_len = seq_len
        self.nh = H // ph
        self.nw = W // pw
        self.n_patches = self.nh * self.nw
        self.ph, self.pw = ph, pw

        self.proj = nn.Linear(in_ch * ph * pw, dim)
        self.pos_s = nn.Parameter(torch.zeros(1, 1, self.n_patches, dim))
        self.pos_t = nn.Parameter(torch.zeros(1, seq_len, 1, dim))
        self.blocks = nn.ModuleList([STBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

        nn.init.trunc_normal_(self.pos_s, std=0.02)
        nn.init.trunc_normal_(self.pos_t, std=0.02)

    def forward(self, x):
        """x: (B, T, C, H, W) -> logits (B, T, nh, nw)"""
        B, T, C, H, W = x.shape
        x = x.view(B, T, C, self.nh, H // self.nh, self.nw, W // self.nw)
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        x = x.view(B, T, self.n_patches, -1)
        z = self.proj(x) + self.pos_s + self.pos_t
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        return self.head(z).squeeze(-1).view(B, T, self.nh, self.nw)


class HierTimesformer(nn.Module):
    """Two-stage hierarchical Timesformer (small patches -> large patches)."""

    def __init__(self, seq_len, in_ch, H, W, ph, pw,
                 sp_h=2, sp_w=2, dim_s=64, depth_s=2,
                 dim_l=128, depth_l=4, heads=8):
        super().__init__()
        self.seq_len = seq_len
        self.sp_h, self.sp_w = sp_h, sp_w
        self.ph, self.pw = ph, pw
        self.nsh = H // sp_h
        self.nsw = W // sp_w
        self.nh = H // ph
        self.nw = W // pw
        n_small = self.nsh * self.nsw
        n_large = self.nh * self.nw

        self.proj_s = nn.Linear(in_ch * sp_h * sp_w, dim_s)
        self.pos_s_s = nn.Parameter(torch.zeros(1, 1, n_small, dim_s))
        self.pos_t_s = nn.Parameter(torch.zeros(1, seq_len, 1, dim_s))
        self.blocks_s = nn.ModuleList([STBlock(dim_s, heads) for _ in range(depth_s)])
        self.norm_s = nn.LayerNorm(dim_s)

        self.skip_proj = nn.Linear(dim_s, dim_l)
        self.proj_l = nn.Linear(dim_s, dim_l)
        self.pos_s_l = nn.Parameter(torch.zeros(1, 1, n_large, dim_l))
        self.pos_t_l = nn.Parameter(torch.zeros(1, seq_len, 1, dim_l))
        self.blocks_l = nn.ModuleList([STBlock(dim_l, heads) for _ in range(depth_l)])
        self.norm_l = nn.LayerNorm(dim_l)
        self.head = nn.Linear(dim_l, 1)

        for p in (self.pos_s_s, self.pos_t_s, self.pos_s_l, self.pos_t_l):
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, x):
        """x: (B, T, C, H, W) -> logits (B, T, nh, nw)"""
        B, T, C, H, W = x.shape
        # small-patch stage
        xs = x.view(B, T, C, self.nsh, self.sp_h, self.nsw, self.sp_w)
        xs = xs.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        xs = xs.view(B, T, self.nsh * self.nsw, C * self.sp_h * self.sp_w)
        zs = self.proj_s(xs) + self.pos_s_s + self.pos_t_s
        for blk in self.blocks_s:
            zs = blk(zs)
        zs = self.norm_s(zs)

        # pool small -> large
        zs = zs.view(B, T, self.nh, self.ph // self.sp_h,
                      self.nw, self.pw // self.sp_w, -1)
        zp = zs.mean(dim=5).mean(dim=3)
        zl = zp.view(B, T, self.nh * self.nw, -1)

        skip = self.skip_proj(zl)
        zl = self.proj_l(zl) + skip
        zl = zl + self.pos_s_l + self.pos_t_l
        for blk in self.blocks_l:
            zl = blk(zl)
        zl = self.norm_l(zl)
        return self.head(zl).squeeze(-1).view(B, T, self.nh, self.nw)


class _ConvStack(nn.Module):
    def __init__(self, in_ch, out_ch, n=6):
        super().__init__()
        layers, ch = [], in_ch
        for _ in range(n):
            layers += [nn.Conv2d(ch, out_ch, 3, padding=1),
                       nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """U-Net with MLP bottleneck for patch-level heatmaps."""

    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.enc1 = _ConvStack(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(True),
        )
        self.drop2 = nn.Dropout(0.2)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(5 * 5 * 256, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 5 * 5 * 512)

        self.up1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(True),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(True),
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(True),
        )
        self.drop_out = nn.Dropout(0.2)
        self.out_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        c2 = self.drop2(self.enc2(p1))
        p2 = self.pool2(c2)

        b = self.fc3(self.fc2(self.fc1(p2.flatten(1)))).view(-1, 512, 5, 5)

        d1 = self.dec1(torch.cat([self.up1(b), c2], 1))
        d2 = self.dec2(torch.cat([self.up2(d1), c1], 1))
        return self.out_conv(self.drop_out(d2)).squeeze(1)


def build_model(cfg):
    """Instantiate a Timesformer variant from config."""
    m = cfg["model"]
    H, W = m["grid"]
    ph, pw = m["patch"]
    shared = dict(seq_len=m["seq_len"], in_ch=m["in_ch"], H=H, W=W,
                  ph=ph, pw=pw, heads=m["num_heads"])

    if m["type"] == "patch":
        return PatchTimesformer(**shared, dim=m["embed_dim"], depth=m["depth"])

    if m["type"] == "hierarchical":
        sp = m["small_patch"]
        return HierTimesformer(
            **shared, sp_h=sp[0], sp_w=sp[1],
            dim_s=m["embed_small"], depth_s=m["depth_small"],
            dim_l=m["embed_large"], depth_l=m["depth_large"],
        )

    raise ValueError(f"Unknown model type: {m['type']}")
