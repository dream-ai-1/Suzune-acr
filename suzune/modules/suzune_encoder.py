# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Suzune FastConformer Encoder Module (8x Downsampling + Conformer Blocks)
Self-contained implementation for Suzune Model Family.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class SuzuneConvSubsampling8x(nn.Module):
    """
    8x Time-Downsampling Convolutional Frontend for FastConformer.
    Reduces time resolution by 8x using 3 depthwise-separable conv layers with stride 2.
    """

    def __init__(self, in_channels: int = 80, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            # Conv 1: Stride 2 (2x downsampling)
            nn.Conv2d(1, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Conv 2: Stride 2 (4x total downsampling)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Conv 3: Stride 2 (8x total downsampling)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.out_proj = nn.Linear(out_channels * (in_channels // 8), out_channels)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shape: [B, D_mel, T_spec] -> Add channel dim: [B, 1, D_mel, T_spec]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.conv(x)  # [B, out_channels, D_mel//8, T_spec//8]
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # [B, T_sub, C*F]
        x = self.out_proj(x)  # [B, T_sub, D_model]
        
        # Calculate downsampled length
        subsampled_lengths = ((lengths - 1) // 2 - 1) // 2
        subsampled_lengths = (subsampled_lengths - 1) // 2 + 1
        subsampled_lengths = torch.clamp(subsampled_lengths, min=1)
        
        return x, subsampled_lengths


class SuzuneConformerBlock(nn.Module):
    """
    Standard Conformer Block (Macaron-style FFN + Self-Attention + Depthwise Conv + FFN).
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, conv_kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        # 1. First FFN (1/2 Macaron)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        
        # 2. Multi-Head Self-Attention
        self.norm_att = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        
        # 3. Depthwise Conv Module
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )
        
        # 4. Second FFN (1/2 Macaron)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN 1
        x = x + 0.5 * self.ffn1(x)
        
        # Self-Attention
        att_in = self.norm_att(x)
        att_out, _ = self.attn(att_in, att_in, att_in)
        x = x + att_out
        
        # Depthwise Conv
        conv_in = self.norm_conv(x).transpose(1, 2)  # [B, D_model, T]
        conv_out = self.conv(conv_in).transpose(1, 2) # [B, T, D_model]
        x = x + conv_out
        
        # FFN 2
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class SuzuneEncoder(nn.Module):
    """
    Complete Suzune FastConformer Encoder Engine.
    Self-contained 12-layer FastConformer Encoder for Suzune Nano (~25M parameters).
    """

    def __init__(
        self,
        feat_in: int = 80,
        n_layers: int = 12,
        d_model: int = 256,
        n_heads: int = 4,
        conv_kernel_size: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._feat_out = d_model
        self.subsampling = SuzuneConvSubsampling8x(in_channels=feat_in, out_channels=d_model)
        self.layers = nn.ModuleList([
            SuzuneConformerBlock(d_model=d_model, n_heads=n_heads, conv_kernel_size=conv_kernel_size, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, out_len = self.subsampling(audio_signal, length)
        for layer in self.layers:
            x = layer(x)
        return x, out_len
