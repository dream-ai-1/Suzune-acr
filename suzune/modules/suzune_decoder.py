# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Suzune Conv/Linear Decoder Projection Head
"""
import torch
import torch.nn as nn


class SuzuneDecoder(nn.Module):
    """
    Linear projection layer mapping encoder hidden dimension to vocabulary log-probabilities.
    """

    def __init__(self, feat_in: int = 256, num_classes: int = 1024):
        super().__init__()
        self._feat_in = feat_in
        self._num_classes = num_classes
        self.decoder_layers = nn.Sequential(
            nn.Conv1d(feat_in, num_classes, kernel_size=1)
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        # If encoder output is [B, T, D_model], transpose to [B, D_model, T]
        if encoder_output.dim() == 3 and encoder_output.size(1) != self._feat_in:
            encoder_output = encoder_output.transpose(1, 2)
        
        logits = self.decoder_layers(encoder_output)  # [B, num_classes, T]
        logits = logits.transpose(1, 2)  # [B, T, num_classes]
        return logits
