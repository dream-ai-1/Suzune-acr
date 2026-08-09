# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Suzune Spectrogram Augmentation Module (SpecAugment)
Applies random frequency and time masking to Mel-spectrograms during training.
"""
import torch
import torch.nn as nn


class SuzuneSpectrogramAugmentation(nn.Module):
    """
    SpecAugment implementation (Time & Frequency Masking).
    """

    def __init__(self, rect_masks: int = 5, rect_time: int = 120, rect_freq: int = 50):
        super().__init__()
        self.rect_masks = rect_masks
        self.rect_time = rect_time
        self.rect_freq = rect_freq

    def forward(self, input_spec: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return input_spec

        b, f, t = input_spec.shape
        augmented = input_spec.clone()

        # Frequency Masking
        for _ in range(self.rect_masks):
            f_len = torch.randint(0, self.rect_freq + 1, (1,)).item()
            f0 = torch.randint(0, max(1, f - f_len), (1,)).item()
            augmented[:, f0 : f0 + f_len, :] = 0.0

        # Time Masking
        for _ in range(self.rect_masks):
            t_len = torch.randint(0, self.rect_time + 1, (1,)).item()
            t0 = torch.randint(0, max(1, t - t_len), (1,)).item()
            augmented[:, :, t0 : t0 + t_len] = 0.0

        return augmented
