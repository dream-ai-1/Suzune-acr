# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Suzune CTC Loss Module Wrapper
"""
import torch
import torch.nn as nn


class SuzuneCTCLoss(nn.Module):
    """
    Computes Connectionist Temporal Classification (CTC) Loss for ASR models.
    """

    def __init__(self, blank: int = 0, reduction: str = "mean", zero_infinity: bool = True):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        # Input log_probs shape: [B, T, V] -> Transpose to [T, B, V] as expected by PyTorch CTCLoss
        log_probs = log_probs.transpose(0, 1)
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
