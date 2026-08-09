# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Suzune Model Orchestrator
"""
from typing import Optional
from omegaconf import DictConfig
from lightning.pytorch import Trainer

from suzune.models.suzune_ctc_bpe import SuzuneEncDecCTCModelBPE


class SuzuneModel(SuzuneEncDecCTCModelBPE):
    """
    100% Standalone Suzune ASR Model Class.
    Zero external NeMo dependencies - pure PyTorch Speech Recognition Engine.
    """

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None):
        print("Initializing Standalone Suzune Speech AI Model...")
        super().__init__(cfg=cfg, trainer=trainer)
        print(
            f"Suzune Model Successfully Initialized! Encoder: {self.encoder.__class__.__name__}, "
            f"Parameters: {sum(p.numel() for p in self.parameters()):,}"
        )
