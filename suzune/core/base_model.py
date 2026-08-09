# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Suzune ModelPT Base Class - LightningModule Interface for Suzune Models
"""
import torch
from lightning.pytorch import LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf

from suzune.core.save_restore import SuzuneSaveRestoreConnector


class SuzuneModelPT(LightningModule):
    """
    Base PyTorch Lightning model orchestrator for Suzune Speech Models.
    Handles optimizer configuration, LR scheduling, and model serialization.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        self._cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self._save_restore_connector = SuzuneSaveRestoreConnector()

    @property
    def cfg(self) -> DictConfig:
        return self._cfg

    def configure_optimizers(self):
        """
        Configures AdamW optimizer and CosineAnnealing learning rate scheduler.
        """
        optim_cfg = self._cfg.get("optim", None)
        if optim_cfg is None:
            return torch.optim.AdamW(self.parameters(), lr=1e-3)

        lr = optim_cfg.get("lr", 1e-3)
        weight_decay = optim_cfg.get("weight_decay", 1e-3)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        sched_cfg = optim_cfg.get("sched", None)
        if sched_cfg is None:
            return optimizer

        warmup_steps = sched_cfg.get("warmup_steps", 1000)
        min_lr = sched_cfg.get("min_lr", 1e-5)
        max_epochs = self._cfg.get("max_epochs", 50)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=min_lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def save_to(self, save_path: str):
        """
        Saves current Suzune model instance to a .suzune checkpoint archive.
        """
        self._save_restore_connector.save_to(self, save_path)

    @classmethod
    def restore_from(cls, restore_path: str, map_location: str = "cpu", trainer: Trainer = None):
        """
        Restores Suzune model instance from a .suzune / .nemo checkpoint archive.
        """
        return SuzuneSaveRestoreConnector.restore_from(
            cls, restore_path, map_location=map_location, trainer=trainer
        )
