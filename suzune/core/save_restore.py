# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Suzune Checkpoint Save & Restore Connector
Manages packing model weights, YAML configs, and tokenizer models into .suzune / .nemo tar archives.
"""
import os
import tarfile
import tempfile
import torch
from omegaconf import OmegaConf, DictConfig


class SuzuneSaveRestoreConnector:
    """
    Handles saving Suzune models to gzipped tarball archives and restoring from checkpoint archives.
    """

    @staticmethod
    def save_to(model, save_path: str):
        """
        Packs model config, state dict weights, and tokenizer files into a .suzune archive.
        """
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Save YAML Config
            config_yaml = os.path.join(tmpdir, "model_config.yaml")
            OmegaConf.save(model.cfg, config_yaml)

            # 2. Save PyTorch State Dict Weights
            weights_ckpt = os.path.join(tmpdir, "model_weights.ckpt")
            torch.save(model.state_dict(), weights_ckpt)

            # 3. Create Tarball Archive
            with tarfile.open(save_path, "w:gz") as tar:
                tar.add(config_yaml, arcname="model_config.yaml")
                tar.add(weights_ckpt, arcname="model_weights.ckpt")

    @staticmethod
    def restore_from(model_cls, restore_path: str, map_location: str = "cpu", trainer=None):
        """
        Restores Suzune model instance from a .suzune / .nemo checkpoint archive.
        """
        if not os.path.exists(restore_path):
            raise FileNotFoundError(f"Checkpoint archive not found at: {restore_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(restore_path, "r:gz") as tar:
                tar.extractall(tmpdir)

            config_path = os.path.join(tmpdir, "model_config.yaml")
            weights_path = os.path.join(tmpdir, "model_weights.ckpt")

            cfg = OmegaConf.load(config_path)
            model = model_cls(cfg=cfg, trainer=trainer)
            state_dict = torch.load(weights_path, map_location=map_location)
            model.load_state_dict(state_dict)
            return model
