# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Standalone Training Entrypoint for Suzune Speech Models (Nano / Small / Base)
Zero external NeMo dependencies - pure PyTorch Lightning orchestration.
"""
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import Trainer

from suzune.models.suzune_ctc_bpe import SuzuneEncDecCTCModelBPE


@hydra.main(config_path="conf", config_name="suzune_nano", version_base=None)
def main(cfg: DictConfig):
    print(f"Hydra Config Loaded Successfully for: {cfg.name}")
    
    # 1. Instantiate PyTorch Lightning Trainer
    # Force checkpoints and logs to be saved in the 'training' directory
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    
    # Check if we are running in Colab (optional: you can adjust paths here)
    training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training"))
    trainer_cfg["default_root_dir"] = training_dir
    
    trainer = Trainer(**trainer_cfg)
    
    # 2. Instantiate 100% Standalone Suzune CTC Model
    model = SuzuneEncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)
    
    print(f"Suzune Model Instantiated with {sum(p.numel() for p in model.parameters()):,} parameters!")
    print(f"Checkpoints and logs will be saved to: {training_dir}")
    print("Ready for GPU Training!")
    
    # 3. Start Training!
    trainer.fit(model)


if __name__ == "__main__":
    main()
