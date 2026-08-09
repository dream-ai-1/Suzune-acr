# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Standalone Training Entrypoint for Suzune Speech Models (Nano / Small / Base)
Zero external NeMo dependencies - pure PyTorch Lightning orchestration.
"""
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback

from suzune.models.suzune_ctc_bpe import SuzuneEncDecCTCModelBPE


class SuzuneConsoleLogger(Callback):
    """
    Custom PyTorch Lightning Callback for guaranteed live console logging in Google Colab.
    Prints epoch, step, and loss every 10 batches.
    """
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == trainer.num_training_batches:
            loss = outputs["loss"].item() if isinstance(outputs, dict) and "loss" in outputs else (outputs.item() if hasattr(outputs, "item") else float(outputs))
            epoch = trainer.current_epoch + 1
            total_epochs = trainer.max_epochs
            total_batches = trainer.num_training_batches
            print(f"🔥 [Epoch {epoch:02d}/{total_epochs:02d}] Step {batch_idx+1:03d}/{total_batches:03d} | Loss: {loss:.4f}", flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        print(f"✅ --- Epoch {epoch} Complete! ---", flush=True)


@hydra.main(config_path="conf", config_name="suzune_nano", version_base=None)
def main(cfg: DictConfig):
    print(f"Hydra Config Loaded Successfully for: {cfg.name}")
    
    # 1. Instantiate PyTorch Lightning Trainer
    # Force checkpoints and logs to be saved in the 'training' directory
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    
    # Check if we are running in Colab (optional: you can adjust paths here)
    training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training"))
    trainer_cfg["default_root_dir"] = training_dir
    trainer_cfg["callbacks"] = [SuzuneConsoleLogger()]
    
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
