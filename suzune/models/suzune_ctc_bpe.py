# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
100% Standalone Suzune CTC BPE Model Orchestrator
Zero external NeMo dependencies - pure PyTorch Speech Recognition Engine.
"""
import torch
from omegaconf import DictConfig
from lightning.pytorch import Trainer

from suzune.core.base_model import SuzuneModelPT
from suzune.audio.preprocessor import SuzuneAudioToMelSpectrogramPreprocessor
from suzune.audio.spec_augment import SuzuneSpectrogramAugmentation
from suzune.modules.suzune_encoder import SuzuneEncoder
from suzune.modules.suzune_decoder import SuzuneDecoder
from suzune.losses.ctc_loss import SuzuneCTCLoss
from suzune.metrics.wer import SuzuneWER
from suzune.tokenizer.bpe_tokenizer import SuzuneSentencePieceTokenizer
from suzune.data.dataset import SuzuneAudioDataset
from suzune.data.collate import suzune_collate_fn
from torch.utils.data import DataLoader


class SuzuneEncDecCTCModelBPE(SuzuneModelPT):
    """
    100% Standalone Suzune End-to-End CTC Model.
    Integrates preprocessor, SpecAugment, FastConformer Encoder, Conv Decoder, CTC Loss, and WER Metric.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        
        # Tokenizer Setup
        tokenizer_dir = self._cfg.get("tokenizer", {}).get("dir", None)
        if tokenizer_dir:
            self.tokenizer = SuzuneSentencePieceTokenizer(str(tokenizer_dir))
        else:
            self.tokenizer = None
            
        # Dataloaders
        self._train_dl = None
        self._val_dl = None
        self._test_dl = None

        # 1. Feature Preprocessor
        self.preprocessor = SuzuneAudioToMelSpectrogramPreprocessor(
            sample_rate=self._cfg.preprocessor.sample_rate,
            features=self._cfg.preprocessor.features,
            window_size=self._cfg.preprocessor.window_size,
            window_stride=self._cfg.preprocessor.window_stride,
        )

        # 2. SpecAugment
        self.spec_augment = SuzuneSpectrogramAugmentation()

        # 3. FastConformer Encoder Engine
        self.encoder = SuzuneEncoder(
            feat_in=self._cfg.encoder.feat_in,
            n_layers=self._cfg.encoder.n_layers,
            d_model=self._cfg.encoder.d_model,
            n_heads=self._cfg.encoder.n_heads,
            conv_kernel_size=self._cfg.encoder.get("conv_kernel_size", 9),
            dropout=self._cfg.encoder.get("dropout", 0.1),
        )

        # 4. Decoder Projection Head
        vocab_size = self._cfg.decoder.get("num_classes", 1024)
        if vocab_size < 1:
            vocab_size = 1024
        self.decoder = SuzuneDecoder(
            feat_in=self._cfg.encoder.d_model,
            num_classes=vocab_size,
        )

        # 5. CTC Loss Engine
        self.loss = SuzuneCTCLoss()

        # 6. WER Metric Calculator
        self.wer = SuzuneWER()

    def forward(self, input_signal: torch.Tensor, input_signal_length: torch.Tensor):
        processed_signal, processed_lengths = self.preprocessor(
            input_signal=input_signal, length=input_signal_length
        )
        if self.training:
            processed_signal = self.spec_augment(processed_signal)

        encoded, encoded_lengths = self.encoder(
            audio_signal=processed_signal, length=processed_lengths
        )
        logits = self.decoder(encoder_output=encoded)
        return logits, encoded_lengths

    def training_step(self, batch, batch_idx):
        signal, signal_len, targets, target_len = batch
        logits, logit_len = self.forward(signal, signal_len)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss_val = self.loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=logit_len,
            target_lengths=target_len,
        )
        self.log("train_loss", loss_val, on_step=True, on_epoch=True, prog_bar=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        signal, signal_len, targets, target_len = batch
        logits, logit_len = self.forward(signal, signal_len)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss_val = self.loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=logit_len,
            target_lengths=target_len,
        )
        self.log("val_loss", loss_val, on_epoch=True, prog_bar=True)
        return loss_val

    def setup_training_data(self, train_data_config: DictConfig):
        if self.tokenizer is None:
            print("WARNING: Tokenizer is None. Cannot setup training data.")
            return
            
        dataset = SuzuneAudioDataset(
            manifest_filepath=train_data_config.manifest_filepath,
            tokenizer=self.tokenizer,
            sample_rate=train_data_config.sample_rate,
            min_duration=train_data_config.min_duration,
            max_duration=train_data_config.max_duration,
        )
        self._train_dl = DataLoader(
            dataset=dataset,
            batch_size=train_data_config.batch_size,
            shuffle=train_data_config.shuffle,
            num_workers=train_data_config.num_workers,
            pin_memory=train_data_config.get("pin_memory", False),
            collate_fn=suzune_collate_fn
        )

    def setup_validation_data(self, val_data_config: DictConfig):
        if self.tokenizer is None or not val_data_config.manifest_filepath:
            return
            
        dataset = SuzuneAudioDataset(
            manifest_filepath=val_data_config.manifest_filepath,
            tokenizer=self.tokenizer,
            sample_rate=val_data_config.sample_rate,
            min_duration=0.0,
            max_duration=9999.0, # no filtering for validation
        )
        self._val_dl = DataLoader(
            dataset=dataset,
            batch_size=val_data_config.batch_size,
            shuffle=val_data_config.shuffle,
            num_workers=val_data_config.num_workers,
            pin_memory=val_data_config.get("pin_memory", False),
            collate_fn=suzune_collate_fn
        )
        
    def setup_test_data(self, test_data_config: DictConfig):
        # Implementation is identical to validation for now
        self.setup_validation_data(test_data_config)
        self._test_dl = self._val_dl

    def train_dataloader(self):
        if self._train_dl is None:
            self.setup_training_data(self._cfg.train_ds)
        return self._train_dl

    def val_dataloader(self):
        if self._val_dl is None:
            self.setup_validation_data(self._cfg.validation_ds)
        return self._val_dl

    def test_dataloader(self):
        if self._test_dl is None:
            if "test_ds" in self._cfg and self._cfg.test_ds is not None:
                self.setup_test_data(self._cfg.test_ds)
        return self._test_dl
