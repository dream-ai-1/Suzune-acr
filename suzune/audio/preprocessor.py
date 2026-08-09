# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Suzune Audio-to-Mel Spectrogram Preprocessor Engine
Pure PyTorch implementation of STFT, Mel-filterbanks, and log-compression.
"""
import math
import torch
import torch.nn as nn
from typing import Tuple


class SuzuneAudioToMelSpectrogramPreprocessor(nn.Module):
    """
    Computes STFT and Mel-scale filterbank spectrograms from raw 16kHz audio waveforms.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_size: float = 0.025,
        window_stride: float = 0.01,
        features: int = 80,
        n_fft: int = 512,
        normalize: str = "per_feature",
        log: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = int(sample_rate * window_size)
        self.hop_length = int(sample_rate * window_stride)
        self.features = features
        self.n_fft = n_fft
        self.log = log
        self.normalize = normalize

        # Create Hann Window
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

        # Create Mel Filterbank Matrix
        mel_fb = self._create_mel_filterbank(
            sample_rate=sample_rate, n_fft=n_fft, n_mels=features
        )
        self.register_buffer("mel_filterbank", mel_fb)

    def _create_mel_filterbank(self, sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
        """
        Calculates triangular Mel-filterbank matrix.
        """
        def hz_to_mel(hz):
            return 2595.0 * math.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        min_mel = hz_to_mel(0.0)
        max_mel = hz_to_mel(sample_rate / 2.0)
        mel_pts = torch.linspace(min_mel, max_mel, n_mels + 2)
        hz_pts = mel_to_hz(mel_pts)
        bins = torch.floor((n_fft + 1) * hz_pts / sample_rate).long()

        fb = torch.zeros(n_mels, n_fft // 2 + 1)
        for i in range(n_mels):
            fb[i, bins[i]:bins[i+1]] = torch.linspace(0, 1, bins[i+1] - bins[i])
            fb[i, bins[i+1]:bins[i+2]] = torch.linspace(1, 0, bins[i+2] - bins[i+1])
        return fb

    def forward(self, input_signal: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shape: [B, T_samples]
        stft = torch.stft(
            input_signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        # Power spectrogram
        spectrogram = torch.abs(stft) ** 2  # [B, n_fft//2 + 1, T_spec]

        # Apply Mel Filterbank
        mel_spec = torch.matmul(self.mel_filterbank, spectrogram)  # [B, features, T_spec]

        if self.log:
            mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        if self.normalize == "per_feature":
            mean = mel_spec.mean(dim=-1, keepdim=True)
            std = mel_spec.std(dim=-1, keepdim=True) + 1e-5
            mel_spec = (mel_spec - mean) / std

        # Calculate spectrogram sequence lengths
        out_len = torch.div(length, self.hop_length, rounding_mode="floor") + 1
        return mel_spec, out_len
