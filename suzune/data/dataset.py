# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Suzune Audio Dataset Implementation
Reads JSON manifests, loads audio files, and tokenizes text independently.
"""
import json
import os
import torch
import soundfile as sf
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset
from suzune.tokenizer.bpe_tokenizer import SuzuneSentencePieceTokenizer

class SuzuneAudioDataset(Dataset):
    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: SuzuneSentencePieceTokenizer,
        sample_rate: int = 16000,
        min_duration: float = 0.1,
        max_duration: float = 16.7,
    ):
        """
        Args:
            manifest_filepath: Path to NeMo-style JSON manifest.
            tokenizer: Initialized SuzuneSentencePieceTokenizer.
            sample_rate: Target sample rate for audio.
            min_duration: Minimum audio length to include.
            max_duration: Maximum audio length to include.
        """
        super().__init__()
        self.manifest_filepath = manifest_filepath
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration

        self.data: List[Dict[str, Any]] = []
        self._load_manifest()

    def _load_manifest(self):
        if not os.path.exists(self.manifest_filepath):
            print(f"WARNING: Manifest {self.manifest_filepath} not found. Dataset will be empty.")
            return

        with open(self.manifest_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                dur = item.get('duration', 0.0)
                if self.min_duration <= dur <= self.max_duration:
                    self.data.append(item)
        print(f"Loaded {len(self.data)} items from {self.manifest_filepath}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        audio_path = item['audio_filepath']
        text = item['text']

        # Load Audio (Assume 16kHz WAV as per NeMo standard preprocessing)
        # Using soundfile for robust WAV reading
        audio, sr = sf.read(audio_path, dtype='float32')
        if sr != self.sample_rate:
            raise ValueError(f"Audio sample rate mismatch: {sr} != {self.sample_rate} in {audio_path}")
        
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        audio_len = torch.tensor(audio_tensor.shape[0], dtype=torch.long)

        # Tokenize text
        tokens = self.tokenizer.text_to_ids(text)
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        tokens_len = torch.tensor(tokens_tensor.shape[0], dtype=torch.long)

        return audio_tensor, audio_len, tokens_tensor, tokens_len
