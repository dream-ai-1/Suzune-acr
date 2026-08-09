# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Suzune Collate Function
Dynamically pads batches of audio and token tensors.
"""
import torch
from typing import List, Tuple

def suzune_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Collate function to pad audio and tokens for batched inference/training.
    """
    audio_tensors, audio_lens, token_tensors, token_lens = zip(*batch)
    
    # 1. Pad Audio Tensors
    max_audio_len = max([l.item() for l in audio_lens])
    padded_audio = torch.zeros(len(batch), max_audio_len, dtype=torch.float32)
    for i, a in enumerate(audio_tensors):
        padded_audio[i, :a.shape[0]] = a
        
    # 2. Pad Token Tensors
    max_token_len = max([l.item() for l in token_lens])
    # NeMo typically pads tokens with 0, but CTC ignores padding anyway based on lengths
    padded_tokens = torch.zeros(len(batch), max_token_len, dtype=torch.long)
    for i, t in enumerate(token_tensors):
        padded_tokens[i, :t.shape[0]] = t
        
    audio_lengths_tensor = torch.stack(audio_lens)
    token_lengths_tensor = torch.stack(token_lens)
    
    return padded_audio, audio_lengths_tensor, padded_tokens, token_lengths_tensor
