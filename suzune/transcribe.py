# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Standalone Inference Script for Suzune Speech Models
Transcribes any audio file using a trained PyTorch Lightning checkpoint.
"""
import sys
import os
import argparse
import torch
import soundfile as sf
import librosa
from omegaconf import OmegaConf

from suzune.models.suzune_ctc_bpe import SuzuneEncDecCTCModelBPE
from suzune.tokenizer.bpe_tokenizer import SuzuneSentencePieceTokenizer


def ctc_decode(logits: torch.Tensor, tokenizer: SuzuneSentencePieceTokenizer) -> str:
    """
    Greedy CTC Decoding: Takes logits (T, Vocab), performs argmax, removes consecutive duplicates and blank (0) tokens.
    """
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    dedup = []
    prev = None
    for p in preds:
        if p != prev:
            if p != 0: # 0 is CTC blank token
                dedup.append(int(p))
            prev = p
    return tokenizer.ids_to_text(dedup)


def transcribe(audio_path: str, checkpoint_path: str, tokenizer_path: str):
    print(f"Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1) # Convert stereo to mono
    if sr != 16000:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)

    print("Loading Suzune Model...")
    cfg = OmegaConf.load("suzune/conf/suzune_nano.yaml")
    model = SuzuneEncDecCTCModelBPE(cfg=cfg.model)
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        new_state_dict = {}
        for k, v in state_dict.items():
            key = k
            if key.startswith("model."):
                key = key[6:]
            if key.startswith("_orig_mod."):
                key = key[10:]
            new_state_dict[key] = v
        res = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded weights from {checkpoint_path}")
        if res.missing_keys:
            print(f"Missing keys ({len(res.missing_keys)}): {res.missing_keys[:3]}...")
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found!")

    tokenizer = SuzuneSentencePieceTokenizer(tokenizer_path)
    model.eval()

    with torch.no_grad():
        logits, _ = model(audio_tensor, audio_len)
        text = ctc_decode(logits[0], tokenizer)

    print(f"\n--- Transcribed Text ---")
    print(text)
    print("------------------------\n")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with Suzune ASR")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained .ckpt file")
    parser.add_argument("--tokenizer", type=str, default="suzune/tokenizer/suzune_bpe_1024.model", help="Path to BPE tokenizer model")
    args = parser.parse_args()

    transcribe(args.audio, args.ckpt, args.tokenizer)
