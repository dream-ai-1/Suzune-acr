# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Standalone SentencePiece Tokenizer Training Script for Suzune ASR
"""
import os
import json
import argparse
import sentencepiece as spm


def train_bpe_tokenizer(manifest_filepath: str, output_dir: str, vocab_size: int = 1024):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Extract text from manifest
    text_corpus_path = os.path.join(output_dir, "corpus.txt")
    count = 0
    with open(manifest_filepath, "r", encoding="utf-8") as f_in, open(text_corpus_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if line.strip():
                item = json.loads(line)
                text = item.get("text", "").strip()
                if text:
                    f_out.write(text + "\n")
                    count += 1

    print(f"Extracted {count} text lines to {text_corpus_path}")

    # 2. Train SentencePiece BPE Model
    model_prefix = os.path.join(output_dir, f"suzune_bpe_{vocab_size}")
    cmd = (
        f"--input={text_corpus_path} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type=bpe "
        f"--character_coverage=1.0 "
        f"--pad_id=0 --unk_id=1 --bos_id=-1 --eos_id=-1"
    )
    print(f"Training SentencePiece BPE model with command: {cmd}")
    spm.SentencePieceTrainer.train(cmd)
    print(f"✅ Tokenizer successfully trained and saved to: {model_prefix}.model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SentencePiece BPE Tokenizer")
    parser.add_argument("--manifest_filepath", type=str, required=True, help="Path to combined JSON manifest")
    parser.add_argument("--output_dir", type=str, default="suzune/tokenizer", help="Directory to save trained tokenizer")
    parser.add_argument("--vocab_size", type=int, default=1024, help="Vocabulary size")
    args = parser.parse_args()

    train_bpe_tokenizer(args.manifest_filepath, args.output_dir, args.vocab_size)
