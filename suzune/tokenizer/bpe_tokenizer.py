# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Suzune SentencePiece Tokenizer Wrapper
"""
import os
import sentencepiece as spm
from typing import List


class SuzuneSentencePieceTokenizer:
    """
    Wrapper around SentencePiece BPE tokenizer for English and Hindi text.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SentencePiece model file not found at: {model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def text_to_ids(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def ids_to_text(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()
