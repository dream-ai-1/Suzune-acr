# Copyright (c) 2026, Suzune Speech AI Architect. All rights reserved.
"""
Suzune Word Error Rate (WER) Metric Calculator Engine
"""
from typing import List


class SuzuneWER:
    """
    Computes Word Error Rate (WER) using Levenshtein distance on words.
    """

    @staticmethod
    def lev_distance(ref: List[str], hyp: List[str]) -> int:
        d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(hyp) + 1):
            d[0][j] = j

        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i - 1] == hyp[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
        return d[len(ref)][len(hyp)]

    def compute(self, references: List[str], hypotheses: List[str]) -> float:
        total_words = 0
        total_errors = 0
        for ref, hyp in zip(references, hypotheses):
            ref_words = ref.strip().split()
            hyp_words = hyp.strip().split()
            total_words += len(ref_words)
            total_errors += self.lev_distance(ref_words, hyp_words)
        return total_errors / max(1, total_words)
