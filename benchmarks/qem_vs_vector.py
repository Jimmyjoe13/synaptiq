"""Benchmark reproductible Q-EM contre un classement vectoriel top-k.

Input JSONL: {"expected_ids": ["m1"], "candidates": [{"id": "m1",
"score": 0.9, "content": "...", "type": "semantic", "subtype": "fact"}]}.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "core"))

from synaptiq_core.qem import collapse_by_utility


def recall(selected: set[str], expected: set[str]) -> float:
    return len(selected & expected) / len(expected) if expected else 1.0


def evaluate(path: Path, max_tokens: int) -> dict:
    qem_recalls, vector_recalls, qem_tokens, vector_tokens = [], [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        candidates = {candidate["id"]: candidate for candidate in item["candidates"]}
        expected = set(item["expected_ids"])
        packet, qem_ids, tokens = collapse_by_utility(candidates, max_tokens)
        ranked = sorted(candidates.values(), key=lambda candidate: candidate["score"], reverse=True)
        vector = ranked[:len(qem_ids)]
        qem_recalls.append(recall(set(qem_ids), expected))
        vector_recalls.append(recall({candidate["id"] for candidate in vector}, expected))
        qem_tokens.append(tokens)
        vector_tokens.append(sum(max(1, int(len(candidate["content"].split()) * 1.3)) for candidate in vector))
    count = len(qem_recalls)
    if not count:
        raise ValueError("Benchmark input is empty")
    return {
        "cases": count,
        "qem_recall": sum(qem_recalls) / count,
        "vector_recall": sum(vector_recalls) / count,
        "qem_average_tokens": sum(qem_tokens) / count,
        "vector_average_tokens": sum(vector_tokens) / count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-tokens", type=int, default=1200)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset, args.max_tokens), indent=2))
