"""Report overall and history-length bucket metrics for generated results."""

from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List, Tuple

import fire


BUCKETS: List[Tuple[str, int, int]] = [
    ("1-2", 1, 2),
    ("3-5", 3, 5),
    ("6-10", 6, 10),
    (">10", 11, 10**9),
]


def _target(sample: Dict) -> str:
    value = sample["output"]
    if isinstance(value, list):
        value = value[0]
    return value.strip(" \n\"")


def _metrics(samples: Iterable[Dict], topk: int, valid_items=None) -> Dict[str, float]:
    samples = list(samples)
    hits = ndcg = invalid = 0.0
    predictions = 0
    for sample in samples:
        target = _target(sample)
        candidates = [value.strip(" \n\"") for value in sample["predict"]]
        predictions += len(candidates)
        invalid += sum(
            (value not in valid_items) if valid_items is not None else (not value)
            for value in candidates
        )
        try:
            rank = candidates.index(target)
        except ValueError:
            rank = len(candidates)
        if rank < topk:
            hits += 1
            ndcg += 1 / math.log2(rank + 2)
    count = len(samples)
    return {
        "samples": count,
        f"HR@{topk}": hits / max(count, 1),
        f"NDCG@{topk}": ndcg / max(count, 1),
        "invalid_generation_rate": invalid / max(predictions, 1),
    }


def analyze(result_paths: str, topk: int = 10, item_info_file: str = ""):
    """Analyze comma-separated ``name=path`` entries or plain paths."""
    valid_items = None
    if item_info_file:
        with open(item_info_file, "r", encoding="utf-8") as stream:
            valid_items = {line.split("\t", 1)[0].strip() for line in stream if line.strip()}
    report = {}
    for entry in result_paths.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, path = entry.split("=", 1) if "=" in entry else (entry, entry)
        with open(path, "r", encoding="utf-8") as stream:
            samples = json.load(stream)
        if any("history_length" not in sample for sample in samples):
            raise ValueError(
                f"{path} has no history_length; regenerate it with the updated evaluate.py"
            )
        result = {"overall": _metrics(samples, topk, valid_items)}
        for label, lower, upper in BUCKETS:
            bucket = [
                sample
                for sample in samples
                if lower <= int(sample["history_length"]) <= upper
            ]
            result[label] = _metrics(bucket, topk, valid_items)
        report[name] = result
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    fire.Fire(analyze)
