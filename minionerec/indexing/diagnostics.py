"""Structural diagnostics for content-only and LETTER Semantic IDs."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np


def _sorted_item_ids(indices: Dict[str, Sequence[str]]) -> List[str]:
    try:
        return sorted(indices, key=lambda value: int(value))
    except ValueError:
        return sorted(indices)


def _entropy(counts: Sequence[int]) -> float:
    probabilities = np.asarray(counts, dtype=np.float64)
    probabilities /= probabilities.sum()
    return float(-(probabilities * np.log(probabilities + 1e-12)).sum())


def evaluate_tokenizer(
    index_path: str,
    cf_path: str,
    item_manifest: str = "",
    neighbor_k: int = 10,
    sample_size: int = 2000,
    seed: int = 42,
    output_path: str = "",
) -> Dict[str, object]:
    """Measure collisions, code usage, and CF-neighbor SID prefix agreement."""
    with open(index_path, "r", encoding="utf-8") as stream:
        indices = json.load(stream)
    item_ids = _sorted_item_ids(indices)
    if item_manifest:
        with open(item_manifest, "r", encoding="utf-8") as stream:
            item_ids = json.load(stream)
    cf = np.load(cf_path).astype(np.float32)
    if len(item_ids) != len(cf):
        raise ValueError(f"index/CF item counts differ: {len(item_ids)} != {len(cf)}")
    codes = [tuple(indices[item_id]) for item_id in item_ids]
    if not codes or not codes[0]:
        raise ValueError("index file contains no Semantic IDs")
    if len(codes) < 2:
        raise ValueError("at least two items are required for neighbor diagnostics")
    levels = len(codes[0])
    if any(len(code) != levels for code in codes):
        raise ValueError("all Semantic IDs must have the same number of levels")

    level_metrics = []
    for level in range(levels):
        counts = Counter(code[level] for code in codes)
        level_metrics.append(
            {
                "level": level + 1,
                "used_codes": len(counts),
                "entropy": _entropy(list(counts.values())),
                "normalized_entropy": _entropy(list(counts.values()))
                / max(math.log(len(counts)), 1e-12),
                "largest_code_share": max(counts.values()) / len(codes),
            }
        )

    norms = np.linalg.norm(cf, axis=1, keepdims=True)
    normalized = cf / np.clip(norms, 1e-12, None)
    rng = np.random.default_rng(seed)
    queries = rng.choice(len(codes), size=min(sample_size, len(codes)), replace=False)
    prefix_hits = np.zeros(levels, dtype=np.float64)
    random_hits = np.zeros(levels, dtype=np.float64)
    comparisons = 0
    for query in queries:
        similarities = normalized @ normalized[query]
        similarities[query] = -np.inf
        k = min(neighbor_k, len(codes) - 1)
        neighbors = np.argpartition(similarities, -k)[-k:]
        random_neighbors = rng.choice(
            np.delete(np.arange(len(codes)), query), size=k, replace=False
        )
        for level in range(1, levels + 1):
            prefix = codes[query][:level]
            prefix_hits[level - 1] += sum(codes[item][:level] == prefix for item in neighbors)
            random_hits[level - 1] += sum(
                codes[item][:level] == prefix for item in random_neighbors
            )
        comparisons += k

    result: Dict[str, object] = {
        "items": len(codes),
        "levels": levels,
        "collision_rate": (len(codes) - len(set(codes))) / len(codes),
        "level_metrics": level_metrics,
        "cf_neighbor_prefix_rate": {
            str(level + 1): float(prefix_hits[level] / max(comparisons, 1))
            for level in range(levels)
        },
        "random_prefix_rate": {
            str(level + 1): float(random_hits[level] / max(comparisons, 1))
            for level in range(levels)
        },
    }
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    return result
