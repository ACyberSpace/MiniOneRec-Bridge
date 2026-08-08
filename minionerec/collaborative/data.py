"""Datasets and collation for collaborative MiniOneRec training."""

from __future__ import annotations

import ast
from typing import Any, Dict, List

import torch

from .model import COLLAB_TOKEN
from data import EvalSidDataset, SidSFTDataset


def _causal_history(row: Any, max_history_len: int, padding_idx: int) -> Dict[str, List[int]]:
    values = row["history_item_id"]
    history = ast.literal_eval(values) if isinstance(values, str) else list(values)
    history = [int(item) for item in history[-max_history_len:]]
    if not history:
        raise ValueError("collaborative training requires non-empty histories")
    if min(history) < 0 or max(history) >= padding_idx:
        raise ValueError(
            f"history item IDs must be in [0, {padding_idx - 1}]; got range "
            f"[{min(history)}, {max(history)}]"
        )
    mask = [1] * len(history)
    pad_length = max_history_len - len(history)
    return {
        "history_item_ids": history + [padding_idx] * pad_length,
        "history_mask": mask + [0] * pad_length,
    }


class _CollaborativePromptMixin:
    max_history_len: int
    padding_idx: int

    def get_history(self, row: Any) -> Dict[str, Any]:
        result = super().get_history(row)
        raw_history = row["history_item_id"]
        parsed_history = ast.literal_eval(raw_history) if isinstance(raw_history, str) else raw_history
        result["history_length"] = len(parsed_history)
        result["input"] += (
            f" The user's collaborative preference is represented by {COLLAB_TOKEN}."
        )
        return result

    def pre(self, idx: int) -> Dict[str, Any]:
        result = super().pre(idx)
        result.update(
            _causal_history(self.data.iloc[idx], self.max_history_len, self.padding_idx)
        )
        return result


class CollaborativeSidSFTDataset(_CollaborativePromptMixin, SidSFTDataset):
    def __init__(self, *args: Any, max_history_len: int, padding_idx: int, **kwargs: Any):
        self.max_history_len = max_history_len
        self.padding_idx = padding_idx
        super().__init__(*args, **kwargs)


class CollaborativeEvalSidDataset(_CollaborativePromptMixin, EvalSidDataset):
    def __init__(self, *args: Any, max_history_len: int, padding_idx: int, **kwargs: Any):
        self.max_history_len = max_history_len
        self.padding_idx = padding_idx
        super().__init__(*args, **kwargs)


class CollaborativeDataCollator:
    def __init__(self, base_collator: Any) -> None:
        self.base_collator = base_collator

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        histories = [feature.pop("history_item_ids") for feature in features]
        masks = [feature.pop("history_mask") for feature in features]
        batch = self.base_collator(features)
        batch["history_item_ids"] = torch.tensor(histories, dtype=torch.long)
        batch["history_mask"] = torch.tensor(masks, dtype=torch.bool)
        return batch
