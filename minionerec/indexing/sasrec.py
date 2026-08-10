"""Small SASRec trainer used to produce frozen item CF embeddings for LETTER."""

from __future__ import annotations

import ast
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class SASRecConfig:
    num_items: int
    hidden_dim: int = 32
    max_length: int = 50
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.num_items <= 1:
            raise ValueError("num_items must be greater than one")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")


class SASRecBlock(nn.Module):
    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_dim)
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(config.hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        value: torch.Tensor,
        padding_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(value)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            attn_mask=causal_mask,
            need_weights=False,
        )
        value = value + attended
        value = value + self.feed_forward(self.feed_forward_norm(value))
        return value.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class SASRec(nn.Module):
    """SASRec with zero reserved for sequence padding."""

    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(
            config.num_items + 1, config.hidden_dim, padding_idx=0
        )
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [SASRecBlock(config) for _ in range(config.num_layers)]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        if item_ids.ndim != 2 or item_ids.size(1) > self.config.max_length:
            raise ValueError("item_ids must have shape [B, L] with L <= max_length")
        padding_mask = item_ids.eq(0)
        positions = torch.arange(item_ids.size(1), device=item_ids.device)
        value = self.item_embedding(item_ids) + self.position_embedding(positions)
        value = self.dropout(value)
        causal_mask = torch.triu(
            torch.ones(
                item_ids.size(1), item_ids.size(1), dtype=torch.bool, device=item_ids.device
            ),
            diagonal=1,
        )
        for block in self.blocks:
            value = block(value, padding_mask, causal_mask)
        return self.output_norm(value).masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def score_items(self, histories: torch.Tensor) -> torch.Tensor:
        encoded = self(histories)
        lengths = histories.ne(0).sum(dim=1).clamp_min(1)
        final = encoded[torch.arange(histories.size(0), device=histories.device), lengths - 1]
        return final @ self.item_embedding.weight[1:].T

    def export_item_embeddings(self) -> torch.Tensor:
        return self.item_embedding.weight[1:].detach()


def _load_item_mapping(item_file: str) -> Tuple[List[str], Dict[int, int]]:
    with open(item_file, "r", encoding="utf-8") as stream:
        items = json.load(stream)
    try:
        item_ids = sorted(items, key=lambda value: int(value))
    except ValueError:
        item_ids = sorted(items)
    raw_to_row = {int(item_id): row for row, item_id in enumerate(item_ids)}
    return item_ids, raw_to_row


def _parse_history(value: object) -> List[int]:
    text = str(value).strip()
    if not text:
        return []
    if not text.startswith("["):
        return [int(item) for item in text.split()]
    parsed = ast.literal_eval(text)
    return [int(item) for item in parsed]


def _load_interactions(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, usecols=["history_item_id", "item_id"])
    frame = pd.read_csv(path, sep="\t")
    history_column = next(
        (column for column in frame if column.split(":", 1)[0] in {"item_id_list", "history_item_id"}),
        None,
    )
    target_column = next(
        (column for column in frame if column.split(":", 1)[0] == "item_id"), None
    )
    if history_column is None or target_column is None:
        raise ValueError(f"cannot find history and target columns in {path}")
    return frame[[history_column, target_column]].rename(
        columns={history_column: "history_item_id", target_column: "item_id"}
    )


class SASRecTrainDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        raw_to_row: Dict[int, int],
        max_length: int,
        seed: int,
    ) -> None:
        frame = _load_interactions(csv_path)
        self.sequences: List[List[int]] = []
        self.max_length = max_length
        self.num_items = len(raw_to_row)
        self.seed = seed
        for row in frame.itertuples(index=False):
            raw_sequence = _parse_history(row.history_item_id) + [int(row.item_id)]
            sequence = [raw_to_row[item] + 1 for item in raw_sequence if item in raw_to_row]
            if len(sequence) >= 2:
                self.sequences.append(sequence[-(max_length + 1) :])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        sequence = self.sequences[index]
        inputs, positives = sequence[:-1], sequence[1:]
        interacted = set(sequence)
        rng = random.Random(self.seed + index + random.randint(0, 2**16))
        negatives = []
        for _ in positives:
            if len(interacted) >= self.num_items:
                raise ValueError("negative sampling requires at least one unseen item")
            negative = rng.randint(1, self.num_items)
            while negative in interacted:
                negative = rng.randint(1, self.num_items)
            negatives.append(negative)
        pad = self.max_length - len(inputs)
        return (
            torch.tensor(inputs + [0] * pad, dtype=torch.long),
            torch.tensor(positives + [0] * pad, dtype=torch.long),
            torch.tensor(negatives + [0] * pad, dtype=torch.long),
        )


class SASRecEvalDataset(Dataset):
    def __init__(
        self, csv_path: str, raw_to_row: Dict[int, int], max_length: int
    ) -> None:
        frame = _load_interactions(csv_path)
        self.samples = []
        for row in frame.itertuples(index=False):
            history = [
                raw_to_row[item] + 1
                for item in _parse_history(row.history_item_id)
                if item in raw_to_row
            ][-max_length:]
            target = raw_to_row.get(int(row.item_id))
            if history and target is not None:
                self.samples.append((history + [0] * (max_length - len(history)), target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        history, target = self.samples[index]
        return torch.tensor(history), torch.tensor(target)


@torch.no_grad()
def _evaluate(model: SASRec, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    hits = total = 0
    for histories, targets in loader:
        histories, targets = histories.to(device), targets.to(device)
        scores = model.score_items(histories)
        topk = scores.topk(min(10, scores.size(1)), dim=1).indices
        hits += topk.eq(targets[:, None]).any(dim=1).sum().item()
        total += targets.numel()
    return hits / max(total, 1)


def train_sasrec_embeddings(
    train_file: str,
    valid_file: str,
    item_file: str,
    output_path: str,
    checkpoint_path: str = "",
    hidden_dim: int = 32,
    max_length: int = 50,
    num_layers: int = 2,
    num_heads: int = 2,
    dropout: float = 0.2,
    batch_size: int = 256,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    seed: int = 42,
    device: str = "",
) -> None:
    """Train SASRec on train interactions and export row-aligned item embeddings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    item_ids, raw_to_row = _load_item_mapping(item_file)
    config = SASRecConfig(
        num_items=len(item_ids),
        hidden_dim=hidden_dim,
        max_length=max_length,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )
    train_data = SASRecTrainDataset(train_file, raw_to_row, max_length, seed)
    valid_data = SASRecEvalDataset(valid_file, raw_to_row, max_length)
    if not train_data:
        raise ValueError("no valid SASRec training sequences were found")
    train_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = SASRec(config).to(train_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    criterion = nn.BCEWithLogitsLoss()
    best_metric, stale, best_state = -1.0, 0, None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, positives, negatives in train_loader:
            inputs = inputs.to(train_device)
            positives = positives.to(train_device)
            negatives = negatives.to(train_device)
            encoded = model(inputs)
            positive_logits = (encoded * model.item_embedding(positives)).sum(dim=-1)
            negative_logits = (encoded * model.item_embedding(negatives)).sum(dim=-1)
            mask = positives.ne(0)
            loss = criterion(positive_logits[mask], torch.ones_like(positive_logits[mask]))
            loss += criterion(negative_logits[mask], torch.zeros_like(negative_logits[mask]))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
        metric = (
            _evaluate(model, valid_loader, train_device)
            if valid_data
            else -running_loss
        )
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "loss": running_loss / max(len(train_loader), 1),
                    "valid_hr@10": metric,
                }
            )
        )
        if metric > best_metric:
            best_metric, stale = metric, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("SASRec training did not produce a checkpoint")
    model.load_state_dict(best_state)
    embeddings = model.export_item_embeddings().cpu().numpy().astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, embeddings)
    mapping_path = os.path.splitext(output_path)[0] + ".items.json"
    with open(mapping_path, "w", encoding="utf-8") as stream:
        json.dump(item_ids, stream, ensure_ascii=True, indent=2)
    checkpoint_path = checkpoint_path or os.path.splitext(output_path)[0] + ".pt"
    torch.save(
        {
            "config": asdict(config),
            "model": best_state,
            "item_ids": item_ids,
            "valid_hr@10": best_metric,
        },
        checkpoint_path,
    )
    print(f"Saved CF embeddings {embeddings.shape} to {output_path}")
