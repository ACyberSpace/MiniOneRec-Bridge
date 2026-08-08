"""Pretrain the causal DIN encoder on next-item prediction."""

from __future__ import annotations

import ast
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from minionerec.models import CausalDINEncoder, DINConfig


class NextItemDataset(Dataset):
    def __init__(self, csv_path: str, max_history_len: int, padding_idx: int) -> None:
        frame = pd.read_csv(csv_path, usecols=["history_item_id", "item_id"])
        self.samples = []
        for row in frame.itertuples(index=False):
            history = ast.literal_eval(row.history_item_id)
            history = [int(item) for item in history[-max_history_len:]]
            if not history:
                continue
            mask = [1] * len(history)
            pad = max_history_len - len(history)
            self.samples.append(
                (
                    history + [padding_idx] * pad,
                    mask + [0] * pad,
                    int(row.item_id),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        history, mask, target = self.samples[index]
        return (
            torch.tensor(history, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(target, dtype=torch.long),
        )


def _infer_num_items(*paths: str) -> int:
    maximum = -1
    for path in paths:
        if not path:
            continue
        frame = pd.read_csv(path, usecols=["history_item_id", "item_id"])
        maximum = max(maximum, int(frame["item_id"].max()))
        for value in frame["history_item_id"]:
            history = ast.literal_eval(value)
            if history:
                maximum = max(maximum, max(map(int, history)))
    return maximum + 1


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    hits = total = 0
    for history, mask, target in loader:
        history, mask, target = history.to(device), mask.to(device), target.to(device)
        scores = model.score_items(model(history, mask))
        hits += scores.topk(min(10, scores.size(1)), dim=1).indices.eq(target[:, None]).any(dim=1).sum().item()
        total += target.numel()
    return hits / max(total, 1)


def train(
    train_file: str,
    valid_file: str,
    output_path: str = "./output/din.pt",
    num_items: int = 0,
    embedding_dim: int = 64,
    attention_hidden_dim: int = 128,
    max_history_len: int = 50,
    batch_size: int = 512,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 3,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if num_items <= 0:
        num_items = _infer_num_items(train_file, valid_file)

    config = DINConfig(
        num_items=num_items,
        embedding_dim=embedding_dim,
        attention_hidden_dim=attention_hidden_dim,
        output_dim=embedding_dim,
    )
    model = CausalDINEncoder(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(
        NextItemDataset(train_file, max_history_len, config.padding_idx),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        NextItemDataset(valid_file, max_history_len, config.padding_idx),
        batch_size=batch_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_metric, stale, best_state = -1.0, 0, None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for history, mask, target in train_loader:
            history, mask, target = history.to(device), mask.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model.score_items(model(history, mask)), target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        metric = evaluate(model, valid_loader, device)
        print(json.dumps({"epoch": epoch + 1, "loss": running_loss / max(len(train_loader), 1), "hr@10": metric}))
        if metric > best_metric:
            best_metric, stale = metric, 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save({"config": config.__dict__, "model": best_state, "valid_hr@10": best_metric}, output_path)
    print(f"Saved DIN checkpoint to {output_path}; best validation HR@10={best_metric:.6f}")


if __name__ == "__main__":
    import fire

    fire.Fire(train)
