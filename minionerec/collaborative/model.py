"""Collaborative-signal modules for generative recommendation.

The implementation adapts CoLLM to next-item generation. Unlike candidate
ranking, the target item is unknown at inference time, so the behavior encoder
uses the last observed item as its query and never consumes the target item.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn


COLLAB_TOKEN = "<|collab_user|>"


@dataclass
class DINConfig:
    num_items: int
    embedding_dim: int = 64
    attention_hidden_dim: int = 128
    output_dim: int = 64
    dropout: float = 0.1
    padding_idx: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_items <= 0:
            raise ValueError("num_items must be positive")
        if self.padding_idx is None:
            self.padding_idx = self.num_items
        if self.padding_idx != self.num_items:
            raise ValueError("padding_idx must equal num_items")


class CausalDINEncoder(nn.Module):
    """DIN-style encoder whose query is the last observed behavior.

    Standard DIN is target-aware. Feeding the next-item target into a
    generative recommender would leak the answer, therefore this causal variant
    derives the query from the final non-padding history item.
    """

    def __init__(self, config: DINConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(
            config.num_items + 1,
            config.embedding_dim,
            padding_idx=config.padding_idx,
        )
        attention_input_dim = config.embedding_dim * 4
        self.attention = nn.Sequential(
            nn.Linear(attention_input_dim, config.attention_hidden_dim),
            nn.PReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.attention_hidden_dim, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.output_dim),
            nn.PReLU(),
            nn.LayerNorm(config.output_dim),
        )

    def forward(
        self, history_item_ids: torch.Tensor, history_mask: torch.Tensor
    ) -> torch.Tensor:
        if history_item_ids.ndim != 2 or history_mask.shape != history_item_ids.shape:
            raise ValueError("history_item_ids and history_mask must have shape [B, L]")
        if not torch.all(history_mask.sum(dim=1) > 0):
            raise ValueError("each sample must contain at least one observed item")

        history_mask = history_mask.bool()
        positions = torch.arange(history_item_ids.size(1), device=history_item_ids.device)
        query_positions = positions.unsqueeze(0).expand_as(history_item_ids)
        query_positions = query_positions.masked_fill(~history_mask, -1).max(dim=1).values
        batch_index = torch.arange(history_item_ids.size(0), device=history_item_ids.device)

        keys = self.item_embedding(history_item_ids)
        query = keys[batch_index, query_positions]
        expanded_query = query.unsqueeze(1).expand_as(keys)
        attention_features = torch.cat(
            [expanded_query, keys, expanded_query - keys, expanded_query * keys], dim=-1
        )
        scores = self.attention(attention_features).squeeze(-1)
        scores = scores.masked_fill(~history_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        interest = torch.sum(weights.unsqueeze(-1) * keys, dim=1)
        return self.output(torch.cat([query, interest], dim=-1))

    def score_items(self, user_embedding: torch.Tensor) -> torch.Tensor:
        """Score all real items for standalone DIN pretraining."""
        if user_embedding.size(-1) != self.config.embedding_dim:
            raise ValueError(
                "score_items requires output_dim == embedding_dim during DIN pretraining"
            )
        return user_embedding @ self.item_embedding.weight[: self.config.num_items].T


class CollaborativeProjector(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int = 256, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class CollaborativeCausalLM(nn.Module):
    """Inject a projected collaborative vector at a placeholder token."""

    def __init__(
        self,
        base_model: nn.Module,
        behavior_encoder: CausalDINEncoder,
        collab_token_id: int,
        projector_hidden_dim: int = 256,
        projector_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.behavior_encoder = behavior_encoder
        self.collab_token_id = collab_token_id
        llm_dim = base_model.get_input_embeddings().embedding_dim
        self.projector = CollaborativeProjector(
            behavior_encoder.config.output_dim,
            llm_dim,
            hidden_dim=projector_hidden_dim,
            dropout=projector_dropout,
        )
        self.config = base_model.config

    def freeze_backbones(self, train_behavior_encoder: bool = False) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        for parameter in self.behavior_encoder.parameters():
            parameter.requires_grad = train_behavior_encoder
        for parameter in self.projector.parameters():
            parameter.requires_grad = True

    def _build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        placeholder_mask = input_ids.eq(self.collab_token_id)
        placeholder_count = placeholder_mask.sum(dim=1)
        if not torch.all(placeholder_count == 1):
            raise ValueError(
                "each collaborative sample must contain exactly one collaborative placeholder"
            )

        token_embeddings = self.base_model.get_input_embeddings()(input_ids)
        collaborative = self.projector(
            self.behavior_encoder(history_item_ids, history_mask)
        ).to(token_embeddings.dtype)
        result = token_embeddings.clone()
        result[placeholder_mask] = collaborative
        return result

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        history_item_ids: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        if history_item_ids is None or history_mask is None:
            raise ValueError("collaborative history tensors are required")
        # Trainer >=4.46 may pass this bookkeeping value to compute_loss.
        kwargs.pop("num_items_in_batch", None)
        inputs_embeds = self._build_inputs_embeds(
            input_ids, history_item_ids, history_mask
        )
        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        history_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        inputs_embeds = self._build_inputs_embeds(
            input_ids, history_item_ids, history_mask
        )
        return self.base_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    def save_collaborative_adapter(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            {
                "behavior_encoder": self.behavior_encoder.state_dict(),
                "projector": self.projector.state_dict(),
            },
            os.path.join(output_dir, "collaborative_adapter.pt"),
        )
        config: Dict[str, Any] = {
            "collab_token_id": self.collab_token_id,
            "din_config": asdict(self.behavior_encoder.config),
            "projector_hidden_dim": self.projector.network[1].out_features,
            "projector_dropout": self.projector.network[3].p,
        }
        with open(
            os.path.join(output_dir, "collaborative_adapter.json"),
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(config, stream, indent=2)


def load_din_checkpoint(path: str, map_location: str = "cpu") -> CausalDINEncoder:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config = DINConfig(**checkpoint["config"])
    model = CausalDINEncoder(config)
    model.load_state_dict(checkpoint["model"])
    return model


def load_collaborative_adapter(
    base_model: nn.Module, adapter_dir: str, map_location: str = "cpu"
) -> CollaborativeCausalLM:
    with open(
        os.path.join(adapter_dir, "collaborative_adapter.json"),
        "r",
        encoding="utf-8",
    ) as stream:
        config = json.load(stream)
    behavior_encoder = CausalDINEncoder(DINConfig(**config["din_config"]))
    model = CollaborativeCausalLM(
        base_model,
        behavior_encoder,
        config["collab_token_id"],
        config["projector_hidden_dim"],
        config["projector_dropout"],
    )
    state = torch.load(
        os.path.join(adapter_dir, "collaborative_adapter.pt"),
        map_location=map_location,
        weights_only=False,
    )
    model.behavior_encoder.load_state_dict(state["behavior_encoder"])
    model.projector.load_state_dict(state["projector"])
    return model
