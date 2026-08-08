"""Causal collaborative-signal integration for MiniOneRec."""

from .data import (
    CollaborativeDataCollator,
    CollaborativeEvalSidDataset,
    CollaborativeSidSFTDataset,
)
from .model import (
    COLLAB_TOKEN,
    CausalDINEncoder,
    CollaborativeCausalLM,
    CollaborativeProjector,
    DINConfig,
    load_collaborative_adapter,
    load_din_checkpoint,
)

__all__ = [
    "COLLAB_TOKEN",
    "CausalDINEncoder",
    "CollaborativeCausalLM",
    "CollaborativeDataCollator",
    "CollaborativeEvalSidDataset",
    "CollaborativeProjector",
    "CollaborativeSidSFTDataset",
    "DINConfig",
    "load_collaborative_adapter",
    "load_din_checkpoint",
]
