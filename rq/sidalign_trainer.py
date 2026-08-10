"""Training utilities for the SIDAlign item tokenizer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset

from .models.sidalign_rqvae import SIDAlignRQVAE


class SIDAlignEmbeddingDataset(Dataset):
    def __init__(self, content_path: str, cf_path: str) -> None:
        self.content = _load_array(content_path)
        self.cf = _load_array(cf_path)
        if self.content.ndim != 2 or self.cf.ndim != 2:
            raise ValueError("content and CF embeddings must both be rank-2 arrays")
        if len(self.content) != len(self.cf):
            raise ValueError(
                f"content/CF item counts differ: {len(self.content)} != {len(self.cf)}"
            )
        if not np.isfinite(self.content).all() or not np.isfinite(self.cf).all():
            raise ValueError("embedding files contain NaN or infinite values")
        self.content = self.content.astype(np.float32, copy=False)
        self.cf = self.cf.astype(np.float32, copy=False)

    @property
    def content_dim(self) -> int:
        return self.content.shape[1]

    @property
    def cf_dim(self) -> int:
        return self.cf.shape[1]

    def __len__(self) -> int:
        return len(self.content)

    def __getitem__(self, index: int):
        return torch.from_numpy(self.content[index]), torch.from_numpy(self.cf[index])


def _load_array(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        return np.load(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        for key in ("item_embeddings", "embeddings", "weight"):
            if key in value:
                value = value[key]
                break
    if not torch.is_tensor(value):
        raise ValueError(f"unsupported embedding checkpoint structure in {path}")
    return value.detach().squeeze().cpu().numpy()


def balanced_codebook_clusters(
    codebook: torch.Tensor,
    num_clusters: int,
    iterations: int = 5,
    seed: int = 42,
) -> torch.Tensor:
    """Cluster code embeddings with equal-size assignment constraints.

    The diversity objective uses constrained K-means over code embeddings. This
    implementation alternates centroid updates with an exact capacity assignment.
    """
    values = codebook.detach().float().cpu().numpy()
    num_codes = len(values)
    if num_clusters < 1 or num_clusters > num_codes:
        raise ValueError("num_clusters must be between 1 and the codebook size")
    initial = KMeans(
        n_clusters=num_clusters,
        n_init=10,
        max_iter=100,
        random_state=seed,
    ).fit(values)
    centers = initial.cluster_centers_
    capacities = np.full(num_clusters, num_codes // num_clusters, dtype=np.int64)
    capacities[: num_codes % num_clusters] += 1
    slots = np.repeat(np.arange(num_clusters), capacities)
    labels = np.zeros(num_codes, dtype=np.int64)

    for _ in range(iterations):
        cost = ((values[:, None, :] - centers[slots][None, :, :]) ** 2).sum(axis=-1)
        rows, columns = linear_sum_assignment(cost)
        labels[rows] = slots[columns]
        updated = np.stack([values[labels == cluster].mean(axis=0) for cluster in range(num_clusters)])
        if np.allclose(updated, centers):
            centers = updated
            break
        centers = updated
    return torch.from_numpy(labels)


@dataclass
class SIDAlignTrainResult:
    checkpoint_path: str
    best_collision_rate: float
    best_loss: float


class SIDAlignTrainer:
    def __init__(
        self,
        model: SIDAlignRQVAE,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
        collaborative_weight: float,
        diversity_weight: float,
        diversity_clusters: int,
        seed: int,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.collaborative_weight = collaborative_weight
        self.diversity_weight = diversity_weight
        self.diversity_clusters = diversity_clusters
        self.seed = seed
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    @torch.no_grad()
    def initialize_codebooks(self, loader: DataLoader) -> None:
        """Initialize every residual codebook from all encoded item content."""
        self.model.eval()
        encoded_batches = []
        for content, _ in loader:
            encoded_batches.append(self.model.encoder(content.to(self.device)).cpu())
        residual = torch.cat(encoded_batches, dim=0).to(self.device)
        for quantizer in self.model.rq.vq_layers:
            if residual.size(0) < quantizer.n_e:
                raise ValueError(
                    "SIDAlign codebook initialization needs at least as many items as codes"
                )
            quantizer.init_emb(residual)
            quantized, _, _ = quantizer(residual, use_sk=False)
            residual = residual - quantized

    def _cluster_labels(self, epoch: int) -> List[torch.Tensor]:
        return [
            balanced_codebook_clusters(
                quantizer.embedding.weight,
                min(self.diversity_clusters, quantizer.n_e),
                seed=self.seed + epoch + level,
            ).to(self.device)
            for level, quantizer in enumerate(self.model.rq.vq_layers)
        ]

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        cluster_labels = self._cluster_labels(epoch)
        totals: Dict[str, float] = {}
        for content, cf in loader:
            content, cf = content.to(self.device), cf.to(self.device)
            output = self.model.forward_sidalign(
                content,
                cf,
                cluster_labels,
                self.collaborative_weight,
                self.diversity_weight,
            )
            self.optimizer.zero_grad()
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            for key in (
                "loss",
                "reconstruction_loss",
                "quantization_loss",
                "collaborative_loss",
                "diversity_loss",
            ):
                totals[key] = totals.get(key, 0.0) + float(output[key].detach())
        return {key: value / max(len(loader), 1) for key, value in totals.items()}

    @torch.no_grad()
    def collision_rate(self, loader: DataLoader) -> float:
        self.model.eval()
        unique, total = set(), 0
        for content, _ in loader:
            indices = self.model.get_indices(content.to(self.device), use_sk=False)
            for code in indices.cpu().tolist():
                unique.add(tuple(code))
                total += 1
        return (total - len(unique)) / max(total, 1)


def train_sidalign_tokenizer(
    content_path: str,
    cf_path: str,
    output_dir: str,
    epochs: int = 20000,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    eval_step: int = 200,
    num_workers: int = 0,
    num_emb_list: Sequence[int] = (256, 256, 256, 256),
    latent_dim: int = 32,
    layers: Sequence[int] = (2048, 1024, 512, 256, 128, 64),
    commitment_weight: float = 0.25,
    collaborative_weight: float = 0.01,
    diversity_weight: float = 0.0001,
    diversity_clusters: int = 10,
    sk_epsilons: Sequence[float] = (0.0, 0.0, 0.0, 0.003),
    sk_iters: int = 50,
    kmeans_iters: int = 100,
    seed: int = 42,
    device: str = "",
) -> SIDAlignTrainResult:
    """Train SIDAlign while leaving downstream MiniOneRec unchanged."""
    import argparse
    import json

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dataset = SIDAlignEmbeddingDataset(content_path, cf_path)
    if dataset.cf_dim != latent_dim:
        raise ValueError(
            f"CF embedding dimension ({dataset.cf_dim}) must equal latent_dim ({latent_dim})"
        )
    num_emb_list = list(map(int, num_emb_list))
    sk_epsilons = list(map(float, sk_epsilons))
    if len(num_emb_list) != len(sk_epsilons):
        raise ValueError("num_emb_list and sk_epsilons must have equal lengths")
    model = SIDAlignRQVAE(
        in_dim=dataset.content_dim,
        num_emb_list=num_emb_list,
        e_dim=latent_dim,
        layers=list(map(int, layers)),
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        beta=commitment_weight,
        kmeans_init=True,
        kmeans_iters=kmeans_iters,
        sk_epsilons=sk_epsilons,
        sk_iters=sk_iters,
    )
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=selected_device.type == "cuda",
    )
    trainer = SIDAlignTrainer(
        model,
        selected_device,
        learning_rate,
        weight_decay,
        collaborative_weight,
        diversity_weight,
        diversity_clusters,
        seed,
    )
    os.makedirs(output_dir, exist_ok=True)
    args = argparse.Namespace(
        data_path=content_path,
        cf_path=cf_path,
        num_emb_list=num_emb_list,
        e_dim=latent_dim,
        layers=list(map(int, layers)),
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        beta=commitment_weight,
        kmeans_init=True,
        kmeans_iters=kmeans_iters,
        sk_epsilons=sk_epsilons,
        sk_iters=sk_iters,
        collaborative_weight=collaborative_weight,
        diversity_weight=diversity_weight,
        diversity_clusters=diversity_clusters,
        num_workers=num_workers,
    )
    best_loss = float("inf")
    best_collision = float("inf")
    best_saved_loss = float("inf")
    checkpoint_path = os.path.join(output_dir, "best_sidalign_model.pth")

    trainer.initialize_codebooks(loader)
    for epoch in range(epochs):
        metrics = trainer.train_epoch(loader, epoch)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
        if (epoch + 1) % eval_step == 0 or epoch + 1 == epochs:
            collision = trainer.collision_rate(loader)
            metrics.update({"epoch": epoch + 1, "collision_rate": collision})
            print(json.dumps(metrics, sort_keys=True))
            should_save = collision < best_collision or (
                np.isclose(collision, best_collision) and metrics["loss"] < best_saved_loss
            )
            if should_save:
                best_collision = min(best_collision, collision)
                best_saved_loss = metrics["loss"]
                torch.save(
                    {
                        "args": args,
                        "epoch": epoch + 1,
                        "best_loss": best_loss,
                        "best_collision_rate": best_collision,
                        "state_dict": model.state_dict(),
                        "optimizer": trainer.optimizer.state_dict(),
                    },
                    checkpoint_path,
                )
    return SIDAlignTrainResult(checkpoint_path, best_collision, best_loss)
