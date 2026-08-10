"""Collaborative-aware SID tokenizer built on MiniOneRec's RQ-VAE."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch.nn import functional as F

from .rqvae import RQVAE


class SIDAlignRQVAE(RQVAE):
    """RQ-VAE with collaborative alignment and code diversity regularization."""

    def forward_sidalign(
        self,
        content_embeddings: torch.Tensor,
        cf_embeddings: torch.Tensor,
        cluster_labels: Sequence[torch.Tensor],
        collaborative_weight: float,
        diversity_weight: float,
        use_sk: bool = True,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(content_embeddings)
        quantized, quantization_loss, indices = self.rq(encoded, use_sk=use_sk)
        reconstructed = self.decoder(quantized)
        semantic_loss, reconstruction_loss = self.compute_loss(
            reconstructed,
            quantization_loss,
            xs=content_embeddings,
        )
        collaborative_loss = self.collaborative_loss(quantized, cf_embeddings)
        diversity_loss = self.diversity_loss(indices, cluster_labels)
        total_loss = (
            semantic_loss
            + collaborative_weight * collaborative_loss
            + diversity_weight * diversity_loss
        )
        return {
            "loss": total_loss,
            "semantic_loss": semantic_loss,
            "reconstruction_loss": reconstruction_loss,
            "quantization_loss": quantization_loss,
            "collaborative_loss": collaborative_loss,
            "diversity_loss": diversity_loss,
            "indices": indices,
            "quantized": quantized,
        }

    @staticmethod
    def collaborative_loss(
        quantized_embeddings: torch.Tensor, cf_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if quantized_embeddings.shape != cf_embeddings.shape:
            raise ValueError(
                "SIDAlign requires CF embedding dimension to equal the RQ-VAE latent dimension"
            )
        labels = torch.arange(quantized_embeddings.size(0), device=quantized_embeddings.device)
        similarities = quantized_embeddings @ cf_embeddings.T
        return F.cross_entropy(similarities, labels)

    def diversity_loss(
        self, indices: torch.Tensor, cluster_labels: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        if len(cluster_labels) != len(self.rq.vq_layers):
            raise ValueError("one set of diversity cluster labels is required per codebook")
        losses = []
        for level, (quantizer, labels) in enumerate(
            zip(self.rq.vq_layers, cluster_labels)
        ):
            codebook = quantizer.embedding.weight
            labels = labels.to(codebook.device)
            assigned_ids = indices[..., level].reshape(-1)
            positive_for_code = self._sample_cluster_positives(labels)
            valid = positive_for_code[assigned_ids].ge(0)
            if not valid.any():
                continue
            assigned_ids = assigned_ids[valid]
            positives = positive_for_code[assigned_ids]
            selected_codes = codebook[assigned_ids]
            logits = selected_codes @ codebook.T
            logits.scatter_(1, assigned_ids[:, None], torch.finfo(logits.dtype).min)
            losses.append(F.cross_entropy(logits, positives))
        if not losses:
            return indices.new_zeros((), dtype=torch.float32)
        return torch.stack(losses).mean()

    @staticmethod
    def _sample_cluster_positives(labels: torch.Tensor) -> torch.Tensor:
        positives = torch.full_like(labels, -1)
        for cluster_id in labels.unique():
            members = torch.where(labels.eq(cluster_id))[0]
            if members.numel() > 1:
                shift = int(torch.randint(1, members.numel(), ()).item())
                positives[members] = members.roll(shift)
        return positives
