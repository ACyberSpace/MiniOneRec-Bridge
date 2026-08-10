"""Generate MiniOneRec-compatible item indices from a SIDAlign checkpoint."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rq.models import RQVAE


PREFIXES = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]


def _load_item_ids(item_file: str, expected: int) -> List[str]:
    if not item_file:
        return [str(index) for index in range(expected)]
    with open(item_file, "r", encoding="utf-8") as stream:
        items = json.load(stream)
    try:
        item_ids = sorted(items, key=lambda value: int(value))
    except ValueError:
        item_ids = sorted(items)
    if len(item_ids) != expected:
        raise ValueError(f"item metadata count {len(item_ids)} != embedding count {expected}")
    return item_ids


def generate(
    checkpoint_path: str,
    output_path: str,
    item_file: str = "",
    batch_size: int = 256,
    device: str = "",
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    embeddings = np.load(args.data_path).astype(np.float32)
    model = RQVAE(
        in_dim=embeddings.shape[1],
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
    model.load_state_dict(checkpoint["state_dict"])
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(selected_device).eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(embeddings)), batch_size=batch_size)
    raw_codes = []
    with torch.no_grad():
        for (batch,) in loader:
            raw_codes.extend(model.get_indices(batch.to(selected_device)).cpu().tolist())
    for quantizer in model.rq.vq_layers[:-1]:
        quantizer.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003
    embedding_tensor = torch.from_numpy(embeddings)
    for _ in range(20):
        groups = defaultdict(list)
        for item_index, codes in enumerate(raw_codes):
            groups[tuple(codes)].append(item_index)
        collisions = [items for items in groups.values() if len(items) > 1]
        if not collisions:
            break
        with torch.no_grad():
            for items in collisions:
                reassigned = model.get_indices(
                    embedding_tensor[items].to(selected_device), use_sk=True
                ).cpu().tolist()
                for item_index, codes in zip(items, reassigned):
                    raw_codes[item_index] = codes

    if len(args.num_emb_list) > len(PREFIXES):
        raise ValueError("no token prefixes are defined for this many codebook levels")
    item_ids = _load_item_ids(item_file, len(raw_codes))
    indices = {
        item_id: [PREFIXES[level].format(int(code)) for level, code in enumerate(codes)]
        for item_id, codes in zip(item_ids, raw_codes)
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(indices, stream, ensure_ascii=True)
    counts = defaultdict(int)
    for codes in raw_codes:
        counts[tuple(codes)] += 1
    collision_rate = (len(raw_codes) - len(counts)) / max(len(raw_codes), 1)
    print(
        json.dumps(
            {
                "items": len(raw_codes),
                "unique_ids": len(counts),
                "collision_rate": collision_rate,
                "output_path": output_path,
            }
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire(generate)
