# MiniOneRec-SIDAlign

**Collaborative Semantic IDs for supervised generative recommendation.**

MiniOneRec-SIDAlign is a research extension of
[MiniOneRec](https://github.com/AkaliKong/MiniOneRec). It replaces the
content-only item tokenizer with a collaborative-aware tokenizer, while keeping
MiniOneRec's SFT, constrained generation, and ranking evaluation unchanged.

The project intentionally excludes reinforcement learning and runtime
collaborative adapters. Collaborative signals enter only during Semantic ID
construction, so the downstream comparison remains controlled.

## Method

```text
item title + description -> content embedding -> RQ-VAE -> quantized embedding -> SID
                                                     |              ^
                                                     |              |
train interactions -> SASRec -> frozen item CF embedding ------------+
```

The tokenizer jointly optimizes three objectives:

```text
L = L_semantic + alpha * L_collaborative + beta * L_diversity
```

- `L_semantic`: content reconstruction plus residual quantization loss.
- `L_collaborative`: in-batch contrastive alignment between quantized item
  representations and frozen SASRec item embeddings.
- `L_diversity`: constrained-cluster regularization over every codebook to
  reduce biased code assignment.

MiniOneRec SFT is held fixed so that changes can be attributed to item
tokenization rather than downstream training changes.

## Layout

```text
minionerec/
|-- data/          # SFT and evaluation datasets
|-- indexing/      # SASRec CF-embedding training
|-- training/      # Original supervised MiniOneRec training
`-- evaluation/    # Constrained generation and ranking metrics
rq/
|-- models/        # RQ-VAE and SIDAlign objective implementation
|-- train_sidalign.py
`-- generate_sidalign_indices.py
scripts/           # Stable command wrappers
tests/             # Tokenizer and data regression tests
docs/              # Architecture and experiment narrative
```

Root `sft.py`, `evaluate.py`, `calc.py`, `data.py`, and `LogitProcessor.py`
remain compatibility wrappers for the original commands.

## Installation

```bash
conda create -n minionerec-sidalign python=3.11 -y
conda activate minionerec-sidalign
pip install -r requirements.txt
```

Model weights, raw data, generated embeddings, and checkpoints are excluded
from Git.

## Pipeline

The examples below use `Office_Products`; replace paths consistently for other
datasets.

### 1. Prepare interactions

```bash
python data/amazon18_data_process.py \
  --dataset Office_Products \
  --user_k 5 \
  --item_k 5 \
  --output_path ./OneRec_data
```

### 2. Encode item content

```bash
python rq/text2emb/amazon_text2emb.py \
  --dataset Office_Products \
  --root ./OneRec_data \
  --plm_checkpoint ./Qwen2.5-3B-Instruct \
  --output_path ./OneRec_data/Office_Products/Office_Products.content.npy
```

### 3. Train SASRec and export CF embeddings

Only training interactions optimize SASRec. Validation interactions are used
for early stopping and test interactions are never loaded.

```bash
python -m scripts.train_sasrec \
  --train_file ./OneRec_data/Office_Products/Office_Products.train.inter \
  --valid_file ./OneRec_data/Office_Products/Office_Products.valid.inter \
  --item_file ./OneRec_data/Office_Products/Office_Products.item.json \
  --output_path ./OneRec_data/Office_Products/Office_Products.cf-sasrec-32.npy \
  --hidden_dim 32
```

The exporter sorts item keys numerically, matching the content-embedding order.
It also writes an item-order manifest beside the `.npy` file.

### 4. Train the SIDAlign tokenizer

The defaults use four 256-entry codebooks, 32-dimensional quantized/CF
embeddings, `alpha=0.01`, and `beta=0.0001`.

```bash
python -m rq.train_sidalign \
  --content_path ./OneRec_data/Office_Products/Office_Products.content.npy \
  --cf_path ./OneRec_data/Office_Products/Office_Products.cf-sasrec-32.npy \
  --output_dir ./outputs/office/sidalign-tokenizer
```

### 5. Generate MiniOneRec-compatible SIDs

```bash
python -m rq.generate_sidalign_indices \
  --checkpoint_path ./outputs/office/sidalign-tokenizer/best_sidalign_model.pth \
  --item_file ./OneRec_data/Office_Products/Office_Products.item.json \
  --output_path ./OneRec_data/Office_Products/Office_Products.index.json
```

Convert the generated SIDs into MiniOneRec's train/valid/test CSV layout:

```bash
python convert_dataset.py \
  --data_dir ./OneRec_data \
  --dataset_name Office_Products \
  --output_dir ./OneRec_data/Office_Products \
  --category Office_Products
```

Then use `sft.sh`, `evaluate.py`, and `calc.py` exactly as in MiniOneRec. No CF
model is loaded during SFT or inference.

Tokenizer structure can be checked before the expensive SFT run:

```bash
python -m scripts.evaluate_tokenizer \
  --index_path ./OneRec_data/Office_Products/Office_Products.index.json \
  --cf_path ./OneRec_data/Office_Products/Office_Products.cf-sasrec-32.npy \
  --item_manifest ./OneRec_data/Office_Products/Office_Products.cf-sasrec-32.items.json
```

## Controlled Evaluation

Use identical data splits, content embeddings, codebook sizes, SFT settings,
beam sizes, and seeds for:

1. `Content SID`: original RQ-VAE objective.
2. `SIDAlign-CF`: semantic plus collaborative regularization.
3. `SIDAlign-Full`: semantic, collaborative, and diversity regularization.
4. `Shuffled-CF`: item CF embeddings shuffled before SIDAlign training.

## Tests

```bash
python -m unittest discover -s tests -v
```

## References and License

This repository retains MiniOneRec's Apache-2.0 license. The design is informed
by collaborative-aware tokenization research, including
[LETTER](https://arxiv.org/abs/2405.07314); cite the relevant sources in
derived research.
