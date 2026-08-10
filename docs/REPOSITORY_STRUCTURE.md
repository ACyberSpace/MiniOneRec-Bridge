# Repository Structure

MiniOneRec-SIDAlign changes item tokenization and keeps downstream MiniOneRec
supervised training intact.

```text
MiniOneRec-SIDAlign/
|-- minionerec/
|   |-- data/
|   |   `-- datasets.py              # SFT and evaluation datasets
|   |-- indexing/
|   |   |-- sasrec.py                # Frozen item CF embeddings for SIDAlign
|   |   `-- diagnostics.py           # SID code usage and CF-prefix analysis
|   |-- training/
|   |   `-- sft.py                   # Multi-task SID supervised tuning
|   `-- evaluation/
|       |-- constrained_decoding.py  # Valid-SID beam constraints
|       |-- generate.py              # Offline generation
|       `-- ranking_metrics.py       # HR/NDCG
|-- rq/
|   |-- models/
|   |   |-- rqvae.py                 # Content RQ-VAE baseline
|   |   `-- sidalign_rqvae.py        # CF and diversity regularization
|   |-- sidalign_trainer.py          # Constrained code clustering and training
|   |-- train_sidalign.py            # Tokenizer CLI
|   `-- generate_sidalign_indices.py # MiniOneRec index export
|-- scripts/train_sasrec.py          # CF embedding CLI
|-- tests/                           # Focused regression tests
`-- docs/                            # Design and interview narrative
```

## Supported Pipeline

```text
data preparation
  -> content embedding extraction
  -> SASRec item embedding pretraining
  -> SIDAlign Semantic ID construction
  -> unchanged MiniOneRec SFT
  -> constrained SID generation
  -> ranking evaluation
```

Collaborative embeddings are tokenizer supervision only. They are not injected
into the LLM and are not required at inference time.

## Artifact Policy

Model weights, raw datasets, checkpoints, generated arrays, logs, virtual
environments, and IDE metadata stay local and are excluded by `.gitignore`.
Source, compact configuration, tests, and documentation belong in Git.
