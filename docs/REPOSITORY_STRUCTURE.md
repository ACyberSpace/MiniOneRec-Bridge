# Repository Structure

MiniOneRec-Bridge keeps a small set of root command wrappers for compatibility,
while implementation code lives under the `minionerec` package.

```text
MiniOneRec-Bridge/
|-- minionerec/
|   |-- data/
|   |   |-- datasets.py              # SFT and evaluation datasets
|   |   `-- collaborative.py         # Behavior-history tensors and collator
|   |-- models/
|   |   `-- collaborative.py         # Causal DIN and LLM-space projector
|   |-- training/
|   |   |-- sft.py                   # Multi-task SID supervised tuning
|   |   |-- din.py                   # Causal behavior encoder pretraining
|   |   `-- collaborative.py         # Stage-2 embedding alignment
|   `-- evaluation/
|       |-- constrained_decoding.py  # Valid-SID beam constraints
|       |-- generate.py              # Offline generation
|       |-- ranking_metrics.py       # HR/NDCG
|       `-- collaborative_metrics.py # History-length bucket analysis
|-- scripts/                         # Thin CLI wrappers
|-- rq/                              # Semantic ID construction
|-- data/                            # Preprocessing scripts and sample data
|-- tests/                           # Focused regression tests
|-- docs/                            # Design and experiment documentation
|-- sft.py / evaluate.py / calc.py   # Backward-compatible commands
`-- data.py / LogitProcessor.py      # Backward-compatible imports
```

## Supported Pipeline

```text
data preparation
  -> text embedding
  -> SID construction
  -> SID supervised fine-tuning
  -> causal DIN pretraining
  -> collaborative embedding alignment
  -> constrained SID generation
  -> overall and bucketed evaluation
```

The supported pipeline uses supervised objectives throughout. Additional
post-training stages are outside this project's scope.

## Import Policy

New code should import stable package modules:

```python
from minionerec.data import SidSFTDataset
from minionerec.models import CollaborativeCausalLM
from minionerec.evaluation import ConstrainedLogitsProcessor
```

Root wrappers exist only so the original shell scripts keep working.

## Artifact Policy

Model weights, raw datasets, checkpoints, generated arrays, logs, virtual
environments, and IDE metadata stay local and are excluded by `.gitignore`.
Source code, compact configuration, tests, and documentation belong in Git.

Large artifacts should be published through a model or dataset registry rather
than committed to the source repository.
