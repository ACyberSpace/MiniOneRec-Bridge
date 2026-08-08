# Repository Structure

The repository keeps the original MiniOneRec entry points at the root for
upstream compatibility. New extensions should use package modules and explicit
command entry points instead of adding more root-level variants.

```text
MiniOneRec/
|-- minionerec/
|   `-- collaborative/       # Causal DIN, LLM injection, datasets, metrics
|-- scripts/                 # Reproducible command-line experiment entry points
|-- rq/                      # Semantic-ID construction and text embedding code
|-- data/                    # Preprocessing code; local dataset folders are ignored
|-- config/                  # Training configuration
|-- tests/                   # Focused automated tests
|-- docs/                    # Design notes, experiment protocol, interview story
|-- assets/                  # README images
|-- sft.py / rl.py           # Original MiniOneRec training entry points
|-- evaluate.py / calc.py    # Generation and metric entry points
`-- *_gpr.py                 # Existing experimental variants; not the CoLLM path
```

## Collaborative Pipeline

Use module execution from the repository root so imports are deterministic:

```bash
python -m scripts.train_din --help
python -m scripts.train_collaborative --help
python -m minionerec.collaborative.metrics --help
```

The stable import surface is `minionerec.collaborative`. Internal files may be
reorganized without changing callers as long as that package API remains
compatible.

## Artifact Policy

Model weights, raw/processed datasets, checkpoints, generated arrays, logs, and
IDE metadata stay local and are excluded by `.gitignore`. Source code, compact
configuration, tests, and documentation belong in Git.

Large artifacts should be published through a model or dataset registry and
referenced from documentation. They should not be committed to the source
repository.
