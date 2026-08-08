# MiniOneRec-Bridge

**Semantic ID and behavioral-signal fusion for generative recommendation.**

MiniOneRec-Bridge is a research extension built on
[MiniOneRec](https://github.com/AkaliKong/MiniOneRec). It keeps MiniOneRec's
Semantic ID construction and supervised generative recommendation pipeline,
then adds a causal behavior encoder that injects collaborative user signals
into the language-model embedding space.

This repository uses a supervised-only training path. The complete path is
data preparation, SID construction, SFT, behavioral encoder
pretraining, collaborative alignment, constrained generation, and offline
evaluation.

## What Changed

| Area | MiniOneRec baseline | MiniOneRec-Bridge |
|---|---|---|
| Item representation | Text-derived Semantic IDs | Same SID pipeline |
| User representation | SID sequence in the prompt | SID sequence plus causal behavior embedding |
| Behavioral query | Not applicable | Last observed item, never the prediction target |
| Cross-space fusion | Not applicable | MLP projector into the LLM token space |
| Training | Multi-stage generative tuning | SFT followed by isolated collaborative alignment |
| Evaluation | Overall HR/NDCG | Overall and history-length bucket metrics |

The key adaptation is causal availability. Candidate-aware DIN normally uses a
target item as its attention query, but the next target is unknown during
open-ended SID generation. MiniOneRec-Bridge therefore queries the behavior
history with the last observed item, preventing target leakage between offline
training and online inference.

## Pipeline

```text
Amazon interactions and item text
        |
        v
Text embedding -> RQ-VAE / RQ-Kmeans+ -> Semantic IDs
        |                                  |
        |                                  v
        |                       Multi-task SID SFT
        |                                  |
        v                                  v
Causal DIN pretraining ---------> Collaborative projector alignment
                                           |
                                           v
                              Constrained SID generation
                                           |
                                           v
                              HR / NDCG / bucket analysis
```

The language-model backbone and pretrained DIN are frozen by default during
the alignment stage. Only the lightweight projector is trained, which separates
behavior-model quality from cross-space alignment quality and keeps ablations
easy to interpret.

## Repository Layout

```text
minionerec/
|-- data/          # SFT/evaluation datasets and collaborative data adapter
|-- models/        # Causal DIN, projector, collaborative CausalLM wrapper
|-- training/      # SFT, DIN pretraining, collaborative alignment
`-- evaluation/    # Constrained decoding, generation, ranking metrics
scripts/           # Stable command-line wrappers
rq/                # Semantic ID construction
data/              # Amazon preprocessing scripts and tracked sample data
tests/              # Collaborative module regression tests
docs/               # Design rationale, experiment matrix, interview story
```

Root-level `sft.py`, `evaluate.py`, `calc.py`, `data.py`, and
`LogitProcessor.py` are compatibility wrappers for the original command style.
New Python code should import from `minionerec.*`.

## Installation

```bash
conda create -n minionerec-bridge python=3.11 -y
conda activate minionerec-bridge
pip install -r requirements.txt
```

GPU requirements depend on the selected Qwen backbone. Model checkpoints and
raw datasets are local artifacts and are excluded from Git.

## End-to-End Usage

### 1. Prepare Amazon data

```bash
bash data/amazon18_data_process.sh \
  --dataset Office_Products \
  --user_k 5 \
  --item_k 5 \
  --output_path ./data
```

### 2. Encode item text and construct SIDs

```bash
bash rq/text2emb/amazon_text2emb.sh
bash rq/rqvae.sh
python rq/generate_indices.py
```

RQ-Kmeans+ and constrained RQ-Kmeans remain available under `rq/` as
alternative SID constructors.

### 3. Convert data and run SFT

```bash
python convert_dataset.py \
  --dataset_name Office_Products \
  --data_dir ./data/Office_Products \
  --output_dir ./OneRec_data/Office_Products

bash sft.sh
```

### 4. Pretrain the causal behavior encoder

```bash
python -m scripts.train_din \
  --train_file ./data/Amazon/train/Office_Products_5_2016-10-2018-11.csv \
  --valid_file ./data/Amazon/valid/Office_Products_5_2016-10-2018-11.csv \
  --output_path ./outputs/office/din.pt
```

### 5. Align behavioral signals with the SFT model

```bash
python -m scripts.train_collaborative \
  --base_model ./SFT_Model/final_checkpoint \
  --din_checkpoint ./outputs/office/din.pt \
  --train_file ./data/Amazon/train/Office_Products_5_2016-10-2018-11.csv \
  --eval_file ./data/Amazon/valid/Office_Products_5_2016-10-2018-11.csv \
  --output_dir ./outputs/office/collaborative_adapter
```

### 6. Generate and evaluate

```bash
python evaluate.py \
  --base_model ./SFT_Model/final_checkpoint \
  --collaborative_adapter ./outputs/office/collaborative_adapter \
  --info_file ./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt \
  --test_data_path ./data/Amazon/test/Office_Products_5_2016-10-2018-11.csv \
  --result_json_data ./outputs/office/predictions.json

python calc.py \
  --path ./outputs/office/predictions.json \
  --item_path ./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt

python -m minionerec.evaluation.collaborative_metrics \
  --result_paths bridge=./outputs/office/predictions.json \
  --item_info_file ./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt
```

## Evaluation Protocol

Use identical data splits, SID codebooks, SFT checkpoints, beam sizes, and
random seeds for every comparison. At minimum, compare:

1. SFT-only MiniOneRec baseline.
2. DIN score-level late fusion.
3. Random-vector injection with an equal-size projector.
4. One-stage joint training.
5. Two-stage alignment with a frozen DIN.
6. Two-stage alignment with DIN fine-tuning.

Report overall HR@10/NDCG@10 and buckets for history lengths `1-2`, `3-5`,
`6-10`, and `>10`. Current resume numbers must not be attributed to the
collaborative extension until this controlled experiment is complete.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover causal padding behavior, placeholder replacement, stage-2
freezing, and invalid prompt detection. A tiny Qwen forward and beam-generation
smoke test is also used during development.

## Documentation

- `docs/collaborative_optimization_story.md`: design decisions, ablations, and interview narrative.
- `docs/REPOSITORY_STRUCTURE.md`: module ownership and artifact policy.

## Upstream and License

This work is derived from MiniOneRec by Xiaoyu Kong et al. The original source,
license, framework design, and citation are available at
[AkaliKong/MiniOneRec](https://github.com/AkaliKong/MiniOneRec). This repository
retains the Apache-2.0 `LICENSE` and does not claim ownership of upstream code.

Our changes focus on the supervised semantic-behavior fusion path described above.
