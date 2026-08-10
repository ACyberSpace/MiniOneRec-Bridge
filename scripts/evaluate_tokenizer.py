"""CLI wrapper for LETTER Semantic-ID structural diagnostics."""

import fire

from minionerec.indexing import evaluate_tokenizer


if __name__ == "__main__":
    fire.Fire(evaluate_tokenizer)
