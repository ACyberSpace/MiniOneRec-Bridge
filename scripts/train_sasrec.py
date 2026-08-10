"""CLI wrapper for the SASRec item-embedding stage used by LETTER."""

import fire

from minionerec.indexing import train_sasrec_embeddings


if __name__ == "__main__":
    fire.Fire(train_sasrec_embeddings)
