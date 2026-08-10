"""Train a LETTER tokenizer for MiniOneRec Semantic IDs."""

import fire

from rq.letter_trainer import train_letter_tokenizer


if __name__ == "__main__":
    fire.Fire(train_letter_tokenizer)
