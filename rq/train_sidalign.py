"""Train the SIDAlign tokenizer for MiniOneRec Semantic IDs."""

import fire

from rq.sidalign_trainer import train_sidalign_tokenizer


if __name__ == "__main__":
    fire.Fire(train_sidalign_tokenizer)
