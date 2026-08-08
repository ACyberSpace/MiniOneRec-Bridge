"""CLI wrapper for causal DIN pretraining."""

import fire

from minionerec.training.din import train


if __name__ == "__main__":
    fire.Fire(train)
