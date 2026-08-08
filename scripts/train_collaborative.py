"""CLI wrapper for stage-2 collaborative alignment."""

import fire

from minionerec.training.collaborative import train


if __name__ == "__main__":
    fire.Fire(train)
