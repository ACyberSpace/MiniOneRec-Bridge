"""Backward-compatible SFT command entry point."""

import fire

from minionerec.training.sft import train


if __name__ == "__main__":
    fire.Fire(train)
