"""Backward-compatible constrained generation command entry point."""

import fire

from minionerec.evaluation.generate import main


if __name__ == "__main__":
    fire.Fire(main)
