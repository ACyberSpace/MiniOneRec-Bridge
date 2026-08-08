"""Backward-compatible ranking metric command entry point."""

import fire

from minionerec.evaluation.ranking_metrics import gao


if __name__ == "__main__":
    fire.Fire(gao)
