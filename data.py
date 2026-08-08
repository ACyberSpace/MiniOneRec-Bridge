"""Backward-compatible dataset imports.

New code should import from :mod:`minionerec.data`.
"""

from minionerec.data.datasets import *  # noqa: F401,F403
