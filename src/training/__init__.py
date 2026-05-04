from __future__ import annotations


def main(*args, **kwargs):
    """
    Backward-compatible entry point.

    Importing scripts.train.train at module import time creates a circular
    dependency because scripts.train.train imports src.training.jepa21_utils.
    Keep this as a lazy import instead.
    """

    from scripts.train.train import main as train_main

    return train_main(*args, **kwargs)


__all__ = ["main"]
