from __future__ import annotations

import math
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DistributedSampler


class DistributedWeightedSampler(DistributedSampler):
    """
    Weighted distributed sampler for mixed datasets.

    This mirrors the interface of torch DistributedSampler while replacing
    permutation sampling with weighted sampling with replacement.
    """

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if not hasattr(dataset, "sample_weights"):
            raise ValueError("Dataset must expose `sample_weights` for weighted sampling.")
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    @property
    def sample_probabilities(self) -> np.ndarray:
        sample_weights = self.dataset.sample_weights
        if isinstance(sample_weights, torch.Tensor):
            sample_weights = sample_weights.cpu().numpy()
        elif isinstance(sample_weights, list):
            sample_weights = np.array(sample_weights, dtype=np.float64)
        elif not isinstance(sample_weights, np.ndarray):
            raise ValueError(
                "sample_weights must be a numpy array, torch.Tensor, or python list"
            )
        return sample_weights / np.sum(sample_weights)

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = rng.choice(
            np.arange(n),
            size=self.total_size,
            p=self.sample_probabilities,
            replace=True,
        ).tolist()

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[
                        :padding_size
                    ]
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

