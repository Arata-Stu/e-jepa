from __future__ import annotations

from typing import Mapping

import torch


class EventRepresentation:
    def convert(self, events: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class EventVoxelGrid(EventRepresentation):
    def __init__(self, input_size: tuple[int, int, int], normalize: bool):
        if len(input_size) != 3:
            raise ValueError("input_size must be a tuple of (t_bins, height, width)")

        self.voxel_grid = torch.zeros(input_size, dtype=torch.float32, requires_grad=False)
        self.nb_channels = int(input_size[0])
        self.normalize = bool(normalize)

    def convert(self, events: Mapping[str, torch.Tensor]) -> torch.Tensor:
        required_keys = ("x", "y", "p", "t")
        missing = [k for k in required_keys if k not in events]
        if missing:
            raise KeyError(f"missing event keys: {missing}")

        _, height, width = self.voxel_grid.shape
        if events["t"].numel() == 0:
            return self.voxel_grid.clone()

        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events["p"].device)
            voxel_grid = self.voxel_grid.clone()

            x = events["x"].float()
            y = events["y"].float()
            p = events["p"].float()
            t = events["t"].float()

            if t.numel() <= 1:
                t_norm = torch.zeros_like(t)
            else:
                duration = t[-1] - t[0]
                if torch.abs(duration).item() < 1e-12:
                    t_norm = torch.zeros_like(t)
                else:
                    t_norm = (self.nb_channels - 1) * (t - t[0]) / duration

            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()
            value = 2 * p - 1

            for xlim in (x0, x0 + 1):
                for ylim in (y0, y0 + 1):
                    for tlim in (t0, t0 + 1):
                        mask = (
                            (xlim < width)
                            & (xlim >= 0)
                            & (ylim < height)
                            & (ylim >= 0)
                            & (tlim >= 0)
                            & (tlim < self.nb_channels)
                        )

                        interp_weights = (
                            value
                            * (1 - (xlim - x).abs())
                            * (1 - (ylim - y).abs())
                            * (1 - (tlim - t_norm).abs())
                        )

                        index = height * width * tlim.long() + width * ylim.long() + xlim.long()
                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].numel() > 0:
                    values = voxel_grid[mask]
                    mean = values.mean()
                    std = values.std(unbiased=False)
                    if std > 0:
                        voxel_grid[mask] = (values - mean) / std
                    else:
                        voxel_grid[mask] = values - mean

        return voxel_grid


# Backward-compatible alias.
VoxelGrid = EventVoxelGrid

