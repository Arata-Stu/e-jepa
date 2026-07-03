from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.checkpoint import checkpoint


class SIGReg(torch.nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer.

    This follows the Epps-Pulley-style implementation in ``tmp/le_wm``. The
    input layout used here is [B, N, D]. As in the reference implementation,
    the characteristic function is averaged across B independently for each
    token position N. Projection chunking reduces peak memory, while optional
    token subsampling reduces both memory and compute for large video grids.
    """

    def __init__(
        self,
        *,
        knots: int = 17,
        num_proj: int = 1024,
        projection_chunk_size: int = 64,
        max_tokens: int | None = 512,
        t_max: float = 3.0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        if int(knots) < 2:
            raise ValueError("knots must be >= 2")
        if int(num_proj) <= 0:
            raise ValueError("num_proj must be > 0")
        if int(projection_chunk_size) <= 0:
            raise ValueError("projection_chunk_size must be > 0")
        if max_tokens is not None and int(max_tokens) <= 0:
            raise ValueError("max_tokens must be > 0 or null")
        if float(t_max) <= 0.0:
            raise ValueError("t_max must be > 0")

        self.num_proj = int(num_proj)
        self.projection_chunk_size = min(
            int(projection_chunk_size),
            self.num_proj,
        )
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        self.use_checkpoint = bool(use_checkpoint)

        t = torch.linspace(
            0.0,
            float(t_max),
            int(knots),
            dtype=torch.float32,
        )
        dt = float(t_max) / float(int(knots) - 1)
        weights = torch.full((int(knots),), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        gaussian_characteristic_fn = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", gaussian_characteristic_fn)
        self.register_buffer(
            "weights",
            weights * gaussian_characteristic_fn,
        )

    def _subsample_tokens(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.max_tokens is None or embeddings.shape[1] <= self.max_tokens:
            return embeddings
        indices = torch.randperm(
            embeddings.shape[1],
            device=embeddings.device,
        )[: self.max_tokens]
        return embeddings.index_select(1, indices)

    def _projection_statistic(
        self,
        values: torch.Tensor,
        projections: torch.Tensor,
    ) -> torch.Tensor:
        projected = values @ projections
        x_t = projected.unsqueeze(-1) * self.t
        error = (
            x_t.cos().mean(dim=0) - self.phi
        ).square() + x_t.sin().mean(dim=0).square()
        statistic = (error @ self.weights) * int(values.shape[0])
        return statistic.sum()

    def _forward_one(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError(
                "SIGReg embeddings must have shape [B,N,D], "
                f"got {tuple(embeddings.shape)}"
            )
        if embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0:
            raise ValueError("SIGReg requires non-empty batch and token axes")

        embeddings = self._subsample_tokens(embeddings)
        device_type = embeddings.device.type
        statistic_sum = embeddings.new_zeros((), dtype=torch.float32)
        statistic_count = 0

        # The characteristic-function statistic is evaluated in float32 even
        # when the JEPA forward pass uses bf16/fp16 autocast.
        with torch.autocast(device_type=device_type, enabled=False):
            values = embeddings.float()
            embed_dim = int(values.shape[-1])

            for start in range(0, self.num_proj, self.projection_chunk_size):
                chunk_size = min(
                    self.projection_chunk_size,
                    self.num_proj - start,
                )
                projections = torch.randn(
                    embed_dim,
                    chunk_size,
                    device=values.device,
                    dtype=torch.float32,
                )
                projections = projections.div_(
                    projections.norm(p=2, dim=0).clamp_min_(1e-12)
                )

                if (
                    self.use_checkpoint
                    and torch.is_grad_enabled()
                    and values.requires_grad
                ):
                    chunk_statistic = checkpoint(
                        self._projection_statistic,
                        values,
                        projections,
                        use_reentrant=False,
                    )
                else:
                    chunk_statistic = self._projection_statistic(
                        values,
                        projections,
                    )
                statistic_sum = statistic_sum + chunk_statistic
                statistic_count += int(values.shape[1]) * chunk_size

        return statistic_sum / max(statistic_count, 1)

    def forward(
        self,
        embeddings: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if torch.is_tensor(embeddings):
            return self._forward_one(embeddings)
        values = list(embeddings)
        if len(values) == 0:
            raise ValueError("SIGReg embeddings sequence must be non-empty")
        return torch.stack([self._forward_one(value) for value in values]).mean()
