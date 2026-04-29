# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import torch
import torch.distributed as dist

from src.utils.logging import get_logger

logger = get_logger()


def init_distributed(port=37129, rank_and_world_size=(None, None)):
    # Try to set all environment variables before init_process_group.
    if "SLURM_JOB_ID" in os.environ:
        tmpdir = Path(f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}")
        if tmpdir.exists():
            os.environ["TMPDIR"] = str(tmpdir)

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size

    if rank is None:
        rank_env = os.environ.get("RANK")
        if rank_env is not None:
            rank = int(rank_env)
    if world_size is None:
        world_env = os.environ.get("WORLD_SIZE")
        if world_env is not None:
            world_size = int(world_env)

    if rank is None or world_size is None:
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            os.environ.setdefault("MASTER_ADDR", os.environ.get("HOSTNAME", "localhost"))
            os.environ.setdefault("MASTER_PORT", str(port))
        except Exception:
            logger.info("Distributed env vars not set, falling back to single-process mode.")
            world_size, rank = 1, 0
            return world_size, rank

    if world_size <= 1:
        return 1, 0

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(port))
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    try:
        torch.distributed.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f"Rank: {rank}. Distributed training not available ({e})")

    return world_size, rank


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
