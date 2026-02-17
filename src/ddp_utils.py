"""DDP (Distributed Data Parallel) helper utilities."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _is_ddp() -> bool:
    """Return True if running under torchrun / distributed launch."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _ddp_rank() -> int:
    return int(os.environ.get("RANK", 0))


def _ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def _is_main_process() -> bool:
    return _ddp_rank() == 0


def setup_ddp():
    """Initialize the distributed process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(_ddp_local_rank())


def cleanup_ddp():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _unwrap_model(model):
    """Return the underlying model, unwrapping DDP if needed."""
    return model.module if isinstance(model, DDP) else model
