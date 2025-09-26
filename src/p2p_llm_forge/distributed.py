"""Process group lifecycle management for PyTorch distributed training."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from .config import TrainingConfig


class DistributedSetup:
    """Handles initialization and cleanup of the distributed process group."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._initialized = False

    def setup_process_group(self) -> None:
        """Initializes torch.distributed according to the provided config."""

        self._export_environment()
        if dist.is_initialized():
            self._initialized = True
            return
        # Skip distributed initialization for single-process training
        if self.config.world_size == 1:
            self._initialized = True
            return
        dist.init_process_group(
            backend=self.config.backend,
            rank=self.config.rank,
            world_size=self.config.world_size,
        )
        self._select_device()
        self._initialized = True

    def cleanup(self) -> None:
        """Destroys the process group when distributed training is finished."""

        if dist.is_initialized() and self.config.world_size > 1:
            dist.destroy_process_group()
        self._initialized = False

    def is_initialized(self) -> bool:
        """Reports whether the process group is active."""

        return self._initialized

    def global_barrier(self) -> None:
        """Synchronizes all ranks at a safe checkpoint."""

        if dist.is_initialized() and self.config.world_size > 1:
            dist.barrier()

    def _export_environment(self) -> None:
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = str(self.config.master_port)
        os.environ["RANK"] = str(self.config.rank)
        os.environ["WORLD_SIZE"] = str(self.config.world_size)
        os.environ.setdefault("LOCAL_RANK", str(self.config.local_rank))

    def _select_device(self) -> None:
        if not torch.cuda.is_available():
            return
        if self.config.backend != "nccl":
            return
        torch.cuda.set_device(self.config.local_rank)


def get_rank(default: int = 0) -> int:
    """Retrieves the current rank when available."""

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return default


def get_world_size(default: int = 1) -> int:
    """Retrieves the number of participating ranks when available."""

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return default
