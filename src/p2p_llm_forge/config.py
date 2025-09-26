"""Configuration utilities for the distributed training workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Encapsulates hyperparameters and runtime settings for training."""

    model_name: str
    dataset_path: str
    output_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    sequence_length: int
    master_addr: str
    master_port: int
    rank: int
    world_size: int
    backend: str
    local_rank: int
    num_workers: int
    gradient_clip_norm: Optional[float]
    mixed_precision: bool
    log_interval: int
    seed: int
    save_every: Optional[int]

    @classmethod
    def from_namespace(cls, args: object) -> "TrainingConfig":
        """Builds a configuration instance from parsed CLI arguments."""

        env_rank = os.environ.get("RANK")
        env_world_size = os.environ.get("WORLD_SIZE")
        env_local_rank = os.environ.get("LOCAL_RANK")
        rank = args.rank if args.rank is not None else int(env_rank or 0)
        world_size = args.world_size if args.world_size is not None else int(env_world_size or 1)
        local_rank = args.local_rank if args.local_rank is not None else int(env_local_rank or rank)
        master_port = int(args.master_port)
        gradient_clip = args.gradient_clip_norm if args.gradient_clip_norm is not None else None
        save_every = args.save_every if args.save_every is not None else None
        return cls(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            sequence_length=args.sequence_length,
            master_addr=args.master_addr,
            master_port=master_port,
            rank=rank,
            world_size=world_size,
            backend=args.backend,
            local_rank=local_rank,
            num_workers=args.num_workers,
            gradient_clip_norm=gradient_clip,
            mixed_precision=args.mixed_precision,
            log_interval=args.log_interval,
            seed=args.seed,
            save_every=save_every,
        )

    def is_distributed(self) -> bool:
        """Indicates whether multiple processes participate in training."""

        return self.world_size > 1

    def is_root_process(self) -> bool:
        """Detects if the current process is responsible for coordination."""

        return self.rank == 0
