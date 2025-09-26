"""Dataset preparation and distributed data loading."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .config import TrainingConfig


class TokenBlockDataset(Dataset):
    """Slices tokenized text into fixed-length language modeling blocks."""

    def __init__(self, token_ids: torch.Tensor, block_size: int) -> None:
        if token_ids.ndim != 1:
            token_ids = token_ids.view(-1)
        usable_length = (token_ids.numel() - 1) // block_size * block_size
        if usable_length <= 0:
            raise ValueError("Dataset is too small for the configured sequence length")
        inputs = token_ids[:usable_length]
        labels = token_ids[1 : usable_length + 1]
        self.inputs = inputs.view(-1, block_size)
        self.labels = labels.view(-1, block_size)
        self.attention = torch.ones_like(self.inputs)

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.inputs[index],
            "labels": self.labels[index],
            "attention_mask": self.attention[index],
        }


class DataManager:
    """Loads raw corpora, tokenizes them, and builds sharded dataloaders."""

    def __init__(self, config: TrainingConfig, tokenizer) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self._dataset: Optional[TokenBlockDataset] = None

    def prepare_dataset(self, dataset_path: str) -> TokenBlockDataset:
        """Reads and tokenizes the corpus for language model training."""

        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        text = path.read_text(encoding="utf-8")
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
        )
        token_ids = encoded["input_ids"].squeeze(0)
        self._dataset = TokenBlockDataset(token_ids, self.config.sequence_length)
        return self._dataset

    def create_dataloader(self) -> DataLoader:
        """Builds a DataLoader backed by a DistributedSampler."""

        if self._dataset is None:
            raise RuntimeError("Dataset has not been prepared")
        sampler = DistributedSampler(
            self._dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=True,
        )
        return DataLoader(
            self._dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self._should_pin_memory(),
            drop_last=True,
        )

    def _should_pin_memory(self) -> bool:
        return torch.cuda.is_available() and self.config.backend == "nccl"
