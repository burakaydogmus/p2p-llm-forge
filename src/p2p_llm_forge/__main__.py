"""CLI entrypoint for the P2P-LLM-Forge training workflow."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .config import TrainingConfig
from .data import DataManager
from .distributed import DistributedSetup
from .trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed causal language model training")
    parser.add_argument("--model-name", type=str, required=True, help="Pretrained model identifier")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the training corpus")
    parser.add_argument("--output-dir", type=str, default="./artifacts", help="Directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Global batch size per process")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Optimizer learning rate")
    parser.add_argument("--sequence-length", type=int, default=128, help="Tokens per training sequence")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1", help="Process group master address")
    parser.add_argument("--master-port", type=int, default=29500, help="Process group master port")
    parser.add_argument("--rank", type=int, default=None, help="Global rank for this process")
    parser.add_argument("--world-size", type=int, default=None, help="Total number of processes")
    parser.add_argument("--local-rank", type=int, default=None, help="Local GPU index for this process")
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"], help="Torch distributed backend")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--gradient-clip-norm", type=float, default=None, help="Max gradient norm for clipping")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between loss logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-every", type=int, default=None, help="Save checkpoints every N epochs")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = TrainingConfig.from_namespace(args)
    setup = DistributedSetup(config)
    setup.setup_process_group()
    seed_everything(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = config.sequence_length
    data_manager = DataManager(config, tokenizer)
    data_manager.prepare_dataset(config.dataset_path)
    dataloader = data_manager.create_dataloader()
    trainer = Trainer(config, dataloader)
    try:
        trainer.run_training_loop()
    finally:
        setup.global_barrier()
        if config.is_root_process():
            tokenizer.save_pretrained(Path(config.output_dir))
        setup.cleanup()


if __name__ == "__main__":
    main()
