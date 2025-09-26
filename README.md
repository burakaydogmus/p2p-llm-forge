# P2P-LLM-Forge

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

P2P-LLM-Forge is an MVP (Minimum Viable Product) that demonstrates data-parallel training of causal language models across peer-to-peer clusters using PyTorch DistributedDataParallel (DDP).

This project addresses the challenges faced in modern distributed machine learning applications and enables researchers and developers to test multi-node training environments.

## ✨ Features

- **Centralized Configuration**: Centralized configuration system for hyperparameters and distributed runtime settings
- **Automatic Process Group Management**: Automatic process group setup and teardown for PyTorch DDP
- **Intelligent Data Processing**: Tokenization and sharded data loading for text corpora on shared storage
- **Advanced Training Loop**: Gradient synchronization, optional gradient clipping, and checkpointing from rank 0
- **Flexible Distribution**: Support for both single-machine and multi-node distributed training
- **CPU/GPU Compatibility**: Works in both CPU and GPU environments

## 📋 Requirements

- **Python**: 3.10 or higher
- **PyTorch**: 2.8+
- **CUDA**: For GPU training (optional)
- **uv**: Modern Python package manager

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd p2p-llm-forge
```

### 2. Install Dependencies
```bash
uv sync
```

This command will install all required dependencies (PyTorch, Transformers, tqdm).

## 📖 Usage

### Single-Machine Test

To run a quick test with the sample dataset included in the project:

```bash
uv run python tests/smoke_single_node.py
```

This command:
- Downloads the small `sshleifer/tiny-gpt2` model
- Uses sample data from `data/sample_corpus.txt`
- Performs 1 epoch of training using the CPU-friendly `gloo` backend
- Saves results to the `artifacts/smoke/` directory

### Distributed Training

For multi-node training, run the following command on each node:

```bash
# Basic usage
uv run p2p-llm-forge \
  --model-name gpt2 \
  --dataset-path /path/to/dataset.txt \
  --output-dir ./artifacts \
  --epochs 1 \
  --batch-size 2 \
  --sequence-length 128

# Multi-node training with torchrun
torchrun --nnodes=2 --nproc_per_node=1 \
  --master_addr=node1 --master_port=29500 \
  uv run p2p-llm-forge \
  --model-name gpt2 \
  --dataset-path /path/to/dataset.txt \
  --output-dir ./artifacts \
  --epochs 10 \
  --batch-size 4 \
  --sequence-length 512 \
  --backend nccl
```

### Command Line Options

To see all available options:

```bash
uv run p2p-llm-forge --help
```

**Important Parameters:**
- `--model-name`: HuggingFace model identifier
- `--dataset-path`: Path to training dataset
- `--output-dir`: Directory for saving checkpoints
- `--epochs`: Number of training epochs
- `--batch-size`: Global batch size per process
- `--sequence-length`: Maximum tokens per training sequence
- `--backend`: Distributed backend (`nccl` or `gloo`)
- `--master-addr/--master-port`: Master node connection information

## 🏗️ Project Structure

```
p2p-llm-forge/
├── src/
│   └── p2p_llm_forge/
│       ├── __init__.py          # Package initialization
│       ├── __main__.py          # CLI entry point
│       ├── config.py            # Configuration management
│       ├── data.py              # Data loading and processing
│       ├── distributed.py       # Distributed training management
│       └── trainer.py           # Training loop
├── data/
│   └── sample_corpus.txt        # Sample dataset
├── tests/
│   └── smoke_single_node.py     # Single-machine test scenario
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

### Core Modules

- **`config.py`**: Centralized configuration for training hyperparameters and distributed settings
- **`data.py`**: Tokenization and sharded data loading operations
- **`distributed.py`**: PyTorch DDP process group management
- **`trainer.py`**: Training loop, loss computation, and optimization

## ⚙️ Configuration

The project supports the following configuration options:

### Distributed Training
- **Backend**: `nccl` (GPU) or `gloo` (CPU)
- **Rank/World Size**: Process ranking and total number of processes
- **Master Node**: Master coordination node information

### Model and Data
- **Model**: Any HuggingFace causal LM model
- **Data**: Text file format datasets
- **Sequence Length**: Maximum token length

### Training Parameters
- **Batch Size**: Global batch size (total across all processes)
- **Learning Rate**: Optimization learning rate
- **Epochs**: Number of training epochs
- **Gradient Clipping**: Gradient clipping threshold

### Development

```bash
# Set up development environment
uv sync --editable

# Run tests
uv run python tests/smoke_single_node.py

# Code quality checks
uv run ruff check .
uv run ruff format .
```

## 📞 Contact

Feel free to open an issue for questions and feedback!

---
