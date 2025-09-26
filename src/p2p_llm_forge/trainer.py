"""Training orchestration for distributed causal language modeling."""
from __future__ import annotations
from dataclasses import asdict

from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM

from .config import TrainingConfig
from .distributed import get_rank


class Trainer:
    """Coordinates model setup, optimization, and training loops."""

    def __init__(self, config: TrainingConfig, dataloader: DataLoader) -> None:
        self.config = config
        self.dataloader = dataloader
        self.device = self._resolve_device()
        self.model: Optional[torch.nn.Module] = None
        self._base_model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self._prepare_model()
        self._prepare_optimizer()
        self._prepare_precision()
        self.global_step: int = 0
        self.start_epoch: int = 0
        self._maybe_resume()

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available() and self.config.backend == "nccl":
            return torch.device("cuda", index=self.config.local_rank)
        return torch.device("cpu")

    def _prepare_model(self) -> None:
        base_model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        base_model.to(self.device)
        if hasattr(base_model.config, "pad_token_id") and base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = base_model.config.eos_token_id
        self._base_model = base_model
        if self.config.is_distributed():
            if self.device.type == "cuda":
                self.model = DDP(
                    base_model,
                    device_ids=[self.config.local_rank],
                    output_device=self.config.local_rank,
                )
            else:
                self.model = DDP(base_model)
        else:
            self.model = base_model

    def _prepare_optimizer(self) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been initialized")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

    def _prepare_precision(self) -> None:
        if self.device.type != "cuda":
            return
        if not self.config.mixed_precision:
            return
        self.scaler = torch.cuda.amp.GradScaler()

    def run_training_loop(self) -> None:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Trainer is not fully initialized")
        for epoch in range(self.start_epoch, self.config.epochs):
            loss = self._run_epoch(epoch)
            self.log_progress(f"Epoch {epoch + 1} completed with loss {loss:.4f}")
            if self.config.save_every and (epoch + 1) % self.config.save_every == 0:
                self.save_model(self.config.output_dir, epoch=epoch, step=self.global_step)
        self.save_model(self.config.output_dir, epoch=self.config.epochs - 1, step=self.global_step)

    def _run_epoch(self, epoch_id: int) -> float:
        if isinstance(self.dataloader.sampler, DistributedSampler):
            self.dataloader.sampler.set_epoch(epoch_id)
        running_loss = 0.0
        total_steps = 0
        for step, batch in enumerate(self.dataloader, start=1):
            loss = self._run_batch(batch)
            running_loss += loss
            total_steps += 1
            if step % self.config.log_interval == 0:
                self.log_progress(f"Epoch {epoch_id + 1} step {step} loss {loss:.4f}")
        if total_steps == 0:
            return 0.0
        return running_loss / total_steps

    def _run_batch(self, batch: dict[str, torch.Tensor]) -> float:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Trainer is not fully initialized")
        self.model.train()
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
            self.scaler.scale(loss).backward()
            self._apply_gradient_clipping()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self._apply_gradient_clipping()
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        return loss.detach().float().item()

    def _apply_gradient_clipping(self) -> None:
        if self.config.gradient_clip_norm is None:
            return
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self._parameters(), self.config.gradient_clip_norm)

    def _parameters(self) -> Iterable[torch.nn.Parameter]:
        if self.model is None:
            return []
        return self.model.parameters()

    def save_model(self, output_path: str, epoch: int | None = None, step: int | None = None) -> None:
        if not self.config.is_root_process():
            return
        if self._base_model is None:
            raise RuntimeError("Base model was not initialized")
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self._base_model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": epoch,
            "step": step,
            "config": asdict(self.config),
        }
        torch.save(state, path / "checkpoint.pt")
        
    def _maybe_resume(self) -> None:
        """Load training state if checkpoint exists in output_dir."""
        ckpt_path = Path(self.config.output_dir) / "checkpoint.pt"
        if not ckpt_path.exists():
            return

        state = torch.load(ckpt_path, map_location=self.device)

        if self._base_model is None:
            raise RuntimeError("Base model was not initialized")
        self._base_model.load_state_dict(state.get("model", {}))

        if self.optimizer is not None and state.get("optimizer"):
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler is not None and state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])

        self.start_epoch = int(state.get("epoch") or 0) + 1
        self.global_step = int(state.get("step") or 0)

        self.log_progress(f"Resumed from checkpoint: epoch={self.start_epoch}, step={self.global_step}")

    def log_progress(self, message: str) -> None:
        if get_rank() == 0:
            print(message)
