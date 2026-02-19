"""Background training session manager for the integrated dashboard."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch

from src.pipeline import (
    load_airfrans_data, convert_to_pyg, build_model, build_physics_loss,
    Trainer, train_one_epoch, validate_one_epoch,
    save_model_checkpoint, log_training_metrics,
)
from src.benchmark import run_benchmark_and_log_experiment


LOSS_KEYS = ("total_loss", "mse_loss", "continuity_loss", "momentum_loss", "bc_loss")


def build_lr_scheduler(optimizer, cfg: dict):
    """Build a PyTorch LR scheduler from a flat config dict.

    Returns None for Constant (no scheduling).
    """
    stype = cfg.get("scheduler_type", "CosineAnnealingLR")

    if stype == "Constant":
        return None
    elif stype == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("scheduler_T_max", 100),
            eta_min=cfg.get("scheduler_eta_min", 0.0),
        )
    elif stype == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.get("scheduler_step_size", 10),
            gamma=cfg.get("scheduler_gamma", 0.1),
        )
    elif stype == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.get("scheduler_factor", 0.5),
            patience=int(cfg.get("scheduler_patience", 10)),
            min_lr=cfg.get("scheduler_min_lr", 1e-6),
        )
    elif stype == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.get("scheduler_T_0", 10),
            T_mult=int(cfg.get("scheduler_T_mult", 1)),
            eta_min=cfg.get("scheduler_eta_min", 0.0),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {stype}")


@dataclass
class TrainingState:
    state: str = "idle"  # idle | loading | training | benchmarking | completed | failed | stopping
    session_id: str = ""
    current_epoch: int = 0
    total_epochs: int = 0
    best_val: float = float("inf")
    best_epoch: int = -1
    elapsed_sec: float = 0.0
    error_message: str = ""
    experiment_id: str = ""
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=lambda: {
        "epochs": [], "train": {}, "val": {}, "lr": [],
    })


class TrainingSession:
    """Manages a single training run in a background thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = TrainingState()
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_status(self) -> dict:
        with self._lock:
            return asdict(self._state)

    def request_stop(self):
        self._stop_flag.set()
        with self._lock:
            if self._state.state == "training":
                self._state.state = "stopping"

    def start(self, config_dict: dict):
        if self.is_running:
            raise RuntimeError("Training already running")
        self._stop_flag.clear()
        self._state = TrainingState(
            state="loading",
            session_id=time.strftime("%Y%m%d_%H%M%S"),
            total_epochs=config_dict.get("num_epochs", 20),
            config=config_dict,
        )
        self._thread = threading.Thread(target=self._run, args=(config_dict,), daemon=True)
        self._thread.start()

    def _set_state(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._state, k, v)

    def _on_epoch_end(self, *, epoch, train_logs, val_logs, lr, is_best):
        """Callback invoked by Trainer.fit() after each epoch."""
        with self._lock:
            s = self._state
            s.current_epoch = epoch + 1
            s.elapsed_sec = time.time() - self._train_start
            if is_best:
                s.best_val = float(val_logs.get("total_loss", float("inf")))
                s.best_epoch = epoch

            s.metrics["epochs"].append(epoch)
            s.metrics["lr"].append(float(lr))
            for key in LOSS_KEYS:
                s.metrics["train"].setdefault(key, []).append(float(train_logs.get(key, float("nan"))))
                s.metrics["val"].setdefault(key, []).append(float(val_logs.get(key, float("nan"))))

        return self._stop_flag.is_set()

    def _run(self, cfg: dict):
        """Execute the full training pipeline in a background thread."""
        try:
            self._set_state(state="loading")

            # 0. Device
            device = cfg.get("device", "cuda")
            amp = cfg.get("amp", True)

            # 1. Data
            data = load_airfrans_data(task=cfg.get("task", "scarce"), seed=cfg.get("seed", 42))
            bundle = convert_to_pyg(
                data,
                batch_size=cfg.get("batch_size", 16),
                num_workers=cfg.get("num_workers", 4),
            )

            # 2. Model
            model = build_model({
                "type": "EnhancedCFDModelWithGlobalContext",
                "input": {
                    "node_dim": 7,
                    "edge_dim": bundle.edge_dim,
                    "hidden_dim": cfg.get("hidden_dim", 16),
                    "num_layers": cfg.get("num_layers", 14),
                    "num_global_tokens": cfg.get("num_global_tokens", 0),
                    "dropout": cfg.get("dropout", 0.1),
                },
                "output": {"output_dim": 4},
            }, device=device)

            # 3. Physics loss
            physics_cfg = {}
            for comp in ("continuity", "momentum", "bc"):
                sub = cfg.get(comp, {})
                if isinstance(sub, dict):
                    physics_cfg[comp] = sub
                else:
                    physics_cfg[comp] = {}
            criterion = build_physics_loss(physics_cfg, steps_per_epoch=len(bundle.train_loader))

            # 4. Optimizer & scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.get("lr", 1e-3),
                weight_decay=cfg.get("weight_decay", 1e-4),
            )
            scheduler = build_lr_scheduler(optimizer, cfg)

            # 5. Train
            self._set_state(state="training")
            self._train_start = time.time()
            trainer = Trainer(
                model, optimizer, scheduler, criterion,
                device=device, live_plot=False, amp=amp,
            )
            history = trainer.fit(
                bundle.train_loader, bundle.val_loader,
                num_epochs=cfg.get("num_epochs", 20),
                routine={
                    "train": train_one_epoch,
                    "validate": validate_one_epoch,
                    "save_checkpoint": save_model_checkpoint,
                    "log_metrics": log_training_metrics,
                },
                on_epoch_end=self._on_epoch_end,
            )
            training_duration = time.time() - self._train_start
            self._set_state(elapsed_sec=training_duration)

            # Update final train/val loss in history
            if history.get("train"):
                history["final_train_loss"] = history["train"][-1].get("total_loss", float("nan"))
            if history.get("val"):
                history["final_val_loss"] = history["val"][-1].get("total_loss", float("nan"))

            # 6. Benchmark scoring
            self._set_state(state="benchmarking")
            exp_id = run_benchmark_and_log_experiment(
                model=model,
                data_bundle=bundle,
                scfg={
                    "task": cfg.get("task", "scarce"),
                    "hidden": cfg.get("hidden_dim", 16),
                    "layers": cfg.get("num_layers", 14),
                },
                device=torch.device(device),
                train_summary=history,
                training_duration_sec=training_duration,
                notes=cfg.get("notes", ""),
            )

            self._set_state(state="completed", experiment_id=exp_id or "")

        except Exception as exc:
            self._set_state(state="failed", error_message=str(exc))
            import traceback
            traceback.print_exc()
