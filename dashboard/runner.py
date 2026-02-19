"""Background training session manager for the integrated dashboard."""

from __future__ import annotations

import threading
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import json

from src.pipeline import (
    load_airfrans_data, convert_to_pyg, build_model, build_physics_loss,
    Trainer, train_one_epoch, validate_one_epoch,
    save_model_checkpoint, log_training_metrics,
)
from src.benchmark import run_benchmark_and_log_experiment


LOSS_KEYS = ("total_loss", "mse_loss", "continuity_loss", "momentum_loss", "bc_loss")


# ---------------------------------------------------------------------------
# LR Scheduler: canonical names, aliases, param specs, validation
# ---------------------------------------------------------------------------

SCHEDULER_ALIASES: dict[str, str] = {
    # CLI-style aliases -> canonical class names
    "constant": "Constant",
    "none": "Constant",
    "cosine": "CosineAnnealingLR",
    "cosine_annealing": "CosineAnnealingLR",
    "step": "StepLR",
    "plateau": "ReduceLROnPlateau",
    "reduce_on_plateau": "ReduceLROnPlateau",
    "cosine_warm_restarts": "CosineAnnealingWarmRestarts",
    "warm_restarts": "CosineAnnealingWarmRestarts",
    "multistep": "MultiStepLR",
    "multi_step": "MultiStepLR",
    "exponential": "ExponentialLR",
}

ALLOWED_SCHEDULER_TYPES = frozenset([
    "Constant",
    "CosineAnnealingLR",
    "StepLR",
    "ReduceLROnPlateau",
    "CosineAnnealingWarmRestarts",
    "MultiStepLR",
    "ExponentialLR",
])

# Per-type parameter specifications: (key, type, default, constraint_description)
# constraint_description is None when no extra validation is needed.
SCHEDULER_PARAM_SPECS: dict[str, list[tuple[str, type, object, str | None]]] = {
    "Constant": [],
    "CosineAnnealingLR": [
        ("scheduler_T_max", int, 100, "positive"),
        ("scheduler_eta_min", float, 0.0, "non_negative"),
    ],
    "StepLR": [
        ("scheduler_step_size", int, 10, "positive"),
        ("scheduler_gamma", float, 0.1, "unit_range"),
    ],
    "ReduceLROnPlateau": [
        ("scheduler_factor", float, 0.5, "unit_range"),
        ("scheduler_patience", int, 10, "non_negative"),
        ("scheduler_min_lr", float, 1e-6, "non_negative"),
    ],
    "CosineAnnealingWarmRestarts": [
        ("scheduler_T_0", int, 10, "positive"),
        ("scheduler_T_mult", int, 1, "positive"),
        ("scheduler_eta_min", float, 0.0, "non_negative"),
    ],
    "MultiStepLR": [
        ("scheduler_milestones", list, [30, 60, 90], None),
        ("scheduler_gamma", float, 0.1, "unit_range"),
    ],
    "ExponentialLR": [
        ("scheduler_gamma", float, 0.95, "unit_range"),
    ],
}

_CONSTRAINT_VALIDATORS = {
    "positive": lambda v: v > 0,
    "non_negative": lambda v: v >= 0,
    "unit_range": lambda v: 0 < v <= 1,
}

# Warmup params validated globally (apply to all non-Constant schedulers)
WARMUP_PARAM_SPECS: list[tuple[str, type, object, str | None]] = [
    ("scheduler_warmup_steps", int, 0, "non_negative"),
    ("scheduler_warmup_start_factor", float, 0.1, "unit_range"),
    ("scheduler_warmup_end_factor", float, 1.0, "positive"),
]

# Scheduler types that cannot be combined with LinearLR warmup because
# SequentialLR calls .step() without arguments.
_NO_WARMUP_TYPES = frozenset(["ReduceLROnPlateau"])


def flatten_scheduler_config(raw: dict) -> dict:
    """Accept both flat and nested scheduler config payloads.

    Flat (current dashboard format)::

        {"scheduler_type": "cosine", "scheduler_T_max": 50, "base_lr": 0.001}

    Nested (API-client-friendly format)::

        {"scheduler": {"scheduler_type": "cosine", "scheduler_T_max": 50}, "base_lr": 0.001}

    In the nested form the inner ``scheduler`` dict is merged into the outer
    dict so downstream code always sees the flat representation.  Keys in the
    outer dict take precedence over keys in the nested ``scheduler`` dict.
    """
    nested = raw.get("scheduler")
    if not isinstance(nested, dict):
        return raw
    merged = {**nested, **{k: v for k, v in raw.items() if k != "scheduler"}}
    return merged


def normalize_scheduler_type(raw_type: str) -> str:
    """Map a raw scheduler type string to its canonical name.

    Accepts canonical names directly, or case-insensitive CLI aliases.
    Raises ValueError for unrecognised types.
    """
    if raw_type in ALLOWED_SCHEDULER_TYPES:
        return raw_type
    canonical = SCHEDULER_ALIASES.get(raw_type.lower().strip())
    if canonical is not None:
        return canonical
    raise ValueError(
        f"Unknown scheduler type: '{raw_type}'. "
        f"Allowed: {sorted(ALLOWED_SCHEDULER_TYPES)}"
    )


def _validate_param_specs(out: dict, specs: list) -> None:
    """Validate and cast a list of (key, type, default, constraint) specs in-place."""
    for key, typ, default, constraint in specs:
        raw = out.get(key, default)
        try:
            if typ is list:
                if isinstance(raw, str):
                    raw = raw.strip()
                    if raw.startswith("[") and raw.endswith("]"):
                        raw = json.loads(raw)
                    elif "," in raw:
                        raw = [item.strip() for item in raw.split(",")]
                    else:
                        raw = [raw]
                if not isinstance(raw, list):
                    raise TypeError
                value = [int(x) for x in raw]
            else:
                value = typ(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Scheduler param '{key}' must be {typ.__name__}, got {raw!r}"
            ) from exc
        if constraint is not None:
            validator = _CONSTRAINT_VALIDATORS[constraint]
            if not validator(value):
                raise ValueError(
                    f"Scheduler param '{key}' failed constraint '{constraint}': {value}"
                )
        out[key] = value


def validate_scheduler_config(cfg: dict) -> dict:
    """Validate and normalise a flat scheduler config dict.

    Returns a new dict with the canonical ``scheduler_type`` and all
    parameters (including warmup) cast to correct types with defaults filled in.
    Raises ``ValueError`` on invalid input.
    """
    out = dict(cfg)
    stype = normalize_scheduler_type(out.get("scheduler_type", "CosineAnnealingLR"))
    out["scheduler_type"] = stype

    # Per-type params
    _validate_param_specs(out, SCHEDULER_PARAM_SPECS.get(stype, []))

    # Warmup params (validated for all types; only used when warmup_steps > 0)
    _validate_param_specs(out, WARMUP_PARAM_SPECS)

    # Cross-param check: warmup is incompatible with ReduceLROnPlateau
    if out["scheduler_warmup_steps"] > 0 and stype in _NO_WARMUP_TYPES:
        raise ValueError(
            f"scheduler_warmup_steps > 0 is not supported with '{stype}' "
            f"because SequentialLR cannot call its metric-based .step()."
        )

    return out


def _build_base_scheduler(optimizer, cfg: dict, stype: str):
    """Build the base scheduler (no warmup wrapper). Config must be pre-validated."""
    if stype == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["scheduler_T_max"],
            eta_min=cfg["scheduler_eta_min"],
        )
    elif stype == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["scheduler_step_size"],
            gamma=cfg["scheduler_gamma"],
        )
    elif stype == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg["scheduler_factor"],
            patience=cfg["scheduler_patience"],
            min_lr=cfg["scheduler_min_lr"],
        )
    elif stype == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg["scheduler_T_0"],
            T_mult=cfg["scheduler_T_mult"],
            eta_min=cfg["scheduler_eta_min"],
        )
    elif stype == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg["scheduler_milestones"],
            gamma=cfg["scheduler_gamma"],
        )
    elif stype == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg["scheduler_gamma"],
        )
    else:
        raise ValueError(f"Unknown scheduler type: {stype}")


def build_lr_scheduler(optimizer, cfg: dict):
    """Build a PyTorch LR scheduler from a flat config dict.

    Returns None for Constant (no scheduling).  When ``scheduler_warmup_steps``
    is > 0 the base scheduler is wrapped with a ``LinearLR`` warmup phase via
    ``SequentialLR``.  Warmup is not supported with ``ReduceLROnPlateau``.

    The config is validated and normalised before use.
    """
    cfg = validate_scheduler_config(cfg)
    stype = cfg["scheduler_type"]
    warmup_steps = cfg["scheduler_warmup_steps"]

    if stype == "Constant":
        return None

    base = _build_base_scheduler(optimizer, cfg, stype)

    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=cfg["scheduler_warmup_start_factor"],
            end_factor=cfg["scheduler_warmup_end_factor"],
            total_iters=warmup_steps,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, base],
            milestones=[warmup_steps],
        )

    return base


def simulate_lr_schedule(
    cfg: dict,
    num_epochs: int,
    base_lr: float,
    metric_series: list[float] | None = None,
) -> dict:
    """Simulate an LR schedule matching Trainer.fit() logging semantics.

    Trainer.fit() steps the scheduler *then* reads the LR, so the logged LR
    at epoch N is the post-step value (the LR used in epoch N+1).  This
    function reproduces that behaviour so the preview chart is consistent
    with the training dashboard.

    Args:
        cfg: Flat scheduler config dict (will be validated/normalised).
        num_epochs: Number of epochs to simulate.
        base_lr: Initial optimizer LR.
        metric_series: Optional sequence of validation-loss values for
            ``ReduceLROnPlateau``.  Length must equal ``num_epochs``.
            Defaults to a flat (non-improving) sequence of 999.0 each epoch.

    Returns:
        dict with keys:
            ``epochs``   – list of epoch indices [0, …, num_epochs-1]
            ``lr``       – post-step LR at each epoch
            ``metadata`` – dict with scheduler_type, resolved config,
                           min_lr, max_lr, step_mode, metric_mode
    """
    validated = validate_scheduler_config(cfg)
    stype = validated["scheduler_type"]

    dummy = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(dummy.parameters(), lr=base_lr)
    scheduler = build_lr_scheduler(optimizer, validated)

    is_plateau = isinstance(
        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
    )

    if metric_series is None:
        metric_series = [999.0] * num_epochs
        metric_mode = "flat_worst_case"
    else:
        if len(metric_series) != num_epochs:
            raise ValueError(
                f"metric_series length {len(metric_series)} must equal num_epochs {num_epochs}"
            )
        metric_mode = "custom"

    lrs: list[float] = []
    for i in range(num_epochs):
        if scheduler is not None:
            if is_plateau:
                scheduler.step(metric_series[i])
            else:
                scheduler.step()
        lrs.append(float(optimizer.param_groups[0]["lr"]))

    return {
        "epochs": list(range(num_epochs)),
        "lr": lrs,
        "metadata": {
            "scheduler_type": stype,
            "resolved": {
                k: v for k, v in validated.items()
                if k.startswith("scheduler_")
            },
            "min_lr": min(lrs),
            "max_lr": max(lrs),
            "step_mode": "epoch",
            "metric_mode": metric_mode,
        },
    }


def build_scheduler_snapshot(cfg: dict, num_epochs: int, base_lr: float) -> dict:
    """Build a canonical scheduler snapshot for experiment metadata.

    Validates and normalises the config, then runs a quick preview simulation
    to capture first/final LR.  Safe to call after training has already used
    the scheduler — it creates a fresh dummy optimizer for simulation only.

    Returns a dict suitable for embedding in experiment metadata::

        {
            "type": "CosineAnnealingLR",
            "params": {"scheduler_T_max": 100, "scheduler_eta_min": 0.0, ...},
            "step_mode": "epoch",
            "first_lr": 0.001,
            "final_lr": 1e-06,
            "warmup_steps": 0,
        }

    On simulation failure (shouldn't occur after successful training) the
    ``first_lr`` / ``final_lr`` fields are omitted and ``simulation_error``
    is set to the error message instead.
    """
    try:
        validated = validate_scheduler_config(cfg)
    except ValueError as exc:
        # Config already passed validation when building the real scheduler;
        # this path should not be reached.
        return {"error": f"Snapshot validation failed: {exc}"}

    stype = validated["scheduler_type"]
    resolved_params = {
        k: v for k, v in validated.items()
        if k.startswith("scheduler_") and k != "scheduler_type"
    }

    snapshot: dict = {
        "type": stype,
        "params": resolved_params,
        "step_mode": "epoch",
        "warmup_steps": validated.get("scheduler_warmup_steps", 0),
    }

    try:
        preview = simulate_lr_schedule(validated, num_epochs, base_lr)
        lrs = preview["lr"]
        snapshot["first_lr"] = lrs[0]
        snapshot["final_lr"] = lrs[-1]
    except Exception as exc:
        snapshot["simulation_error"] = str(exc)

    return snapshot


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
            scheduler_snapshot = build_scheduler_snapshot(
                cfg,
                num_epochs=cfg.get("num_epochs", 20),
                base_lr=cfg.get("lr", 1e-3),
            )
            exp_id = run_benchmark_and_log_experiment(
                model=model,
                data_bundle=bundle,
                scfg={
                    "task": cfg.get("task", "scarce"),
                    "hidden": cfg.get("hidden_dim", 16),
                    "layers": cfg.get("num_layers", 14),
                    "scheduler": scheduler_snapshot,
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
