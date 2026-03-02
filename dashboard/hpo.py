"""Background HPO session manager for the integrated dashboard.

Implements Latin Hypercube DOE → Optuna TPE two-phase optimisation,
running each trial in a background thread via Trainer.fit().
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import torch

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

from src.pipeline import (
    load_airfrans_data, convert_to_pyg, build_model, build_physics_loss,
    Trainer, train_one_epoch, validate_one_epoch,
)
from dashboard.runner import build_lr_scheduler


# ── Param spec ──────────────────────────────────────────────────────────────

@dataclass
class HpoParamSpec:
    """Describes one hyperparameter for optimisation."""
    name: str
    category: str
    type: str          # "int" | "float" | "categorical" | "bool"
    min: float = 0.0
    max: float = 1.0
    step: float = 0.0  # 0 = continuous
    log_scale: bool = False
    choices: list = field(default_factory=list)
    default: Any = None
    enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# ── Default search space ─────────────────────────────────────────────────────

DEFAULT_SEARCH_SPACE: list[HpoParamSpec] = [
    # Model
    HpoParamSpec("hidden_dim",       "model",          "int",         min=64,    max=256,   step=32,  default=128),
    HpoParamSpec("num_layers",       "model",          "int",         min=8,     max=20,    step=2,   default=14),
    HpoParamSpec("dropout",          "model",          "float",       min=0.0,   max=0.3,   step=0.05, default=0.1),
    HpoParamSpec("num_global_tokens","model",          "int",         min=0,     max=8,     step=1,   default=0),
    # Optimizer
    HpoParamSpec("lr",               "optimizer",      "float",       min=1e-5,  max=1e-2,  log_scale=True, default=1e-3),
    HpoParamSpec("weight_decay",     "optimizer",      "float",       min=1e-6,  max=1e-3,  log_scale=True, default=1e-4),
    HpoParamSpec("optimizer_type",   "optimizer",      "categorical", choices=["adam", "adamw"], default="adamw"),
    # Scheduler
    HpoParamSpec("scheduler_type",   "scheduler",      "categorical",
                 choices=["CosineAnnealingLR", "StepLR", "ReduceLROnPlateau", "ExponentialLR", "Constant"],
                 default="CosineAnnealingLR"),
    HpoParamSpec("scheduler_T_max",  "scheduler",      "int",         min=20,    max=200,   step=10,  default=100),
    HpoParamSpec("scheduler_gamma",  "scheduler",      "float",       min=0.5,   max=0.99,  step=0.01, default=0.95),
    # Physics
    HpoParamSpec("cont_weight",      "physics",        "float",       min=0.001, max=1.0,   log_scale=True, default=0.05),
    HpoParamSpec("mom_weight",       "physics",        "float",       min=0.001, max=1.0,   log_scale=True, default=0.05),
    HpoParamSpec("bc_weight",        "physics",        "float",       min=0.01,  max=1.0,   log_scale=True, default=0.01),
    HpoParamSpec("cont_target",      "physics",        "float",       min=0.01,  max=1.0,   log_scale=True, default=0.1),
    HpoParamSpec("mom_target",       "physics",        "float",       min=0.01,  max=1.0,   log_scale=True, default=0.1),
    HpoParamSpec("ramp_start_epoch", "physics",        "int",         min=0,     max=50,    step=5,   default=20),
    HpoParamSpec("ramp_epochs",      "physics",        "int",         min=10,    max=80,    step=10,  default=30),
    # Training
    HpoParamSpec("batch_size",       "training",       "int",         min=1,     max=8,     step=1,   default=4),
    HpoParamSpec("grad_clip",        "training",       "float",       min=0.1,   max=10.0,  step=0.1, default=1.0),
    HpoParamSpec("amp",              "training",       "bool",        choices=[True, False],  default=True),
    # Regularization
    HpoParamSpec("label_smoothing",  "regularization", "float",       min=0.0,   max=0.1,   step=0.01, default=0.0),
]


# ── LHS sample generation ────────────────────────────────────────────────────

def generate_lhs_samples(specs: list[HpoParamSpec], n: int, seed: int = 42) -> list[dict]:
    """Generate *n* space-filling samples via Latin Hypercube Sampling.

    Only enabled params are varied; disabled ones are filled with their defaults.
    Returns a list of dicts mapping param name → sampled value.
    """
    from scipy.stats.qmc import LatinHypercube
    import numpy as np

    enabled_numeric = [s for s in specs if s.enabled and s.type in ("int", "float")]
    enabled_cat     = [s for s in specs if s.enabled and s.type in ("categorical", "bool")]
    disabled        = [s for s in specs if not s.enabled]

    if enabled_numeric:
        lhs = LatinHypercube(d=len(enabled_numeric), seed=seed)
        unit_samples = lhs.random(n)           # (n, d) in [0, 1]
    else:
        unit_samples = np.zeros((n, 0))

    samples = []
    for i in range(n):
        params: dict[str, Any] = {}

        # Numeric params — map unit cube to param range
        for j, spec in enumerate(enabled_numeric):
            u = float(unit_samples[i, j])
            if spec.log_scale and spec.min > 0:
                raw = math.exp(
                    math.log(spec.min) + u * (math.log(spec.max) - math.log(spec.min))
                )
            else:
                raw = spec.min + u * (spec.max - spec.min)

            if spec.type == "int":
                step = max(1, int(spec.step)) if spec.step > 0 else 1
                val: Any = max(int(spec.min), min(int(spec.max), int(round(raw / step) * step)))
            else:
                if spec.step > 0 and not spec.log_scale:
                    steps_count = round((raw - spec.min) / spec.step)
                    raw = spec.min + steps_count * spec.step
                val = max(float(spec.min), min(float(spec.max), float(raw)))

            params[spec.name] = val

        # Categorical / bool — distribute uniformly across choices
        for k, spec in enumerate(enabled_cat):
            choices = [True, False] if spec.type == "bool" else list(spec.choices)
            params[spec.name] = choices[i % len(choices)] if choices else spec.default

        # Disabled params — use default
        for spec in disabled:
            params[spec.name] = spec.default

        samples.append(params)

    return samples


# ── Internal helpers ─────────────────────────────────────────────────────────

def _suggest_params(trial: Any, specs: list[HpoParamSpec]) -> dict:
    """Map Optuna *trial* suggestions to a flat param dict."""
    params: dict[str, Any] = {}
    for spec in specs:
        if not spec.enabled:
            params[spec.name] = spec.default
            continue

        if spec.type == "int":
            step = max(1, int(spec.step)) if spec.step > 0 else 1
            params[spec.name] = trial.suggest_int(
                spec.name, int(spec.min), int(spec.max), step=step
            )
        elif spec.type == "float":
            if spec.step > 0 and not spec.log_scale:
                params[spec.name] = trial.suggest_float(
                    spec.name, spec.min, spec.max, step=spec.step
                )
            else:
                params[spec.name] = trial.suggest_float(
                    spec.name, spec.min, spec.max, log=spec.log_scale
                )
        elif spec.type in ("categorical", "bool"):
            choices = [True, False] if spec.type == "bool" else list(spec.choices)
            params[spec.name] = trial.suggest_categorical(spec.name, choices)

    return params


def _build_cfg(params: dict, base_cfg: dict, epochs_per_trial: int) -> dict:
    """Map an HPO param dict to the flat training config expected by runner._run."""
    cfg = dict(base_cfg)

    # Model
    cfg["hidden_dim"]        = int(params.get("hidden_dim", 128))
    cfg["num_layers"]        = int(params.get("num_layers", 14))
    cfg["dropout"]           = float(params.get("dropout", 0.1))
    cfg["num_global_tokens"] = int(params.get("num_global_tokens", 0))

    # Optimizer
    cfg["lr"]             = float(params.get("lr", 1e-3))
    cfg["weight_decay"]   = float(params.get("weight_decay", 1e-4))
    cfg["optimizer_type"] = params.get("optimizer_type", "adamw")

    # Scheduler
    cfg["scheduler_type"]  = params.get("scheduler_type", "CosineAnnealingLR")
    cfg["scheduler_T_max"] = int(params.get("scheduler_T_max", 100))
    cfg["scheduler_gamma"] = float(params.get("scheduler_gamma", 0.95))

    # Physics — nested format expected by build_physics_loss
    ramp_start  = int(params.get("ramp_start_epoch", 20))
    ramp_epochs = int(params.get("ramp_epochs", 30))
    cfg["continuity"] = {
        "weight":           float(params.get("cont_weight", 0.05)),
        "target":           float(params.get("cont_target", 0.1)),
        "ramp_start_epoch": ramp_start,
        "ramp_epochs":      ramp_epochs,
    }
    cfg["momentum"] = {
        "weight":           float(params.get("mom_weight", 0.05)),
        "target":           float(params.get("mom_target", 0.1)),
        "ramp_start_epoch": ramp_start,
        "ramp_epochs":      ramp_epochs,
    }
    cfg["bc"] = {
        "weight":           float(params.get("bc_weight", 0.01)),
        "ramp_start_epoch": ramp_start,
        "ramp_epochs":      ramp_epochs,
    }

    # Training
    cfg["batch_size"] = int(params.get("batch_size", 4))
    cfg["amp"]        = bool(params.get("amp", True))
    cfg["grad_clip"]  = float(params.get("grad_clip", 1.0))
    cfg["num_epochs"] = epochs_per_trial

    return cfg


# ── State / session ──────────────────────────────────────────────────────────

@dataclass
class HpoState:
    state: str = "idle"          # idle | loading | running | stopping | stopped | completed | failed
    phase: str = "lhs"           # lhs | tpe
    current_trial: int = 0
    total_trials: int = 0
    n_lhs: int = 0
    best_value: Optional[float] = None
    best_params: Optional[dict] = None
    trials: list = field(default_factory=list)
    error_message: str = ""
    elapsed_sec: float = 0.0


class HpoSession:
    """Manages one HPO run (LHS → TPE) in a background daemon thread."""

    def __init__(self):
        self._lock   = threading.Lock()
        self._state  = HpoState()
        self._stop_flag   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Data cache — loaded once, shared across all trials
        self._raw_data: Any  = None
        self._bundle_cache: dict[int, Any] = {}
        self._start_time: float = 0.0

    # ── Public API ──────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_status(self) -> dict:
        with self._lock:
            d = {
                "state":         self._state.state,
                "phase":         self._state.phase,
                "current_trial": self._state.current_trial,
                "total_trials":  self._state.total_trials,
                "n_lhs":         self._state.n_lhs,
                "best_value":    self._state.best_value,
                "best_params":   self._state.best_params,
                "trials":        list(self._state.trials),
                "error_message": self._state.error_message,
                "elapsed_sec":   self._state.elapsed_sec,
            }
        return d

    def request_stop(self):
        self._stop_flag.set()
        with self._lock:
            if self._state.state == "running":
                self._state.state = "stopping"

    def start(self, search_space_dicts: list[dict], settings: dict):
        if self.is_running:
            raise RuntimeError("HPO already running")
        if not _OPTUNA_AVAILABLE:
            raise RuntimeError("optuna is not installed — run: pip install optuna")

        self._stop_flag.clear()

        # Parse search space from dicts (serialisable from API)
        specs = []
        for s in search_space_dicts:
            specs.append(HpoParamSpec(
                name=s["name"],
                category=s["category"],
                type=s["type"],
                min=float(s.get("min", 0)),
                max=float(s.get("max", 1)),
                step=float(s.get("step", 0)),
                log_scale=bool(s.get("log_scale", False)),
                choices=list(s.get("choices", [])),
                default=s.get("default"),
                enabled=bool(s.get("enabled", True)),
            ))

        n_lhs        = int(settings.get("n_lhs", 20))
        total_trials = int(settings.get("total_trials", 50))

        with self._lock:
            self._state = HpoState(
                state="loading",
                total_trials=total_trials,
                n_lhs=n_lhs,
            )

        self._thread = threading.Thread(
            target=self._run,
            args=(specs, settings),
            daemon=True,
        )
        self._thread.start()

    # ── Internal helpers ─────────────────────────────────────────

    def _set_state(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._state, k, v)

    def _get_bundle(self, batch_size: int, task: str, seed: int):
        """Return a cached DataBundle for *batch_size*, loading raw data if needed."""
        if self._raw_data is None:
            self._raw_data = load_airfrans_data(task=task, seed=seed)
        if batch_size not in self._bundle_cache:
            self._bundle_cache[batch_size] = convert_to_pyg(
                self._raw_data, batch_size=batch_size, num_workers=0
            )
        return self._bundle_cache[batch_size]

    # ── Background thread ────────────────────────────────────────

    def _run(self, specs: list[HpoParamSpec], settings: dict):
        import optuna as _optuna

        task              = settings.get("task", "scarce")
        seed              = int(settings.get("seed", 42))
        n_lhs             = int(settings.get("n_lhs", 20))
        total_trials      = int(settings.get("total_trials", 50))
        epochs_per_trial  = int(settings.get("epochs_per_trial", 30))
        pruning_patience  = int(settings.get("pruning_patience", 10))
        device            = settings.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        base_cfg          = {"task": task, "seed": seed, "device": device}

        self._start_time = time.time()

        try:
            # ── Load data once ───────────────────────────────────
            self._set_state(state="loading")
            self._get_bundle(batch_size=4, task=task, seed=seed)
            self._set_state(state="running", phase="lhs")

            # ── Create Optuna study ──────────────────────────────
            study = _optuna.create_study(
                direction="minimize",
                sampler=_optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_lhs),
                pruner=_optuna.pruners.MedianPruner(
                    n_startup_trials=max(2, n_lhs // 4),
                    n_warmup_steps=5,
                    interval_steps=1,
                ),
            )

            # ── Enqueue LHS samples ──────────────────────────────
            lhs_samples = generate_lhs_samples(specs, n_lhs, seed=seed)
            enabled_names = {s.name for s in specs if s.enabled}
            for sample in lhs_samples:
                study.enqueue_trial({k: v for k, v in sample.items() if k in enabled_names})

            # ── Stop callback ────────────────────────────────────
            def _stop_cb(study, trial):
                if self._stop_flag.is_set():
                    study.stop()

            # ── Objective ────────────────────────────────────────
            def objective(trial):
                trial_num = trial.number
                phase = "lhs" if trial_num < n_lhs else "tpe"

                with self._lock:
                    self._state.current_trial = trial_num + 1
                    self._state.phase = phase
                    self._state.elapsed_sec = time.time() - self._start_time
                    self._state.trials.append({
                        "number": trial_num,
                        "phase":  phase,
                        "state":  "running",
                        "value":  None,
                        "params": {},
                    })

                params  = _suggest_params(trial, specs)
                batch_size = int(params.get("batch_size", 4))
                bundle  = self._get_bundle(batch_size, task, seed)
                cfg     = _build_cfg(params, base_cfg, epochs_per_trial)

                model      = None
                optimizer  = None
                scheduler  = None
                criterion  = None
                pruned_ev  = threading.Event()
                best_val   = [float("inf")]
                patience   = [0]

                try:
                    model = build_model({
                        "type": "EnhancedCFDModelWithGlobalContext",
                        "input": {
                            "node_dim":          7,
                            "edge_dim":          bundle.edge_dim,
                            "hidden_dim":        cfg["hidden_dim"],
                            "num_layers":        cfg["num_layers"],
                            "num_global_tokens": cfg["num_global_tokens"],
                            "dropout":           cfg["dropout"],
                        },
                        "output": {"output_dim": 4},
                    }, device=device)

                    physics_cfg = {
                        k: cfg.get(k, {})
                        for k in ("continuity", "momentum", "bc")
                    }
                    criterion = build_physics_loss(
                        physics_cfg, steps_per_epoch=len(bundle.train_loader)
                    )

                    opt_type = cfg.get("optimizer_type", "adamw").lower()
                    if opt_type == "adam":
                        optimizer = torch.optim.Adam(
                            model.parameters(),
                            lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"],
                        )
                    else:
                        optimizer = torch.optim.AdamW(
                            model.parameters(),
                            lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"],
                        )

                    scheduler = build_lr_scheduler(optimizer, cfg)

                    def on_epoch_end(*, epoch, train_logs, val_logs, lr, is_best):
                        val_loss = float(val_logs.get("total_loss", float("inf")))

                        # Report to Optuna for pruning decisions
                        trial.report(val_loss, epoch)

                        if trial.should_prune():
                            pruned_ev.set()
                            return True

                        # Manual patience-based early stopping
                        if val_loss < best_val[0]:
                            best_val[0] = val_loss
                            patience[0] = 0
                        else:
                            patience[0] += 1
                            if patience[0] >= pruning_patience:
                                pruned_ev.set()
                                return True

                        with self._lock:
                            self._state.elapsed_sec = time.time() - self._start_time

                        return self._stop_flag.is_set()

                    trainer = Trainer(
                        model, optimizer, scheduler, criterion,
                        device=device, live_plot=False, amp=cfg.get("amp", True),
                    )
                    trainer.fit(
                        bundle.train_loader,
                        bundle.val_loader,
                        num_epochs=epochs_per_trial,
                        routine={
                            "train":           train_one_epoch,
                            "validate":        validate_one_epoch,
                            "save_checkpoint": lambda *a, **kw: None,
                            "log_metrics":     lambda *a, **kw: None,
                        },
                        on_epoch_end=on_epoch_end,
                    )

                    final_val = best_val[0]

                finally:
                    del model, optimizer, scheduler, criterion
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Update trial record
                with self._lock:
                    rec_params = {k: v for k, v in params.items() if k in enabled_names}
                    for rec in self._state.trials:
                        if rec["number"] == trial_num:
                            rec["state"]  = "pruned" if pruned_ev.is_set() else "complete"
                            rec["value"]  = None if pruned_ev.is_set() else final_val
                            rec["params"] = rec_params
                            break

                    if not pruned_ev.is_set():
                        cur_best = self._state.best_value
                        if cur_best is None or final_val < cur_best:
                            self._state.best_value  = final_val
                            self._state.best_params = dict(params)

                    self._state.elapsed_sec = time.time() - self._start_time

                if pruned_ev.is_set():
                    raise _optuna.exceptions.TrialPruned()

                return final_val

            study.optimize(
                objective,
                n_trials=total_trials,
                callbacks=[_stop_cb],
                gc_after_trial=True,
            )

            elapsed = time.time() - self._start_time
            with self._lock:
                if self._state.state not in ("stopping", "failed"):
                    self._state.state = "completed"
                self._state.elapsed_sec = elapsed

        except Exception as exc:
            import traceback
            traceback.print_exc()
            with self._lock:
                self._state.state         = "failed"
                self._state.error_message = str(exc)
                self._state.elapsed_sec   = time.time() - self._start_time

        finally:
            self._bundle_cache.clear()
            self._raw_data = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with self._lock:
                if self._state.state == "stopping":
                    self._state.state = "stopped"
