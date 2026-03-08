"""Background HPO session manager for the integrated dashboard.

Implements Latin Hypercube DOE → Optuna TPE two-phase optimisation,
running each trial in a background thread via Trainer.fit().
"""

from __future__ import annotations

import datetime
import math
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
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
    HpoParamSpec("num_global_tokens","model",          "int",         min=0,     max=4,     step=1,   default=0),
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
    # Physics — min=0 means "can be disabled"; log_scale=True gives finer resolution near zero
    HpoParamSpec("cont_weight",      "physics",        "float",       min=0.0,   max=0.5,   log_scale=True, default=0.05),
    HpoParamSpec("mom_weight",       "physics",        "float",       min=0.0,   max=0.5,   log_scale=True, default=0.05),
    HpoParamSpec("bc_weight",        "physics",        "float",       min=0.0,   max=0.5,   log_scale=True, default=0.01),
    HpoParamSpec("cont_target",      "physics",        "float",       min=0.0,   max=0.5,   log_scale=True, default=0.1),
    HpoParamSpec("mom_target",       "physics",        "float",       min=0.0,   max=0.5,   log_scale=True, default=0.1),
    HpoParamSpec("ramp_start_epoch", "physics",        "int",         min=0,     max=50,    step=5,   default=20),
    HpoParamSpec("ramp_epochs",      "physics",        "int",         min=10,    max=80,    step=10,  default=30),
    # Training
    HpoParamSpec("batch_size",       "training",       "int",         min=1,     max=2,     step=1,   default=2),
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


# ── Resource cost estimation ─────────────────────────────────────────────────

def _resource_cost(sample: dict) -> float:
    """Estimate relative GPU memory/compute cost of a trial config.

    Used to sort LHS samples cheapest-first so early trials are less likely
    to OOM and provide signal faster. Not an exact measure — just a ranking key.

    Dominant terms:
      hidden_dim   – parameter count scales as O(hidden²) per layer
      num_layers   – linear multiplier on forward/backward cost
      num_global_tokens – adds attention overhead per token
      batch_size   – multiplies memory linearly
    """
    hidden  = float(sample.get("hidden_dim",        128))
    layers  = float(sample.get("num_layers",         14))
    tokens  = float(sample.get("num_global_tokens",   0))
    batch   = float(sample.get("batch_size",           2))
    return hidden * hidden * layers * batch * (1.0 + 0.5 * tokens)


# ── Daily notebook snapshot ──────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_notebook_snapshot(state: "HpoState") -> None:
    """Append an HPO mid-run snapshot to today's daily notebook."""
    today = datetime.date.today().isoformat()
    nb_dir = _PROJECT_ROOT / "docs" / "daily_notebook"
    nb_dir.mkdir(exist_ok=True)
    nb_path = nb_dir / f"{today}.md"

    trials = state.trials
    completed = sum(1 for t in trials if t.get("state") == "complete")
    pruned    = sum(1 for t in trials if t.get("state") == "pruned")
    elapsed_h = state.elapsed_sec / 3600.0

    best_val = f"{state.best_value:.6f}" if state.best_value is not None else "—"
    best_params_lines = ""
    if state.best_params:
        best_params_lines = "\n".join(
            f"  - `{k}`: {v}" for k, v in state.best_params.items()
        )
    else:
        best_params_lines = "  *(no completed trial yet)*"

    entry = (
        f"\n## HPO Mid-Run Snapshot — {today} ({elapsed_h:.2f}h elapsed)\n\n"
        f"**Status:** {state.state} | **Phase:** {state.phase}\n\n"
        f"**Progress:** Trial {state.current_trial}/{state.total_trials} "
        f"| {completed} complete | {pruned} pruned\n\n"
        f"**Best val loss:** {best_val}\n\n"
        f"**Best params so far:**\n{best_params_lines}\n\n"
        f"---\n"
    )

    with open(nb_path, "a") as f:
        f.write(entry)


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
            low = spec.min
            # Optuna requires low > 0 for log distributions; clamp silently
            if spec.log_scale and low <= 0:
                low = 1e-6
            if spec.step > 0 and not spec.log_scale:
                params[spec.name] = trial.suggest_float(
                    spec.name, low, spec.max, step=spec.step
                )
            else:
                params[spec.name] = trial.suggest_float(
                    spec.name, low, spec.max, log=spec.log_scale
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


def _is_out_of_memory_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like a GPU or host-memory exhaustion error."""
    if isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
        return True

    msg = str(exc).lower()
    return any(token in msg for token in (
        "out of memory",
        "cuda error: out of memory",
        "cudnn_status_alloc_failed",
        "cublas_status_alloc_failed",
        "hip out of memory",
    ))


# ── State / session ──────────────────────────────────────────────────────────

@dataclass
class HpoState:
    state: str = "idle"          # idle | loading | running | stopping | stopped | completed | failed
    phase: str = "lhs"           # lhs | tpe
    current_trial: int = 0
    current_params: Optional[dict] = None
    total_trials: int = 0
    n_lhs: int = 0
    best_value: Optional[float] = None
    best_params: Optional[dict] = None
    trials: list = field(default_factory=list)
    error_message: str = ""
    elapsed_sec: float = 0.0
    current_epoch: int = 0
    epochs_per_trial: int = 0


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
        self._notebook_snapshot_taken: bool = False
        self._notebook_snapshot_sec: float = 9000.0  # 2.5 hours default

    # ── Public API ──────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_status(self) -> dict:
        with self._lock:
            trials = list(self._state.trials)
            completed_trials = sum(1 for t in trials if t.get("state") == "complete")
            pruned_trials = sum(1 for t in trials if t.get("state") == "pruned")
            running_trials = sum(1 for t in trials if t.get("state") == "running")
            finished_trials = completed_trials + pruned_trials
            progress_trials = finished_trials + running_trials
            total_trials = self._state.total_trials
            progress_pct = (100.0 * progress_trials / total_trials) if total_trials else 0.0
            d = {
                "state":         self._state.state,
                "phase":         self._state.phase,
                "current_trial": self._state.current_trial,
                "current_params": dict(self._state.current_params or {}),
                "total_trials":  total_trials,
                "n_lhs":         self._state.n_lhs,
                "best_value":    self._state.best_value,
                "best_params":   self._state.best_params,
                "trials":        trials,
                "completed_trials": completed_trials,
                "pruned_trials": pruned_trials,
                "running_trials": running_trials,
                "finished_trials": finished_trials,
                "progress_pct": progress_pct,
                "error_message":   self._state.error_message,
                "elapsed_sec":     self._state.elapsed_sec,
                "current_epoch":   self._state.current_epoch,
                "epochs_per_trial": self._state.epochs_per_trial,
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
        self._notebook_snapshot_sec = float(settings.get("notebook_snapshot_sec", 9000.0))
        self._notebook_snapshot_taken = False

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
            self._set_state(state="loading", total_trials=total_trials, n_lhs=n_lhs)
            with self._lock:
                self._state.epochs_per_trial = epochs_per_trial
                self._state.current_epoch = 0
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

            # ── Enqueue LHS samples (cheapest first) ─────────────
            lhs_samples = generate_lhs_samples(specs, n_lhs, seed=seed)
            lhs_samples.sort(key=_resource_cost)
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
                    self._state.current_epoch = 0
                    self._state.trials.append({
                        "number": trial_num,
                        "phase":  phase,
                        "state":  "running",
                        "value":  None,
                        "params": {},
                    })

                params: dict[str, Any] = {}
                bundle = None
                cfg = None
                model      = None
                optimizer  = None
                scheduler  = None
                criterion  = None
                pruned_ev  = threading.Event()
                best_val   = [float("inf")]
                patience   = [0]
                final_val: Optional[float] = None
                prune_reason: Optional[str] = None

                try:
                    params = _suggest_params(trial, specs)
                    batch_size = int(params.get("batch_size", 4))
                    active_params = {k: v for k, v in params.items() if k in enabled_names}
                    with self._lock:
                        self._state.current_params = active_params
                    bundle = self._get_bundle(batch_size, task, seed)
                    cfg = _build_cfg(params, base_cfg, epochs_per_trial)

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
                            self._state.current_epoch = epoch + 1
                            # Daily notebook snapshot after configured elapsed time
                            if (not self._notebook_snapshot_taken
                                    and self._state.elapsed_sec >= self._notebook_snapshot_sec):
                                _write_notebook_snapshot(self._state)
                                self._notebook_snapshot_taken = True

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

                except Exception as exc:
                    if not _is_out_of_memory_error(exc):
                        raise

                    pruned_ev.set()
                    prune_reason = "oom"
                    trial.set_user_attr("prune_reason", "oom")
                    trial.set_user_attr("prune_message", str(exc))

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
                            if prune_reason is not None:
                                rec["reason"] = prune_reason
                            break

                    if not pruned_ev.is_set():
                        cur_best = self._state.best_value
                        if cur_best is None or final_val < cur_best:
                            self._state.best_value  = final_val
                            self._state.best_params = dict(params)

                    self._state.elapsed_sec = time.time() - self._start_time
                    self._state.current_params = None

                if pruned_ev.is_set():
                    raise _optuna.exceptions.TrialPruned("Trial pruned by stop signal")

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
                self._state.current_params = None
