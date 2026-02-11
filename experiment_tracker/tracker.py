"""
Experiment Tracker — Auto-logs experiments, model changes, and preprocessing to Markdown.

Usage:
    from experiment_tracker import ExperimentTracker, track_experiment, track_preprocessing

    tracker = ExperimentTracker("experiments/EXPERIMENT_LOG.md")

    @track_experiment(tracker)
    def train(config):
        model = MyModel(config)
        ...
        return {"loss": 0.01, "mae": 0.005}  # return metrics dict

    @track_preprocessing(tracker)
    def preprocess(raw_data, config):
        ...
        return processed_data
"""

import functools
import hashlib
import inspect
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


class ExperimentTracker:
    """Tracks experiments, model architectures, and preprocessing changes to a Markdown file."""

    def __init__(
        self,
        log_path: str = "EXPERIMENT_LOG.md",
        project_name: str = "AirfRANS 2D Airfoil",
        auto_git_info: bool = True,
    ):
        self.log_path = Path(log_path)
        self.project_name = project_name
        self.auto_git_info = auto_git_info
        self._prev_model_hash: Optional[str] = None
        self._prev_preprocess_hash: Optional[str] = None
        self._experiment_count = self._count_existing_experiments()

        # Initialize the log file if it doesn't exist
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_header()

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def log_experiment(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        model: Any = None,
        notes: str = "",
        duration_sec: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Log a complete experiment run."""
        self._experiment_count += 1
        exp_id = f"EXP-{self._experiment_count:04d}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sections: List[str] = []
        sections.append(f"\n---\n\n## {exp_id} | {timestamp}\n")

        # Git info
        if self.auto_git_info:
            git_info = self._get_git_info()
            if git_info:
                sections.append(f"**Branch:** `{git_info['branch']}` | **Commit:** `{git_info['commit_short']}` {git_info['commit_msg']}\n")

        # Duration
        if duration_sec is not None:
            sections.append(f"**Duration:** {self._format_duration(duration_sec)}\n")

        # Hyperparameters
        sections.append("\n### Hyperparameters\n")
        sections.append("| Parameter | Value |")
        sections.append("|-----------|-------|")
        for k, v in sorted(config.items()):
            sections.append(f"| `{k}` | `{self._format_value(v)}` |")
        sections.append("")

        # Model architecture (if provided)
        if model is not None:
            arch_section, arch_changed = self._log_model_architecture(model)
            if arch_section:
                sections.append(arch_section)

        # Metrics
        sections.append("\n### Results\n")
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
        for k, v in sorted(metrics.items()):
            sections.append(f"| `{k}` | `{self._format_value(v)}` |")
        sections.append("")

        # Extra info
        if extra:
            sections.append("\n### Additional Info\n")
            for k, v in extra.items():
                sections.append(f"- **{k}:** {self._format_value(v)}")
            sections.append("")

        # Notes
        if notes:
            sections.append(f"\n### Notes\n\n{notes}\n")

        self._append_to_log("\n".join(sections))
        print(f"📝 Experiment {exp_id} logged to {self.log_path}")
        return exp_id

    def log_preprocessing(
        self,
        config: Dict[str, Any],
        input_shape: Any = None,
        output_shape: Any = None,
        notes: str = "",
        duration_sec: Optional[float] = None,
    ):
        """Log a preprocessing step/change."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config_hash = self._hash_dict(config)
        changed = config_hash != self._prev_preprocess_hash
        self._prev_preprocess_hash = config_hash

        sections: List[str] = []
        status = "🔄 CHANGED" if changed and self._experiment_count > 0 else "📋 Recorded"
        sections.append(f"\n---\n\n## [Preprocessing] {status} | {timestamp}\n")

        if duration_sec is not None:
            sections.append(f"**Duration:** {self._format_duration(duration_sec)}\n")

        sections.append("\n### Preprocessing Config\n")
        sections.append("| Parameter | Value |")
        sections.append("|-----------|-------|")
        for k, v in sorted(config.items()):
            sections.append(f"| `{k}` | `{self._format_value(v)}` |")
        sections.append("")

        if input_shape is not None or output_shape is not None:
            sections.append("\n### Data Shape\n")
            if input_shape is not None:
                sections.append(f"- **Input:** `{input_shape}`")
            if output_shape is not None:
                sections.append(f"- **Output:** `{output_shape}`")
            sections.append("")

        if notes:
            sections.append(f"\n### Notes\n\n{notes}\n")

        self._append_to_log("\n".join(sections))
        print(f"📝 Preprocessing step logged to {self.log_path}")

    def log_model_change(self, model: Any, description: str = ""):
        """Manually log a model architecture change."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        arch_section, _ = self._log_model_architecture(model, force=True)

        sections: List[str] = []
        sections.append(f"\n---\n\n## [Model Change] | {timestamp}\n")
        if description:
            sections.append(f"{description}\n")
        if arch_section:
            sections.append(arch_section)

        self._append_to_log("\n".join(sections))
        print(f"📝 Model change logged to {self.log_path}")

    def add_note(self, note: str, tag: str = "Note"):
        """Add a freeform note to the log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"\n---\n\n## [{tag}] | {timestamp}\n\n{note}\n"
        self._append_to_log(entry)

    # ─────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────

    def _write_header(self):
        header = f"""# 🧪 Experiment Log — {self.project_name}

> Auto-generated by `ExperimentTracker`
> Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Legend:** Each entry records hyperparameters, metrics, model architecture, and preprocessing changes automatically.

| Symbol | Meaning |
|--------|---------|
| 🔄 | Configuration changed from previous run |
| 📋 | Recorded (first run or no change) |
| ⚠️ | Model architecture changed |

"""
        self.log_path.write_text(header, encoding="utf-8")

    def _append_to_log(self, content: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(content)

    def _count_existing_experiments(self) -> int:
        if not self.log_path.exists():
            return 0
        text = self.log_path.read_text(encoding="utf-8")
        return text.count("## EXP-")

    def _log_model_architecture(self, model: Any, force: bool = False) -> tuple:
        """Extract and log model architecture. Returns (section_str, changed_bool)."""
        arch_str = self._extract_architecture(model)
        arch_hash = hashlib.md5(arch_str.encode()).hexdigest()[:8]
        changed = arch_hash != self._prev_model_hash
        self._prev_model_hash = arch_hash

        if not changed and not force:
            return f"\n### Model Architecture\n\nNo changes (hash: `{arch_hash}`)\n", False

        status = "⚠️ CHANGED" if changed and not force else ""
        param_count = self._count_parameters(model)

        section_parts = [f"\n### Model Architecture {status}\n"]
        section_parts.append(f"**Hash:** `{arch_hash}` | **Parameters:** {param_count}\n")
        section_parts.append(f"```\n{arch_str}\n```\n")

        return "\n".join(section_parts), changed

    def _extract_architecture(self, model: Any) -> str:
        """Extract model architecture string. Supports PyTorch, TensorFlow, and generic."""
        # PyTorch
        if hasattr(model, "named_modules"):
            return str(model)
        # TensorFlow/Keras
        if hasattr(model, "summary"):
            import io
            buf = io.StringIO()
            try:
                model.summary(print_fn=lambda x: buf.write(x + "\n"))
                return buf.getvalue()
            except Exception:
                pass
        # Generic fallback
        return repr(model)

    def _count_parameters(self, model: Any) -> str:
        """Count model parameters."""
        try:
            # PyTorch
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return f"{total:,} total ({trainable:,} trainable)"
        except (AttributeError, TypeError):
            pass
        try:
            # TensorFlow/Keras
            total = model.count_params()
            return f"{total:,} total"
        except (AttributeError, TypeError):
            pass
        return "N/A"

    def _get_git_info(self) -> Optional[Dict[str, str]]:
        """Get current git branch and commit info."""
        try:
            import subprocess
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            commit_msg = subprocess.check_output(
                ["git", "log", "-1", "--pretty=%s"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            return {
                "branch": branch,
                "commit": commit,
                "commit_short": commit[:7],
                "commit_msg": commit_msg,
            }
        except Exception:
            return None

    def _hash_dict(self, d: Dict) -> str:
        return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]

    @staticmethod
    def _format_value(v: Any) -> str:
        if isinstance(v, float):
            if abs(v) < 0.001 or abs(v) > 10000:
                return f"{v:.4e}"
            return f"{v:.6f}"
        if isinstance(v, (list, tuple)) and len(v) > 5:
            return f"[{v[0]}, {v[1]}, ... {v[-1]}] (len={len(v)})"
        return str(v)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}min"
        return f"{seconds / 3600:.1f}h"


# ─────────────────────────────────────────────────
# Decorators
# ─────────────────────────────────────────────────

def track_experiment(
    tracker: ExperimentTracker,
    config_param: str = "config",
    model_param: str = "model",
    notes: str = "",
):
    """
    Decorator that auto-logs experiment runs.

    The decorated function should:
      - Accept a config dict (param name configurable via config_param)
      - Optionally accept a model (param name configurable via model_param)
      - Return a dict of metrics (e.g., {"loss": 0.01, "mae": 0.003})

    Usage:
        @track_experiment(tracker, config_param="config")
        def train(config, model=None):
            ...
            return {"loss": final_loss, "val_mae": val_mae}
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract config from args/kwargs
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            config = bound.arguments.get(config_param, {})
            model = bound.arguments.get(model_param, None)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                duration = time.time() - start_time
                tracker.log_experiment(
                    config=config if isinstance(config, dict) else {"config": str(config)},
                    metrics={"status": "FAILED", "error": str(e)},
                    model=model,
                    notes=f"❌ Exception: {type(e).__name__}: {e}",
                    duration_sec=duration,
                )
                raise

            duration = time.time() - start_time

            # Parse metrics from result
            metrics = {}
            if isinstance(result, dict):
                metrics = result
            elif isinstance(result, (int, float)):
                metrics = {"loss": result}
            elif isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                # Support returning (model_or_output, metrics_dict)
                metrics = result[1]
                result = result[0]

            config_dict = config if isinstance(config, dict) else {"config": str(config)}
            tracker.log_experiment(
                config=config_dict,
                metrics=metrics,
                model=model,
                notes=notes,
                duration_sec=duration,
            )
            return result

        return wrapper
    return decorator


def track_preprocessing(
    tracker: ExperimentTracker,
    config_param: str = "config",
    notes: str = "",
):
    """
    Decorator that auto-logs preprocessing steps.

    The decorated function should accept a config dict.
    Optionally return a tuple of (result, {"input_shape": ..., "output_shape": ...}).

    Usage:
        @track_preprocessing(tracker)
        def preprocess(data, config):
            ...
            return processed_data
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            config = bound.arguments.get(config_param, {})

            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            input_shape = None
            output_shape = None

            # Try to extract shapes
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                meta = result[1]
                input_shape = meta.get("input_shape")
                output_shape = meta.get("output_shape")
                result = result[0]
            else:
                # Auto-detect shape from common data types
                output_shape = _try_get_shape(result)

            config_dict = config if isinstance(config, dict) else {"config": str(config)}
            tracker.log_preprocessing(
                config=config_dict,
                input_shape=input_shape,
                output_shape=output_shape,
                notes=notes,
                duration_sec=duration,
            )
            return result

        return wrapper
    return decorator


def _try_get_shape(obj: Any) -> Optional[str]:
    """Try to get shape info from common data types."""
    if hasattr(obj, "shape"):
        return str(obj.shape)
    if isinstance(obj, (list, tuple)):
        return f"len={len(obj)}"
    if isinstance(obj, dict):
        return f"keys={list(obj.keys())}"
    return None