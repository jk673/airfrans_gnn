"""Experiment tracker that logs results in FLOW-GLIDE benchmark format.

Each experiment is saved as:
  - experiments/results/<exp_id>.json  (benchmark-compatible JSON)
  - experiments/EXPERIMENT_LOG.md      (cumulative comparison table + details)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentTracker:
    """Tracks experiments with benchmark-format JSON + Markdown comparison table."""

    METRIC_KEYS = [
        "volume_rel_l2", "surface_rel_l2",
        "cd_relative_error", "cl_relative_error",
        "rho_d", "rho_l",
    ]
    METRIC_HEADERS = [
        "Volume Rel.L₂ ↓", "Surface Rel.L₂ ↓",
        "CD Rel.Err ↓", "CL Rel.Err ↓",
        "ρ_D ↑", "ρ_L ↑",
    ]

    def __init__(
        self,
        log_dir: str | Path = "experiments",
        project_name: str = "AirfRANS GNN Benchmark",
        reference_path: str | Path | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.results_dir = self.log_dir / "results"
        self.log_path = self.log_dir / "EXPERIMENT_LOG.md"
        self.project_name = project_name

        if reference_path is None:
            reference_path = Path("benchmark/benchmark_reference.json")
        self.reference = self._load_reference(reference_path)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_experiment(
        self,
        config: dict[str, Any],
        metrics: dict[str, Any] | None = None,
        model: Any | None = None,
        model_name: str = "Ours",
        notes: str = "",
        duration_sec: float | None = None,
        benchmark_metrics: dict[str, float] | None = None,
    ) -> str:
        """Log an experiment in benchmark format.

        Parameters
        ----------
        config : dict
            Training configuration (SmokeCfg fields).
        metrics : dict, optional
            Training metrics (losses, best epoch, etc.).
        model : optional
            PyTorch model (used to count parameters).
        model_name : str
            Display name for comparison table.
        notes : str
            Free-form description.
        duration_sec : float, optional
            Wall-clock duration in seconds.
        benchmark_metrics : dict, optional
            6 FLOW-GLIDE benchmark metrics from score_test_set().

        Returns
        -------
        str
            Experiment ID (e.g., "EXP_0001").
        """
        exp_id = self._next_exp_id()
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        result: dict[str, Any] = {}

        # Benchmark metrics (6 standard FLOW-GLIDE metrics)
        if benchmark_metrics:
            for k in self.METRIC_KEYS:
                result[k] = benchmark_metrics.get(k, float("nan"))

        # Metadata (underscore-prefixed, same convention as benchmark/results.json)
        result["_experiment_id"] = exp_id
        result["_timestamp"] = timestamp
        result["_model_name"] = model_name
        result["_task"] = config.get("task", "unknown")
        result["_hidden"] = config.get("hidden")
        result["_layers"] = config.get("layers")
        result["_epochs"] = config.get("epochs")
        result["_n_parameters"] = self._count_params(model)
        result["_duration_sec"] = duration_sec
        result["_notes"] = notes

        # Training metrics (prefixed with _train_)
        if metrics:
            for k, v in metrics.items():
                result[f"_train_{k}"] = _json_safe(v)

        # Full config
        result["_config"] = _serializable(config)

        # Save per-experiment JSON
        json_path = self.results_dir / f"{exp_id}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Regenerate cumulative markdown
        self._update_log()

        print(f"Experiment {exp_id} logged -> {json_path}")
        return exp_id

    # ------------------------------------------------------------------
    # Markdown generation
    # ------------------------------------------------------------------

    def _update_log(self):
        """Regenerate EXPERIMENT_LOG.md from all result JSONs."""
        experiments = self._load_all_experiments()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            f"# {self.project_name}\n",
            f"Auto-generated benchmark comparison. Updated: {now}\n",
        ]

        # Comparison table (experiments that have benchmark metrics)
        bench_exps = [
            e for e in experiments
            if any(k in e for k in self.METRIC_KEYS)
        ]
        if bench_exps:
            lines.append("## Benchmark Comparison\n")
            lines.append(self._build_comparison_table(bench_exps))
            lines.append("")

        # Individual experiment details (newest first)
        lines.append("---\n")
        lines.append("## Experiment Details\n")
        for exp in reversed(experiments):
            lines.append(self._format_detail(exp))

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _build_comparison_table(self, experiments: list[dict]) -> str:
        """Build FLOW-GLIDE-style comparison table."""
        rows: list[tuple[str, list[float]]] = []

        # Reference baselines
        baselines = self.reference.get("baselines", {})
        for name in ["Transolver", "FLOW-GLIDE"]:
            if name in baselines:
                vals = [baselines[name].get(k, float("nan")) for k in self.METRIC_KEYS]
                rows.append((name, vals))

        # Our experiments
        for exp in experiments:
            exp_id = exp.get("_experiment_id", "?")
            model = exp.get("_model_name", "Ours")
            label = f"**{exp_id}** ({model})"
            vals = [exp.get(k, float("nan")) for k in self.METRIC_KEYS]
            rows.append((label, vals))

        md = [f"| Model | {' | '.join(self.METRIC_HEADERS)} |"]
        md.append(f"|{'---|' * (len(self.METRIC_HEADERS) + 1)}")
        for name, vals in rows:
            formatted = [_fmt(v) for v in vals]
            md.append(f"| {name} | {' | '.join(formatted)} |")
        return "\n".join(md)

    def _format_detail(self, exp: dict) -> str:
        """Format one experiment's detailed section."""
        exp_id = exp.get("_experiment_id", "?")
        ts = exp.get("_timestamp", "?")
        model_name = exp.get("_model_name", "Ours")
        task = exp.get("_task", "?")
        n_params = exp.get("_n_parameters")
        duration = exp.get("_duration_sec")
        notes = exp.get("_notes", "")

        parts: list[str] = [f"### {exp_id} — {ts}\n"]

        info = [f"**Model:** {model_name}", f"**Task:** {task}"]
        if n_params:
            info.append(f"**Parameters:** {n_params:,}")
        if duration:
            info.append(f"**Duration:** {_fmt_duration(duration)}")
        parts.append(" | ".join(info) + "\n")

        if notes:
            parts.append(f"**Notes:** {notes}\n")

        # Benchmark metrics table
        bench = {k: exp[k] for k in self.METRIC_KEYS if k in exp}
        if bench:
            parts.append("| Benchmark Metric | Value |")
            parts.append("|--------|-------|")
            for k, v in bench.items():
                parts.append(f"| {k} | {_fmt(v)} |")
            parts.append("")

        # Training metrics table
        train = {
            k.replace("_train_", ""): v
            for k, v in exp.items()
            if k.startswith("_train_")
        }
        if train:
            parts.append("| Training Metric | Value |")
            parts.append("|--------|-------|")
            for k, v in train.items():
                parts.append(f"| {k} | {_fmt(v)} |")
            parts.append("")

        # Config collapsed
        config = exp.get("_config")
        if config:
            parts.append("<details><summary>Config</summary>\n")
            parts.append("```json")
            parts.append(json.dumps(config, indent=2, ensure_ascii=False))
            parts.append("```\n")
            parts.append("</details>\n")

        parts.append("---\n")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all_experiments(self) -> list[dict]:
        results = []
        for p in sorted(self.results_dir.glob("EXP_*.json")):
            with open(p) as f:
                results.append(json.load(f))
        return results

    def _next_exp_id(self) -> str:
        existing = list(self.results_dir.glob("EXP_*.json"))
        if not existing:
            return "EXP_0001"
        nums = []
        for p in existing:
            try:
                nums.append(int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
        next_num = max(nums) + 1 if nums else 1
        return f"EXP_{next_num:04d}"

    @staticmethod
    def _load_reference(path: str | Path) -> dict:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    @staticmethod
    def _count_params(model: Any) -> int | None:
        if model is None:
            return None
        try:
            return sum(p.numel() for p in model.parameters())
        except Exception:
            return None


# ------------------------------------------------------------------
# Module-level formatting helpers
# ------------------------------------------------------------------

def _fmt(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if v != v:  # NaN
            return "N/A"
        if abs(v) < 1e-3 or abs(v) >= 1e6:
            return f"{v:.4e}"
        return f"{v:.4f}"
    return str(v)


def _fmt_duration(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _json_safe(v: Any) -> Any:
    if isinstance(v, float):
        return None if v != v else v  # NaN -> None
    if isinstance(v, (int, str, bool, type(None))):
        return v
    return str(v)


def _serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)
