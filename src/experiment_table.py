"""FLOW-GLIDE comparison table generation and management."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FLOW_GLIDE_TABLE_PATH = PROJECT_ROOT / "experiments/flow_glide_comparison_table.md"
FLOW_GLIDE_METRIC_KEYS = [
    "volume_rel_l2",
    "surface_rel_l2",
    "cd_relative_error",
    "cl_relative_error",
    "rho_d",
    "rho_l",
]
FLOW_GLIDE_METRIC_HEADERS = [
    "Volume Rel.L\u2082 \u2193",
    "Surface Rel.L\u2082 \u2193",
    "CD Rel.Err \u2193",
    "CL Rel.Err \u2193",
    "\u03c1_D \u2191",
    "\u03c1_L \u2191",
]
FLOW_GLIDE_REFERENCE_PATH = PROJECT_ROOT / "benchmark/benchmark_reference.json"


def _fmt_table_value(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if v != v:
            return "N/A"
        return f"{v:.4f}"
    return str(v)


def _format_experiment_id_for_table(exp_id: str) -> str:
    if exp_id.startswith("EXP_") and len(exp_id) > 4:
        n = exp_id.split("_", 1)[1]
        if n.isdigit():
            return f"EXP{int(n):03d}"
    return exp_id


def _load_flow_glide_baselines() -> dict:
    if not FLOW_GLIDE_REFERENCE_PATH.exists():
        return {}
    try:
        with open(FLOW_GLIDE_REFERENCE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        baselines = payload.get("baselines", {})
        return {
            "Transolver": baselines.get("Transolver", {}),
            "FLOW-GLIDE": baselines.get("FLOW-GLIDE", {}),
        }
    except Exception:
        return {}


def _collect_flow_glide_experiments():
    rows = []
    results_dir = PROJECT_ROOT / "experiments" / "results"
    if not results_dir.exists():
        return rows

    for p in sorted(results_dir.glob("EXP_*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not all(
            isinstance(data.get(k), (int, float)) and not (isinstance(data.get(k), float) and np.isnan(data.get(k)))
            for k in FLOW_GLIDE_METRIC_KEYS
        ):
            continue
        exp_id = data.get("_experiment_id", p.stem)
        vals = [data.get(k, float("nan")) for k in FLOW_GLIDE_METRIC_KEYS]
        rows.append((_format_experiment_id_for_table(exp_id), vals))
    return rows


def _update_flow_glide_comparison_table() -> None:
    FLOW_GLIDE_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    header_cols = " | ".join(FLOW_GLIDE_METRIC_HEADERS)
    lines = [
        "============================================================",
        "FLOW-GLIDE Comparison Table",
        "============================================================",
        f"| Model | {header_cols} |",
        "|---|---|---|---|---|---|---|",
    ]

    baselines = _load_flow_glide_baselines()
    if "Transolver" in baselines:
        base_vals = [baselines["Transolver"].get(k, float("nan")) for k in FLOW_GLIDE_METRIC_KEYS]
        lines.append(f"| Transolver | {' | '.join(_fmt_table_value(v) for v in base_vals)} |")
    if "FLOW-GLIDE" in baselines:
        base_vals = [baselines["FLOW-GLIDE"].get(k, float("nan")) for k in FLOW_GLIDE_METRIC_KEYS]
        lines.append(f"| FLOW-GLIDE | {' | '.join(_fmt_table_value(v) for v in base_vals)} |")

    for exp_id, vals in _collect_flow_glide_experiments():
        lines.append(f"| {exp_id} | {' | '.join(_fmt_table_value(v) for v in vals)} |")

    with open(FLOW_GLIDE_TABLE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
