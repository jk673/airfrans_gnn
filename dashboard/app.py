#!/usr/bin/env python3
"""AirfRANS integrated training dashboard.

Usage:
    python dashboard/app.py [--port 5000] [--host 0.0.0.0]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request, render_template

from dashboard.runner import TrainingSession
from src.benchmark import (
    FLOW_GLIDE_METRIC_KEYS,
    _load_flow_glide_baselines,
    _collect_flow_glide_experiments,
)

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
session = TrainingSession()

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# -- Default config matching scripts/main.py Config dataclass --
DEFAULT_CONFIG = {
    "data": {
        "task": {"value": "scarce", "type": "select", "options": ["scarce", "full", "reynolds", "aoa"]},
        "seed": {"value": 42, "type": "int"},
        "batch_size": {"value": 16, "type": "int"},
        "num_workers": {"value": 4, "type": "int"},
    },
    "model": {
        "hidden_dim": {"value": 16, "type": "int"},
        "num_layers": {"value": 14, "type": "int"},
        "num_global_tokens": {"value": 0, "type": "int"},
        "dropout": {"value": 0.1, "type": "float"},
    },
    "physics": {
        "continuity": {
            "weight": {"value": 0.0, "type": "float"},
            "target": {"value": 0.01, "type": "float"},
            "ramp_start_epoch": {"value": 50, "type": "int"},
            "ramp_epochs": {"value": 30, "type": "int"},
        },
        "momentum": {
            "weight": {"value": 0.0, "type": "float"},
            "target": {"value": 0.05, "type": "float"},
            "ramp_start_epoch": {"value": 50, "type": "int"},
            "ramp_epochs": {"value": 30, "type": "int"},
        },
        "bc": {
            "weight": {"value": 0.01, "type": "float"},
            "target": {"value": 0.01, "type": "float"},
            "ramp_start_epoch": {"value": 50, "type": "int"},
            "ramp_epochs": {"value": 30, "type": "int"},
        },
    },
    "optimizer": {
        "lr": {"value": 1e-3, "type": "float"},
        "weight_decay": {"value": 1e-4, "type": "float"},
    },
    "scheduler": {
        "scheduler_T_max": {"value": 100, "type": "int"},
    },
    "training": {
        "num_epochs": {"value": 20, "type": "int"},
        "device": {"value": "cuda", "type": "select", "options": ["cuda", "cpu"]},
        "amp": {"value": True, "type": "bool"},
    },
    "experiment": {
        "notes": {"value": "", "type": "text"},
    },
}


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config")
def get_config():
    return jsonify(DEFAULT_CONFIG)


@app.route("/api/start", methods=["POST"])
def start_training():
    if session.is_running:
        status = session.get_status()
        return jsonify({
            "status": "error",
            "message": f"Training already running (epoch {status['current_epoch']}/{status['total_epochs']})",
        }), 409
    config = request.get_json(force=True)
    try:
        session.start(config)
        return jsonify({"status": "started", "session_id": session.get_status()["session_id"]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/stop", methods=["POST"])
def stop_training():
    if not session.is_running:
        return jsonify({"status": "error", "message": "No training running"}), 400
    session.request_stop()
    return jsonify({"status": "stopping", "message": "Will stop after current epoch."})


@app.route("/api/status")
def get_status():
    status = session.get_status()
    # Replace non-JSON-safe float values
    if status.get("best_val") == float("inf"):
        status["best_val"] = None
    return jsonify(status)


@app.route("/api/experiments")
def list_experiments():
    baselines = _load_flow_glide_baselines()
    exp_rows = _collect_flow_glide_experiments()

    experiments = []
    for exp_id, vals in exp_rows:
        # Load full JSON for metadata
        json_path = RESULTS_DIR / f"{exp_id.replace('EXP', 'EXP_')}.json"
        if not json_path.exists():
            # Try exact filename patterns
            candidates = list(RESULTS_DIR.glob(f"*{exp_id.replace('EXP', '')}*.json"))
            json_path = candidates[0] if candidates else None

        meta = {}
        if json_path and json_path.exists():
            try:
                with open(json_path) as f:
                    meta = json.load(f)
            except Exception:
                pass

        row = {
            "experiment_id": meta.get("_experiment_id", exp_id),
            "timestamp": meta.get("_timestamp", ""),
            "model_name": meta.get("_model_name", ""),
            "task": meta.get("_task", ""),
            "hidden": meta.get("_hidden"),
            "layers": meta.get("_layers"),
            "n_parameters": meta.get("_n_parameters"),
            "duration_sec": meta.get("_duration_sec"),
            "notes": meta.get("_notes", ""),
        }
        for i, key in enumerate(FLOW_GLIDE_METRIC_KEYS):
            row[key] = vals[i] if i < len(vals) else None
        experiments.append(row)

    return jsonify({"baselines": baselines, "experiments": experiments})


@app.route("/api/gpu")
def get_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            text=True, timeout=3,
        )
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue
            mem_used, mem_total = int(parts[3]), int(parts[4])
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "gpu_util": int(parts[2]) if parts[2] not in ("N/A", "") else 0,
                "mem_used": mem_used,
                "mem_total": mem_total,
                "mem_util": round(mem_used * 100 / mem_total) if mem_total > 0 else 0,
                "temp": int(parts[5]) if parts[5] not in ("N/A", "") else 0,
                "power_draw": float(parts[6]) if parts[6] not in ("N/A", "") else 0,
                "power_limit": float(parts[7]) if parts[7] not in ("N/A", "") else 0,
            })
        return jsonify({"gpus": gpus})
    except Exception as e:
        return jsonify({"gpus": [], "error": str(e)})


@app.route("/api/experiments/<exp_id>")
def get_experiment(exp_id):
    json_path = RESULTS_DIR / f"{exp_id}.json"
    if not json_path.exists():
        return jsonify({"error": "Not found"}), 404
    with open(json_path) as f:
        return jsonify(json.load(f))


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AirfRANS Training Dashboard")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Dashboard: http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
