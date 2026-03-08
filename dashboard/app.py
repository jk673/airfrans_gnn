#!/usr/bin/env python3
"""AirfRANS integrated training dashboard.

Usage:
    python dashboard/app.py [--port 5000] [--host 0.0.0.0]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request, render_template

from dashboard.preprocessing_runner import PreprocessingSession

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
preproc_session = PreprocessingSession()

_loader_lock = threading.RLock()
_runner_module: Any | None = None
_training_session: Any | None = None
_hpo_module: Any | None = None
_hpo_session: Any | None = None
_benchmark_module: Any | None = None

RESULTS_DIR = PROJECT_ROOT / "docs" / "experiments" / "results"

# -- Default config matching scripts/train.py Config dataclass --
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
        "scheduler_type": {
            "value": "CosineAnnealingLR", "type": "select",
            "options": [
                "Constant", "CosineAnnealingLR", "StepLR",
                "ReduceLROnPlateau", "CosineAnnealingWarmRestarts",
                "MultiStepLR", "ExponentialLR",
            ],
        },
        "scheduler_step_mode": {
            "value": "epoch", "type": "select",
            "options": ["epoch"],
            "help": "When to step the scheduler. Currently only epoch-level stepping is supported.",
        },
        "scheduler_T_max": {
            "value": 100, "type": "int", "for": ["CosineAnnealingLR"],
            "min": 1, "help": "Number of epochs until LR reaches eta_min.",
        },
        "scheduler_eta_min": {
            "value": 0.0, "type": "float", "for": ["CosineAnnealingLR", "CosineAnnealingWarmRestarts"],
            "min": 0.0, "help": "Minimum learning rate.",
        },
        "scheduler_step_size": {
            "value": 10, "type": "int", "for": ["StepLR"],
            "min": 1, "help": "Decay LR every this many epochs.",
        },
        "scheduler_gamma": {
            "value": 0.1, "type": "float", "for": ["StepLR", "MultiStepLR", "ExponentialLR"],
            "min": 0.001, "max": 1.0,
            "help": "Multiplicative factor for LR decay. Must be in (0, 1].",
        },
        "scheduler_factor": {
            "value": 0.5, "type": "float", "for": ["ReduceLROnPlateau"],
            "min": 0.001, "max": 1.0,
            "help": "Factor by which LR is reduced on plateau. Must be in (0, 1].",
        },
        "scheduler_patience": {
            "value": 10, "type": "int", "for": ["ReduceLROnPlateau"],
            "min": 0, "help": "Number of epochs with no improvement before reducing LR.",
        },
        "scheduler_min_lr": {
            "value": 1e-6, "type": "float", "for": ["ReduceLROnPlateau"],
            "min": 0.0, "help": "Lower bound on LR.",
        },
        "scheduler_T_0": {
            "value": 10, "type": "int", "for": ["CosineAnnealingWarmRestarts"],
            "min": 1, "help": "Number of epochs for the first restart cycle.",
        },
        "scheduler_T_mult": {
            "value": 1, "type": "int", "for": ["CosineAnnealingWarmRestarts"],
            "min": 1, "help": "Multiplicative factor for cycle length after each restart.",
        },
        "scheduler_milestones": {
            "value": "30,60,90", "type": "text", "for": ["MultiStepLR"],
            "help": "Comma-separated epoch indices at which to decay LR (e.g. 30,60,90).",
        },
        "scheduler_warmup_steps": {
            "value": 0, "type": "int",
            "for": ["CosineAnnealingLR", "StepLR", "CosineAnnealingWarmRestarts",
                    "MultiStepLR", "ExponentialLR"],
            "min": 0,
            "help": "Number of warmup epochs (LinearLR ramp). 0 = no warmup. "
                    "Not supported with ReduceLROnPlateau.",
        },
        "scheduler_warmup_start_factor": {
            "value": 0.1, "type": "float",
            "for": ["CosineAnnealingLR", "StepLR", "CosineAnnealingWarmRestarts",
                    "MultiStepLR", "ExponentialLR"],
            "min": 0.001, "max": 1.0,
            "help": "Initial LR multiplier at warmup start (e.g. 0.1 = 10% of base LR).",
        },
        "scheduler_warmup_end_factor": {
            "value": 1.0, "type": "float",
            "for": ["CosineAnnealingLR", "StepLR", "CosineAnnealingWarmRestarts",
                    "MultiStepLR", "ExponentialLR"],
            "min": 0.001,
            "help": "LR multiplier at warmup end (1.0 = full base LR).",
        },
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


def _load_runner_module():
    global _runner_module
    if _runner_module is None:
        with _loader_lock:
            if _runner_module is None:
                from dashboard import runner as runner_module
                _runner_module = runner_module
    return _runner_module


def _get_training_session():
    global _training_session
    if _training_session is None:
        with _loader_lock:
            if _training_session is None:
                _training_session = _load_runner_module().TrainingSession()
    return _training_session


def _load_hpo_module():
    global _hpo_module
    if _hpo_module is None:
        with _loader_lock:
            if _hpo_module is None:
                from dashboard import hpo as hpo_module
                _hpo_module = hpo_module
    return _hpo_module


def _get_hpo_session():
    global _hpo_session
    if _hpo_session is None:
        with _loader_lock:
            if _hpo_session is None:
                _hpo_session = _load_hpo_module().HpoSession()
    return _hpo_session


def _load_benchmark_module():
    global _benchmark_module
    if _benchmark_module is None:
        with _loader_lock:
            if _benchmark_module is None:
                from src import benchmark as benchmark_module
                _benchmark_module = benchmark_module
    return _benchmark_module


def _default_training_status() -> dict:
    return {
        "state": "idle",
        "session_id": "",
        "current_epoch": 0,
        "total_epochs": 0,
        "best_val": None,
        "best_epoch": -1,
        "elapsed_sec": 0.0,
        "error_message": "",
        "experiment_id": "",
        "config": {},
        "metrics": {
            "epochs": [],
            "train": {},
            "val": {},
            "lr": [],
        },
    }


def _default_hpo_status() -> dict:
    return {
        "state": "idle",
        "phase": "lhs",
        "current_trial": 0,
        "current_params": {},
        "total_trials": 0,
        "n_lhs": 0,
        "best_value": None,
        "best_params": None,
        "trials": [],
        "completed_trials": 0,
        "pruned_trials": 0,
        "running_trials": 0,
        "finished_trials": 0,
        "progress_pct": 0.0,
        "error_message": "",
        "elapsed_sec": 0.0,
        "current_epoch": 0,
        "epochs_per_trial": 0,
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
    session = _get_training_session()
    if session.is_running:
        status = session.get_status()
        return jsonify({
            "status": "error",
            "message": f"Training already running (epoch {status['current_epoch']}/{status['total_epochs']})",
        }), 409
    runner = _load_runner_module()
    config = runner.flatten_scheduler_config(request.get_json(force=True))
    try:
        session.start(config)
        return jsonify({"status": "started", "session_id": session.get_status()["session_id"]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/stop", methods=["POST"])
def stop_training():
    session = _training_session
    if session is None or not session.is_running:
        return jsonify({"status": "error", "message": "No training running"}), 400
    session.request_stop()
    return jsonify({"status": "stopping", "message": "Will stop after current epoch."})


@app.route("/api/status")
def get_status():
    session = _training_session
    status = session.get_status() if session is not None else _default_training_status()
    # Replace non-JSON-safe float values
    if status.get("best_val") == float("inf"):
        status["best_val"] = None
    return jsonify(status)


@app.route("/api/experiments")
def list_experiments():
    benchmark = _load_benchmark_module()
    baselines = benchmark._load_flow_glide_baselines()
    exp_rows = benchmark._collect_flow_glide_experiments()

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
        for i, key in enumerate(benchmark.FLOW_GLIDE_METRIC_KEYS):
            row[key] = vals[i] if i < len(vals) else None
        experiments.append(row)

    return jsonify({"baselines": baselines, "experiments": experiments})


@app.route("/api/lr-preview", methods=["POST"])
def lr_preview():
    """Preview LR schedule using the same step-then-read semantics as Trainer.fit().

    Request body (JSON):
        scheduler_type   – canonical name or alias (required)
        base_lr          – initial learning rate (default 1e-3)
        num_epochs       – number of epochs to simulate (default 100)
        metric_series    – optional list of val-loss values for ReduceLROnPlateau
        scheduler_*      – per-type params (see SCHEDULER_PARAM_SPECS)

    Response (200):
        epochs    – [0, …, num_epochs-1]
        lr        – post-step LR at each epoch (matches training dashboard)
        metadata  – scheduler_type, resolved config, min_lr, max_lr,
                    step_mode, metric_mode

    Response (400):
        error – validation error message
    """
    runner = _load_runner_module()
    data = runner.flatten_scheduler_config(request.get_json(force=True))

    base_lr = float(data.get("base_lr", 1e-3))
    num_epochs = int(data.get("num_epochs", 100))
    metric_series = data.get("metric_series", None)
    if metric_series is not None:
        try:
            metric_series = [float(v) for v in metric_series]
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"metric_series must be a list of numbers: {exc}"}), 400

    try:
        result = runner.simulate_lr_schedule(data, num_epochs, base_lr, metric_series)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


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


@app.route("/api/restart", methods=["POST"])
def restart_server():
    """Restart the dashboard process in-place (os.execv)."""
    def _do_restart():
        import time as _time
        _time.sleep(0.3)  # allow response to be sent first
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_do_restart, daemon=True).start()
    return jsonify({"status": "restarting", "message": "Dashboard is restarting…"})


@app.route("/api/experiments/<exp_id>")
def get_experiment(exp_id):
    json_path = RESULTS_DIR / f"{exp_id}.json"
    if not json_path.exists():
        return jsonify({"error": "Not found"}), 404
    with open(json_path) as f:
        return jsonify(json.load(f))


# -----------------------------------------------------------------------
# HPO routes
# -----------------------------------------------------------------------

@app.route("/api/hpo/search-space")
def hpo_search_space():
    hpo = _load_hpo_module()
    return jsonify([spec.to_dict() for spec in hpo.DEFAULT_SEARCH_SPACE])


@app.route("/api/hpo/start", methods=["POST"])
def hpo_start():
    hpo_session = _get_hpo_session()
    if hpo_session.is_running:
        status = hpo_session.get_status()
        return jsonify({
            "status": "error",
            "message": f"HPO already running (trial {status['current_trial']}/{status['total_trials']})",
        }), 409
    body = request.get_json(force=True) or {}
    hpo = _load_hpo_module()
    search_space = body.get("search_space", [spec.to_dict() for spec in hpo.DEFAULT_SEARCH_SPACE])
    settings     = body.get("settings", {})
    try:
        hpo_session.start(search_space, settings)
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/hpo/stop", methods=["POST"])
def hpo_stop():
    hpo_session = _hpo_session
    if hpo_session is None or not hpo_session.is_running:
        return jsonify({"status": "error", "message": "No HPO running"}), 400
    hpo_session.request_stop()
    return jsonify({"status": "stopping", "message": "Will stop after current trial."})


@app.route("/api/hpo/status")
def hpo_status():
    hpo_session = _hpo_session
    status = hpo_session.get_status() if hpo_session is not None else _default_hpo_status()
    if status.get("best_value") == float("inf"):
        status["best_value"] = None
    return jsonify(status)


# -----------------------------------------------------------------------
# Preprocessing routes
# -----------------------------------------------------------------------

DOWNSAMPLE_CONFIG = {
    "root":             {"value": "Dataset",              "type": "text",   "help": "Root directory for the raw AirfRANS dataset"},
    "task":             {"value": "scarce",               "type": "select", "options": ["scarce", "full"]},
    "out_dir":          {"value": "Dataset/processed_data/downsampled-graphs","type": "text",   "help": "Output directory for downsampled graphs"},
    "limit_train":      {"value": None,                   "type": "int",    "help": "Limit training samples processed (debug; leave blank for all)"},
    "limit_test":       {"value": None,                   "type": "int",    "help": "Limit test samples processed (debug; leave blank for all)"},
    "target_min_nodes": {"value": 15000,                  "type": "int",    "help": "Minimum acceptable node count after downsampling"},
    "target_max_nodes": {"value": 30000,                  "type": "int",    "help": "Maximum acceptable node count after downsampling"},
    "voxel_frac":       {"value": 0.01,                   "type": "float",  "help": "Initial voxel size as a fraction of chord length"},
    "voxel_iters":      {"value": 5,                      "type": "int",    "help": "Number of iterations for adaptive voxel size search"},
    "voxel_rep":        {"value": "gradient",             "type": "select", "options": ["gradient", "centroid", "first"],
                         "help": "Voxel representative selection policy"},
}

EDGE_CONFIG = {
    "in_dir":                  {"value": "Dataset/processed_data/downsampled-graphs", "type": "text",  "help": "Input directory with downsampled graphs"},
    "out_dir":                 {"value": "Dataset/processed_data/prebuilt_edges",     "type": "text",  "help": "Output directory for graphs with edges"},
    "task":                    {"value": "scarce",   "type": "select", "options": ["scarce", "full"]},
    "global_radius":           {"value": 0.02,       "type": "float",  "help": "Radius for global (volume) edge construction"},
    "surface_radius":          {"value": 0.01,       "type": "float",  "help": "Radius for surface-specific edge construction"},
    "max_num_neighbors":       {"value": 48,         "type": "int",    "help": "Maximum neighbors per node in radius graph"},
    "surface_ring":            {"value": False,      "type": "bool",   "help": "Enable surface ring connectivity"},
    "denormalize":             {"value": False,      "type": "bool",   "help": "Denormalize features before edge construction"},
    "min_degree":              {"value": 2,          "type": "int",    "help": "Minimum degree threshold for QA checks"},
    "knn_backup_k":            {"value": 4,          "type": "int",    "help": "Number of KNN neighbors for backup edges on isolated nodes"},
    "knn_max_radius":          {"value": 0.05,       "type": "float",  "help": "Maximum radius for KNN backup edges"},
    "max_isolated_fraction":   {"value": 0.01,       "type": "float",  "help": "QA warn threshold: fraction of isolated nodes"},
    "max_low_degree_fraction": {"value": 0.05,       "type": "float",  "help": "QA warn threshold: fraction of low-degree nodes"},
    "qa_fail_fast":            {"value": False,      "type": "bool",   "help": "Raise error instead of warning on QA threshold violation"},
}


@app.route("/api/preproc/config")
def preproc_config():
    return jsonify({"downsample": DOWNSAMPLE_CONFIG, "edges": EDGE_CONFIG})


@app.route("/api/preproc/start/downsample", methods=["POST"])
def preproc_start_downsample():
    if preproc_session.is_running:
        return jsonify({"status": "error", "message": "A preprocessing step is already running."}), 409
    config = request.get_json(force=True) or {}
    try:
        preproc_session.start_downsample(config)
        return jsonify({"status": "started", "step": "downsample"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/preproc/start/edges", methods=["POST"])
def preproc_start_edges():
    if preproc_session.is_running:
        return jsonify({"status": "error", "message": "A preprocessing step is already running."}), 409
    config = request.get_json(force=True) or {}
    try:
        preproc_session.start_edges(config)
        return jsonify({"status": "started", "step": "edges"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/preproc/stop", methods=["POST"])
def preproc_stop():
    if not preproc_session.is_running:
        return jsonify({"status": "error", "message": "No preprocessing running."}), 400
    preproc_session.request_stop()
    return jsonify({"status": "stopping", "message": "Stop signal sent."})


@app.route("/api/preproc/status")
def preproc_status():
    return jsonify(preproc_session.get_status())


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
