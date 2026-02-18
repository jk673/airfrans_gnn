"""Declarative high-level API for AirfRANS GNN training.

Wraps existing src/ modules into a concise pipeline:
  load_airfrans_data → convert_to_pyg → build_model → build_physics_loss → Trainer.fit

No logic is duplicated — every function delegates to existing implementations.
"""

from __future__ import annotations

import json
import math
import os
import random
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from src.data import (
    PREBUILT_EDGES_DIR,
    DataBundle,
    NormalizedDataset,
    StandardScaler,
    collate_pyg,
    enrich_edge_features,
)
from src.global_context_processor import EnhancedCFDModelWithGlobalContext
from src.physics_loss import NavierStokesPhysicsLoss
from src.training import get_lr, run_epoch, train_epoch
from src.utils import _prep_graph_for_norm, prep_graph, validate_edges


# ---------------------------------------------------------------------------
# Step 1: Data loading
# ---------------------------------------------------------------------------

@dataclass
class RawData:
    """Container for loaded (but not yet normalized) AirfRANS graphs."""
    train_graphs: list[Data]
    val_graphs: list[Data]
    task: str


def load_airfrans_data(
    task: str = "scarce",
    data_dir: Optional[str] = None,
    seed: int = 42,
) -> RawData:
    """Load prebuilt-edge graphs and split into train/val.

    For the *scarce* task the train directory is split 90/10;
    otherwise the existing train/test directories are used as-is.
    """
    prebuilt_root = (Path(data_dir) if data_dir else PREBUILT_EDGES_DIR) / task
    prebuilt_train_dir = prebuilt_root / "train"
    prebuilt_test_dir = prebuilt_root / "test"

    import glob as _glob

    train_files = sorted(_glob.glob(os.path.join(str(prebuilt_train_dir), "graph_*.pt")))
    test_files = sorted(_glob.glob(os.path.join(str(prebuilt_test_dir), "graph_*.pt")))
    print(f"[load_airfrans_data] found {len(train_files)} train, {len(test_files)} test under {prebuilt_root}")

    def _load(paths):
        out = []
        for p in paths:
            d = torch.load(p, map_location="cpu", weights_only=False)
            if not isinstance(d, Data):
                d = Data(**d)
            out.append(prep_graph(d))
        return out

    train_edges = _load(train_files)
    test_edges = _load(test_files)

    if train_edges:
        print(f"  Example dims -> x: {train_edges[0].x.shape}  "
              f"edge_attr: {train_edges[0].edge_attr.shape if hasattr(train_edges[0], 'edge_attr') and train_edges[0].edge_attr is not None else None}")

    validate_edges(train_edges, "train_edges")

    if task == "scarce":
        n = len(train_edges)
        n_train = int(n * 0.9)
        ids = list(range(n))
        random.Random(seed).shuffle(ids)
        train_subset = [train_edges[i] for i in ids[:n_train]]
        val_subset = [train_edges[i] for i in ids[n_train:]]
    else:
        train_subset = train_edges
        val_subset = test_edges

    return RawData(train_graphs=train_subset, val_graphs=val_subset, task=task)


def convert_to_pyg(
    data: RawData,
    batch_size: int = 2,
    num_workers: int = 0,
) -> DataBundle:
    """Normalize graphs, fit scalers, and build DataLoaders.

    Returns the existing ``DataBundle`` dataclass used throughout the codebase.
    """
    train_prepped = [enrich_edge_features(_prep_graph_for_norm(g)) for g in data.train_graphs]
    val_prepped = [enrich_edge_features(_prep_graph_for_norm(g)) for g in data.val_graphs] if data.val_graphs else []

    X_train = torch.cat([d.x for d in train_prepped if hasattr(d, "x") and d.x is not None], dim=0)
    Y_train = torch.cat([d.y for d in train_prepped if hasattr(d, "y") and d.y is not None], dim=0)

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)

    train_norm = NormalizedDataset(train_prepped, x_scaler, y_scaler)
    val_norm = NormalizedDataset(val_prepped, x_scaler, y_scaler) if val_prepped else []

    print(f"[convert_to_pyg] {len(train_norm)} train | "
          f"{len(val_norm) if isinstance(val_norm, NormalizedDataset) else 0} val")

    train_loader = DataLoader(train_norm, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_pyg)
    val_loader = (DataLoader(val_norm, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_pyg)
                  if isinstance(val_norm, NormalizedDataset) else [])

    edge_dim = (train_prepped[0].edge_attr.shape[1]
                if train_prepped and hasattr(train_prepped[0], "edge_attr") and train_prepped[0].edge_attr is not None
                else 5)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        train_graphs=data.train_graphs,
        val_graphs=data.val_graphs,
        train_norm=train_norm,
        val_norm=val_norm,
        edge_dim=edge_dim,
    )


# ---------------------------------------------------------------------------
# Step 2: Model & loss factories
# ---------------------------------------------------------------------------

def build_model(config: dict, device: str = "cuda") -> nn.Module:
    """Construct a model from a plain dict configuration.

    Example *config*::

        {
            'type': 'EnhancedCFDModelWithGlobalContext',
            'input': {
                'node_dim': 7, 'edge_dim': 10, 'hidden_dim': 128,
                'num_layers': 4, 'num_global_tokens': 16,
                'dropout': 0.1,
                'attention_heads': 2, 'attention_layers': 2,
            },
            'output': {'output_dim': 4},
        }
    """
    inp = config.get("input", {})
    out = config.get("output", {})

    # Build a SimpleNamespace that the model reads as config.xxx
    ns = types.SimpleNamespace(
        use_global_tokens=inp.get("num_global_tokens", 0) > 0,
        num_global_tokens=inp.get("num_global_tokens", 2),
        attention_heads=inp.get("attention_heads", 2),
        attention_layers=inp.get("attention_layers", 2),
        attention_dropout=inp.get("attention_dropout", 0.0),
        use_cross_attention=inp.get("use_cross_attention", True),
        global_pooling_type=inp.get("global_pooling_type", "attention"),
        positional_encoding=inp.get("positional_encoding", False),
        pos_encoding_max_len=inp.get("pos_encoding_max_len", 50000),
        use_residual_attention=inp.get("use_residual_attention", True),
        attention_normalization=inp.get("attention_normalization", "layer"),
        temperature_scaling=inp.get("temperature_scaling", False),
        attention_bias=inp.get("attention_bias", False),
        context_mode=inp.get("context_mode", "global_tokens"),
        num_gcn_layers=inp.get("num_gcn_layers", None),
        num_transformer_layers=inp.get("num_transformer_layers", None),
    )

    model = EnhancedCFDModelWithGlobalContext(
        node_feat_dim=inp.get("node_dim", 7),
        edge_feat_dim=inp.get("edge_dim", 5),
        hidden_dim=inp.get("hidden_dim", 128),
        output_dim=out.get("output_dim", 4),
        num_mp_layers=inp.get("num_layers", 14),
        dropout_p=inp.get("dropout", 0.1),
        config=ns,
    )

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    return model.to(dev)


def build_physics_loss(config: dict, steps_per_epoch: int) -> NavierStokesPhysicsLoss:
    """Build ``NavierStokesPhysicsLoss`` from a concise dict.

    Epoch→step conversion is handled automatically using *steps_per_epoch*.

    Example *config*::

        {
            'continuity': {'weight': 0.0, 'target': 0.15,
                           'ramp_start_epoch': 50, 'ramp_epochs': 30},
            'momentum':   {'weight': 0.0, 'target': 0.01,
                           'ramp_start_epoch': 50, 'ramp_epochs': 30},
            'bc':         {'weight': 0.01, 'ramp_start_epoch': 50, 'ramp_epochs': 30},
        }
    """
    cont = config.get("continuity", {})
    mom = config.get("momentum", {})
    bc = config.get("bc", {})

    return NavierStokesPhysicsLoss(
        data_loss_weight=config.get("data_weight", 1.0),
        # continuity
        continuity_loss_weight=cont.get("weight", 0.05),
        continuity_target_weight=cont.get("target", 0.20),
        cont_ramp_start_step=cont.get("ramp_start_epoch", 0) * steps_per_epoch,
        cont_curriculum_ramp_steps=cont.get("ramp_epochs", 0) * steps_per_epoch,
        # momentum
        momentum_loss_weight=mom.get("weight", 0.05),
        momentum_target_weight=mom.get("target", 0.20),
        mom_ramp_start_step=mom.get("ramp_start_epoch", 0) * steps_per_epoch,
        mom_curriculum_ramp_steps=mom.get("ramp_epochs", 0) * steps_per_epoch,
        # boundary conditions
        bc_loss_weight=bc.get("weight", 0.1),
        bc_ramp_start_step=bc.get("ramp_start_epoch", 0) * steps_per_epoch,
        bc_curriculum_ramp_steps=bc.get("ramp_epochs", 0) * steps_per_epoch,
        # global settings
        ramp_mode=config.get("ramp_mode", "linear"),
        chord_length=config.get("chord_length", 1.0),
        nu_molecular=config.get("nu_molecular", 1.5e-5),
        dynamic_uref_from_data=config.get("dynamic_uref_from_data", True),
        dynamic_re_from_data=config.get("dynamic_re_from_data", True),
        use_huber_for_physics=config.get("use_huber_for_physics", True),
        huber_delta=config.get("huber_delta", 0.05),
        use_perimeter_norm_for_div=config.get("use_perimeter_norm_for_div", True),
    )


# ---------------------------------------------------------------------------
# Step 3: Default routine functions
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, *,
                    epoch: int = 0, global_step: int = 0,
                    scaler=None, amp: bool = False, **_kw):
    """Thin wrapper around ``src.training.train_epoch``."""
    return train_epoch(
        loader, model, optimizer, device, scaler,
        amp_enabled=amp,
        loss_fn=criterion,
        global_step_start=global_step,
        desc=f"train[{epoch}]",
    )


def validate_one_epoch(model, loader, criterion, device, *,
                       amp: bool = False, **_kw):
    """Thin wrapper around ``src.training.run_epoch``."""
    return run_epoch(loader, model, device, amp_enabled=amp, loss_fn=criterion)


def save_model_checkpoint(model, optimizer, scheduler, epoch, val_loss, is_best,
                          *, checkpoint_dir: str = "checkpoints",
                          global_step: int = 0, best_val: float = float("inf"),
                          **_kw):
    """Save best.pt + periodic (every 5 epochs) checkpoints."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
    }
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, "best.pt"))
    if (epoch + 1) % 5 == 0:
        torch.save(state, os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt"))


def log_training_metrics(epoch, train_logs, val_logs, lr, *, is_best=False, **_kw):
    """Print a one-line epoch summary to the console."""
    best_tag = " [BEST]" if is_best else ""
    print(
        f"Epoch {epoch:3d} | "
        f"Train: total={train_logs.get('total_loss', float('nan')):.4f} "
        f"mse={train_logs.get('mse_loss', float('nan')):.4f} "
        f"cont={train_logs.get('continuity_loss', 0):.2e} "
        f"mom={train_logs.get('momentum_loss', 0):.2e} "
        f"bc={train_logs.get('bc_loss', 0):.2e} | "
        f"Val: total={val_logs.get('total_loss', float('nan')):.4f}"
        f"{best_tag}"
    )


# ---------------------------------------------------------------------------
# Step 4: Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates the train/eval loop with pluggable routine functions."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion,
        device: str = "cuda",
        live_plot: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        self.live_plot = live_plot

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        routine: dict[str, Callable],
    ) -> dict:
        """Run the full training loop and return a history dict."""
        scaler = GradScaler(enabled=False)
        history: dict[str, Any] = {
            "train": [],
            "val": [],
            "best_val": float("inf"),
            "best_epoch": -1,
        }
        dashboard: Optional[LiveDashboard] = None
        if self.live_plot:
            dashboard = LiveDashboard()

        global_step = 0

        for epoch in range(num_epochs):
            # --- Train ---
            train_fn = routine["train"]
            train_loss, train_logs, global_step = train_fn(
                self.model, train_loader, self.optimizer, self.criterion,
                self.device, epoch=epoch, global_step=global_step, scaler=scaler,
            )

            # --- Validate ---
            validate_fn = routine["validate"]
            val_loss, val_logs = validate_fn(
                self.model, val_loader, self.criterion, self.device,
            )

            # --- Scheduler step ---
            if self.scheduler is not None:
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # --- Track best ---
            is_best = val_loss < history["best_val"]
            if is_best:
                history["best_val"] = val_loss
                history["best_epoch"] = epoch

            lr = get_lr(self.optimizer) or 0.0

            # --- Checkpoint ---
            save_fn = routine.get("save_checkpoint")
            if save_fn is not None:
                save_fn(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_loss, is_best,
                    global_step=global_step, best_val=history["best_val"],
                )

            # --- Log metrics ---
            log_fn = routine.get("log_metrics")
            if log_fn is not None:
                log_fn(epoch, train_logs, val_logs, lr, is_best=is_best)

            # --- Live dashboard ---
            if dashboard is not None:
                dashboard.update(epoch, train_logs, val_logs, lr)

            # --- History ---
            history["train"].append(train_logs)
            history["val"].append(val_logs)

        return history


# ---------------------------------------------------------------------------
# Step 5: Live HTML dashboard
# ---------------------------------------------------------------------------

class LiveDashboard:
    """Generates a self-refreshing HTML dashboard with Chart.js plots."""

    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: dict[str, Any] = {"epochs": [], "train": {}, "val": {}, "lr": []}
        self.dashboard_path = self.output_dir / "dashboard.html"
        print(f"Live dashboard: file://{self.dashboard_path.resolve()}")

    def update(self, epoch: int, train_logs: dict, val_logs: dict, lr: float):
        self._append_metrics(epoch, train_logs, val_logs, lr)
        self._write_html()

    def _append_metrics(self, epoch: int, train_logs: dict, val_logs: dict, lr: float):
        self.metrics["epochs"].append(epoch)
        self.metrics["lr"].append(lr)
        for key in ("total_loss", "mse_loss", "continuity_loss", "momentum_loss", "bc_loss"):
            self.metrics["train"].setdefault(key, []).append(
                float(train_logs.get(key, float("nan")))
            )
            self.metrics["val"].setdefault(key, []).append(
                float(val_logs.get(key, float("nan")))
            )

    def _write_html(self):
        m = self.metrics
        last_epoch = m["epochs"][-1] if m["epochs"] else 0

        # Best val total
        val_totals = m["val"].get("total_loss", [])
        finite_vals = [v for v in val_totals if math.isfinite(v)]
        min_val = min(finite_vals) if finite_vals else float("nan")

        metrics_json = json.dumps(m)

        html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<title>AirfRANS Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; color: #eee; font-family: 'Courier New', monospace; }}
  header {{ padding: 20px; text-align: center; border-bottom: 1px solid #333; }}
  header h1 {{ font-size: 1.4em; color: #e2e2e2; }}
  .status {{ color: #4ecca3; margin-top: 6px; font-size: 0.95em; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; padding: 16px; }}
  .chart-box {{ background: #16213e; border-radius: 8px; padding: 14px; }}
  .chart-box h3 {{ font-size: 0.85em; color: #a8a8b3; margin-bottom: 8px; text-align: center; }}
  canvas {{ width: 100% !important; }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
  @media (max-width: 600px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head><body>
<header>
  <h1>AirfRANS Training Dashboard</h1>
  <p class="status">Epoch {last_epoch} | Best val: {min_val:.6f} | LR: {m['lr'][-1] if m['lr'] else 0:.2e}</p>
</header>
<div class="grid">
  <div class="chart-box"><h3>Total Loss</h3><canvas id="c_total"></canvas></div>
  <div class="chart-box"><h3>MSE Loss</h3><canvas id="c_mse"></canvas></div>
  <div class="chart-box"><h3>Continuity Loss</h3><canvas id="c_cont"></canvas></div>
  <div class="chart-box"><h3>Momentum Loss</h3><canvas id="c_mom"></canvas></div>
  <div class="chart-box"><h3>BC Loss</h3><canvas id="c_bc"></canvas></div>
  <div class="chart-box"><h3>Learning Rate</h3><canvas id="c_lr"></canvas></div>
</div>
<script>
const M = {metrics_json};
const epochs = M.epochs;

function makeChart(id, title, trainKey, valKey) {{
  const ctx = document.getElementById(id).getContext('2d');
  const datasets = [];
  if (trainKey && M.train[trainKey]) {{
    datasets.push({{
      label: 'Train', data: M.train[trainKey], borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.1)', borderWidth: 1.5, pointRadius: 0, fill: false,
    }});
  }}
  if (valKey && M.val[valKey]) {{
    datasets.push({{
      label: 'Val', data: M.val[valKey], borderColor: '#f97316',
      backgroundColor: 'rgba(249,115,22,0.1)', borderWidth: 1.5, pointRadius: 0, fill: false,
    }});
  }}
  new Chart(ctx, {{
    type: 'line',
    data: {{ labels: epochs, datasets }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ color: '#ccc', font: {{ size: 10 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#222' }} }},
        y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#222' }} }},
      }},
      animation: false,
    }},
  }});
}}

function makeLRChart() {{
  const ctx = document.getElementById('c_lr').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: epochs,
      datasets: [{{
        label: 'LR', data: M.lr, borderColor: '#a855f7',
        borderWidth: 1.5, pointRadius: 0, fill: false,
      }}],
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ color: '#ccc', font: {{ size: 10 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#222' }} }},
        y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#222' }} }},
      }},
      animation: false,
    }},
  }});
}}

makeChart('c_total', 'Total Loss', 'total_loss', 'total_loss');
makeChart('c_mse', 'MSE Loss', 'mse_loss', 'mse_loss');
makeChart('c_cont', 'Continuity', 'continuity_loss', 'continuity_loss');
makeChart('c_mom', 'Momentum', 'momentum_loss', 'momentum_loss');
makeChart('c_bc', 'BC Loss', 'bc_loss', 'bc_loss');
makeLRChart();
</script>
</body></html>"""
        self.dashboard_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Step 6: Post-training visualization helpers
# ---------------------------------------------------------------------------

def plot_training_loss(history: dict, save_dir: str = "experiments/loss_signals"):
    """Plot loss curves from a Trainer history dict (matplotlib)."""
    import matplotlib.pyplot as plt

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    keys = ["total_loss", "mse_loss", "continuity_loss", "momentum_loss", "bc_loss"]
    labels = {
        "total_loss": "Total Loss",
        "mse_loss": "MSE Loss",
        "continuity_loss": "Continuity Loss",
        "momentum_loss": "Momentum Loss",
        "bc_loss": "BC Loss",
    }

    train_logs = history.get("train", [])
    val_logs = history.get("val", [])
    if not train_logs:
        return

    epochs = list(range(1, len(train_logs) + 1))

    active_keys = []
    for k in keys:
        train_vals = [log.get(k, float("nan")) for log in train_logs]
        val_vals = [log.get(k, float("nan")) for log in val_logs]
        if any(math.isfinite(v) for v in train_vals) or any(math.isfinite(v) for v in val_vals):
            active_keys.append(k)

    if not active_keys:
        return

    n_cols = 3
    n_rows = math.ceil(len(active_keys) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.2 * n_rows), squeeze=False)

    for idx, k in enumerate(active_keys):
        ax = axes[idx // n_cols][idx % n_cols]
        train_vals = np.array([log.get(k, float("nan")) for log in train_logs])
        val_vals = np.array([log.get(k, float("nan")) for log in val_logs])
        ep = np.array(epochs)

        t_mask = np.isfinite(train_vals)
        v_mask = np.isfinite(val_vals)
        if np.any(t_mask):
            ax.plot(ep[t_mask], train_vals[t_mask], label="train", marker="o", linewidth=1.2)
        if np.any(v_mask):
            ax.plot(ep[v_mask], val_vals[v_mask], label="val", marker="x", linewidth=1.2)

        ax.set_title(labels.get(k, k))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend()

    for idx in range(len(active_keys), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.tight_layout()
    out_path = save_path / "loss_signals_epoch_vs_loss.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot_training_loss] saved to {out_path}")


def plot_model_predictions(model: nn.Module, bundle: DataBundle, device: str = "cuda"):
    """Run a single val-graph prediction and plot pred-vs-gt (channel 2 = pressure)."""
    from src.prediction import predict_one_for_viz
    from src.visualization import plot_pred_vs_gt

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    graphs = bundle.val_graphs if bundle.val_graphs else bundle.train_graphs
    if not graphs:
        print("[plot_model_predictions] no graphs available")
        return

    d_vis = graphs[0]
    dm_vis, y_pred_vis = predict_one_for_viz(
        d_vis, model, bundle.x_scaler, bundle.y_scaler, dev,
    )
    plot_pred_vs_gt(
        dm_vis, y_pred_vis,
        channel=2, show_mesh=False, denormalize=True,
        y_scaler=bundle.y_scaler, x_scaler=bundle.x_scaler,
    )
    print("[plot_model_predictions] saved to figures/")
