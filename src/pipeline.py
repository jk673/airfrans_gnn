"""Declarative high-level API for AirfRANS GNN training.

Pipeline: load_airfrans_data -> convert_to_pyg -> build_model -> build_physics_loss -> Trainer.fit
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
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from src.data import (
    PREBUILT_EDGES_DIR,
    DataBundle,
    NormalizedDataset,
    StandardScaler,
    collate_pyg,
    enrich_edge_features,
    ensure_edge_features,
    with_pos2,
    _prep_graph_for_norm,
    prep_graph,
    validate_edges,
)
from src.model import EnhancedCFDModelWithGlobalContext
from src.physics_loss import NavierStokesPhysicsLoss
from src.training import get_lr, run_epoch, train_epoch


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
    """Load prebuilt-edge graphs and split into train/val."""
    prebuilt_root = (Path(data_dir) if data_dir else PREBUILT_EDGES_DIR) / task
    import glob as _glob

    train_files = sorted(_glob.glob(os.path.join(str(prebuilt_root / "train"), "graph_*.pt")))
    test_files = sorted(_glob.glob(os.path.join(str(prebuilt_root / "test"), "graph_*.pt")))
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
    """Normalize graphs, fit scalers, and build DataLoaders."""
    train_prepped = [enrich_edge_features(_prep_graph_for_norm(g)) for g in data.train_graphs]
    val_prepped = [enrich_edge_features(_prep_graph_for_norm(g)) for g in data.val_graphs] if data.val_graphs else []

    x_list = [d.x for d in train_prepped if hasattr(d, "x") and d.x is not None]
    y_list = [d.y for d in train_prepped if hasattr(d, "y") and d.y is not None]
    if not x_list:
        raise RuntimeError(
            f"No training graphs have node features (x). "
            f"Found {len(data.train_graphs)} graphs under task='{data.task}'. "
            "Check that prebuilt_edges_v2/ exists for this task."
        )
    if not y_list:
        raise RuntimeError(
            f"No training graphs have target labels (y). "
            f"Found {len(data.train_graphs)} graphs under task='{data.task}'. "
            "Check that the prebuilt graphs include ground-truth labels."
        )
    X_train = torch.cat(x_list, dim=0)
    Y_train = torch.cat(y_list, dim=0)

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)

    train_norm = NormalizedDataset(train_prepped, x_scaler, y_scaler)
    val_norm = NormalizedDataset(val_prepped, x_scaler, y_scaler) if val_prepped else []

    print(f"[convert_to_pyg] {len(train_norm)} train | "
          f"{len(val_norm) if isinstance(val_norm, NormalizedDataset) else 0} val")

    loader_kwargs: dict[str, Any] = dict(
        num_workers=num_workers,
        collate_fn=collate_pyg,
        pin_memory=(num_workers > 0 and torch.cuda.is_available()),
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_norm, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = (DataLoader(val_norm, batch_size=batch_size, shuffle=False, **loader_kwargs)
                  if isinstance(val_norm, NormalizedDataset) else [])

    edge_dim = (train_prepped[0].edge_attr.shape[1]
                if train_prepped and hasattr(train_prepped[0], "edge_attr") and train_prepped[0].edge_attr is not None
                else 5)

    return DataBundle(
        train_loader=train_loader, val_loader=val_loader,
        x_scaler=x_scaler, y_scaler=y_scaler,
        train_graphs=data.train_graphs, val_graphs=data.val_graphs,
        train_norm=train_norm, val_norm=val_norm, edge_dim=edge_dim,
    )


# ---------------------------------------------------------------------------
# Step 2: Model & loss factories
# ---------------------------------------------------------------------------

def build_model(config: dict, device: str = "cuda") -> nn.Module:
    """Construct a model from a plain dict configuration."""
    inp = config.get("input", {})
    out = config.get("output", {})

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
    """Build NavierStokesPhysicsLoss from a concise dict."""
    cont = config.get("continuity", {})
    mom = config.get("momentum", {})
    bc = config.get("bc", {})

    return NavierStokesPhysicsLoss(
        data_loss_weight=config.get("data_weight", 1.0),
        continuity_loss_weight=cont.get("weight", 0.05),
        continuity_target_weight=cont.get("target", 0.20),
        cont_ramp_start_step=cont.get("ramp_start_epoch", 0) * steps_per_epoch,
        cont_curriculum_ramp_steps=cont.get("ramp_epochs", 0) * steps_per_epoch,
        momentum_loss_weight=mom.get("weight", 0.05),
        momentum_target_weight=mom.get("target", 0.20),
        mom_ramp_start_step=mom.get("ramp_start_epoch", 0) * steps_per_epoch,
        mom_curriculum_ramp_steps=mom.get("ramp_epochs", 0) * steps_per_epoch,
        bc_loss_weight=bc.get("weight", 0.1),
        bc_target_weight=bc.get("target", None),
        bc_ramp_start_step=bc.get("ramp_start_epoch", 0) * steps_per_epoch,
        bc_curriculum_ramp_steps=bc.get("ramp_epochs", 0) * steps_per_epoch,
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
                    scaler=None, amp: bool = False, stop_event=None, **_kw):
    return train_epoch(
        loader, model, optimizer, device, scaler,
        amp_enabled=amp, loss_fn=criterion,
        global_step_start=global_step, desc=f"train[{epoch}]",
        stop_event=stop_event,
    )


def validate_one_epoch(model, loader, criterion, device, *, amp: bool = False, **_kw):
    return run_epoch(loader, model, device, amp_enabled=amp, loss_fn=criterion)


def save_model_checkpoint(model, optimizer, scheduler, epoch, val_loss, is_best,
                          *, checkpoint_dir: str = "checkpoints",
                          global_step: int = 0, best_val: float = float("inf"), **_kw):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch, "global_step": global_step, "best_val": best_val,
    }
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, "best.pt"))
    if (epoch + 1) % 5 == 0:
        torch.save(state, os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt"))


def log_training_metrics(epoch, train_logs, val_logs, lr, *, is_best=False, **_kw):
    best_tag = " [BEST]" if is_best else ""
    print(f"Epoch {epoch:3d} | "
          f"Train: total={train_logs.get('total_loss', float('nan')):.4f} "
          f"mse={train_logs.get('mse_loss', float('nan')):.4f} "
          f"cont={train_logs.get('continuity_loss', 0):.2e} "
          f"mom={train_logs.get('momentum_loss', 0):.2e} "
          f"bc={train_logs.get('bc_loss', 0):.2e} | "
          f"Val: total={val_logs.get('total_loss', float('nan')):.4f}{best_tag}")


# ---------------------------------------------------------------------------
# Step 4: Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates the train/eval loop with pluggable routine functions."""

    def __init__(self, model, optimizer, scheduler, criterion,
                 device="cuda", live_plot=False, dashboard_refresh_every=1, amp=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.live_plot = live_plot
        self.dashboard_refresh_every = dashboard_refresh_every
        self.amp = amp

    def fit(self, train_loader, val_loader, num_epochs: int, routine: dict[str, Callable],
            on_epoch_end: Callable | None = None, stop_event=None) -> dict:
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        scaler = GradScaler("cuda", enabled=False)
        history: dict[str, Any] = {"train": [], "val": [], "best_val": float("inf"), "best_epoch": -1}
        dashboard = LiveDashboard(refresh_every=self.dashboard_refresh_every) if self.live_plot else None
        global_step = 0

        for epoch in range(num_epochs):
            train_loss, train_logs, global_step = routine["train"](
                self.model, train_loader, self.optimizer, self.criterion,
                self.device, epoch=epoch, global_step=global_step, scaler=scaler, amp=self.amp,
                stop_event=stop_event,
            )
            val_loss, val_logs = routine["validate"](
                self.model, val_loader, self.criterion, self.device, amp=self.amp,
            )

            if self.scheduler is not None:
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            is_best = val_loss < history["best_val"]
            if is_best:
                history["best_val"] = val_loss
                history["best_epoch"] = epoch

            lr = get_lr(self.optimizer) or 0.0

            save_fn = routine.get("save_checkpoint")
            if save_fn is not None:
                save_fn(self.model, self.optimizer, self.scheduler, epoch, val_loss, is_best,
                        global_step=global_step, best_val=history["best_val"])

            log_fn = routine.get("log_metrics")
            if log_fn is not None:
                log_fn(epoch, train_logs, val_logs, lr, is_best=is_best)

            if dashboard is not None:
                dashboard.update(epoch, train_logs, val_logs, lr, is_last=(epoch == num_epochs - 1))

            if on_epoch_end is not None:
                should_stop = on_epoch_end(epoch=epoch, train_logs=train_logs,
                                           val_logs=val_logs, lr=lr, is_best=is_best)
                if should_stop:
                    history["train"].append(train_logs)
                    history["val"].append(val_logs)
                    break

            history["train"].append(train_logs)
            history["val"].append(val_logs)

        return history


# ---------------------------------------------------------------------------
# Step 5: Live HTML dashboard
# ---------------------------------------------------------------------------

class LiveDashboard:
    """Generates a self-refreshing HTML dashboard with Chart.js plots."""

    def __init__(self, output_dir="experiments", refresh_every=1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: dict[str, Any] = {"epochs": [], "train": {}, "val": {}, "lr": []}
        self.dashboard_path = self.output_dir / "dashboard.html"
        self.refresh_every = max(1, refresh_every)
        self._browser_opened = False
        print(f"Live dashboard: file://{self.dashboard_path.resolve()}")

    def update(self, epoch, train_logs, val_logs, lr, is_last=False):
        self.metrics["epochs"].append(epoch)
        self.metrics["lr"].append(lr)
        for key in ("total_loss", "mse_loss", "continuity_loss", "momentum_loss", "bc_loss"):
            self.metrics["train"].setdefault(key, []).append(float(train_logs.get(key, float("nan"))))
            self.metrics["val"].setdefault(key, []).append(float(val_logs.get(key, float("nan"))))
        if epoch == 0 or (epoch + 1) % self.refresh_every == 0 or is_last:
            self._write_html()
            self._open_browser()

    def _open_browser(self):
        if self._browser_opened:
            return
        self._browser_opened = True
        import subprocess, shutil
        abs_path = str(self.dashboard_path.resolve())
        if shutil.which("wslpath"):
            try:
                win_path = subprocess.check_output(["wslpath", "-w", abs_path], text=True).strip()
                subprocess.Popen(["cmd.exe", "/c", "start", "", win_path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                pass
        import webbrowser
        webbrowser.open(f"file://{abs_path}")

    def _write_html(self):
        m = self.metrics
        last_epoch = m["epochs"][-1] if m["epochs"] else 0
        val_totals = m["val"].get("total_loss", [])
        finite_vals = [v for v in val_totals if math.isfinite(v)]
        min_val = min(finite_vals) if finite_vals else float("nan")

        metrics_json = json.dumps(m)
        html = _DASHBOARD_HTML.format(
            last_epoch=last_epoch, min_val=f"{min_val:.6f}",
            lr=f"{m['lr'][-1] if m['lr'] else 0:.2e}", metrics_json=metrics_json,
        )
        self.dashboard_path.write_text(html, encoding="utf-8")


_DASHBOARD_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta http-equiv="refresh" content="10">
<title>AirfRANS Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#1a1a2e; color:#eee; font-family:'Courier New',monospace; }}
  header {{ padding:20px; text-align:center; border-bottom:1px solid #333; }}
  .status {{ color:#4ecca3; margin-top:6px; }}
  .hint {{ color:#666; margin-top:4px; font-size:0.8em; }}
  .grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; padding:16px; }}
  .cb {{ background:#16213e; border-radius:8px; padding:14px; }}
  .cb h3 {{ font-size:0.85em; color:#a8a8b3; margin-bottom:8px; text-align:center; }}
</style></head><body>
<header><h1>AirfRANS Training Dashboard</h1>
<p class="status">Epoch {last_epoch} | Best val: {min_val} | LR: {lr}</p>
<p class="hint">Scroll to zoom | Drag to pan | Double-click to reset</p></header>
<div class="grid">
  <div class="cb"><h3>Total Loss</h3><canvas id="c0"></canvas></div>
  <div class="cb"><h3>MSE Loss</h3><canvas id="c1"></canvas></div>
  <div class="cb"><h3>Continuity</h3><canvas id="c2"></canvas></div>
  <div class="cb"><h3>Momentum</h3><canvas id="c3"></canvas></div>
  <div class="cb"><h3>BC Loss</h3><canvas id="c4"></canvas></div>
  <div class="cb"><h3>Learning Rate</h3><canvas id="c5"></canvas></div>
</div>
<script>
const M={metrics_json}, E=M.epochs;
const zoomOpts={{pan:{{enabled:true,mode:'xy'}},zoom:{{wheel:{{enabled:true}},pinch:{{enabled:true}},mode:'xy'}}}};
const axOpts={{x:{{ticks:{{color:'#888'}},grid:{{color:'#222'}}}},y:{{ticks:{{color:'#888'}},grid:{{color:'#222'}}}}}};
function mk(id,tk,vk){{
  const ds=[];
  if(tk&&M.train[tk])ds.push({{label:'Train',data:M.train[tk],borderColor:'#3b82f6',borderWidth:1.5,pointRadius:0,fill:false}});
  if(vk&&M.val[vk])ds.push({{label:'Val',data:M.val[vk],borderColor:'#f97316',borderWidth:1.5,pointRadius:0,fill:false}});
  const c=new Chart(document.getElementById(id),{{type:'line',data:{{labels:E,datasets:ds}},
    options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#ccc'}}}},zoom:zoomOpts}},scales:axOpts,animation:false}}}});
  document.getElementById(id).addEventListener('dblclick',()=>c.resetZoom());
}}
mk('c0','total_loss','total_loss');mk('c1','mse_loss','mse_loss');
mk('c2','continuity_loss','continuity_loss');mk('c3','momentum_loss','momentum_loss');
mk('c4','bc_loss','bc_loss');
(function(){{const c=new Chart(document.getElementById('c5'),{{type:'line',
  data:{{labels:E,datasets:[{{label:'LR',data:M.lr,borderColor:'#a855f7',borderWidth:1.5,pointRadius:0,fill:false}}]}},
  options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#ccc'}}}},zoom:zoomOpts}},scales:axOpts,animation:false}}}});
  document.getElementById('c5').addEventListener('dblclick',()=>c.resetZoom());}})();
</script></body></html>"""


# ---------------------------------------------------------------------------
# Step 6: Post-training visualization (merged from prediction.py + visualization.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_one_for_viz(d, model, x_scaler, y_scaler, device):
    """Single-sample prediction for visualization (float32 enforced)."""
    dm = Data(**{k: v for k, v in d})
    if dm.x.size(1) == 5 or not getattr(dm, "pos2_appended", False):
        dm = with_pos2(dm)
    assert hasattr(dm, "edge_index") and dm.edge_index is not None
    dm = ensure_edge_features(dm, want_dim=5)
    dm = enrich_edge_features(dm)

    dm_norm = Data(**{k: v for k, v in dm.items()})
    dm_norm.x = x_scaler.transform(dm.x)
    dm_norm.y = y_scaler.transform(dm.y)

    dm_run = dm_norm.to(device)
    model.eval()
    if hasattr(dm_run, "x"):
        dm_run.x = dm_run.x.float()
    if hasattr(dm_run, "edge_attr"):
        dm_run.edge_attr = dm_run.edge_attr.float()
    y_pred_norm = model(dm_run).detach().cpu()
    return dm_norm, y_pred_norm


@torch.no_grad()
def plot_pred_vs_gt(dm, y_pred, channel=2, show_mesh=True, mask_airfoil=True,
                    cmap='viridis', denormalize=True, save_dir="figures",
                    save=True, y_scaler=None, x_scaler=None):
    """Visualize prediction vs ground truth using tricontour plots."""
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from matplotlib.path import Path as MplPath

    if denormalize and hasattr(dm, 'y') and y_scaler is not None:
        gt = y_scaler.inverse(dm.y.detach().cpu())
        pr = y_scaler.inverse(y_pred.detach().cpu())
    else:
        gt = dm.y.detach().cpu()
        pr = y_pred.detach().cpu()

    xy = ((dm.pos if hasattr(dm, 'pos') and dm.pos is not None else dm.x)[:, :2]
          .detach().cpu().float().numpy())
    tri = Triangulation(xy[:, 0], xy[:, 1])

    dm_x_cpu = dm.x.detach().cpu()
    x_phys = x_scaler.inverse(dm_x_cpu) if x_scaler is not None else dm_x_cpu
    vx, vy = float(x_phys[0, 0]), float(x_phys[0, 1])
    q = 0.5 * (vx * vx + vy * vy)

    gt_c = gt[:, channel].float().numpy() / q
    pr_c = pr[:, channel].float().numpy() / q
    err = np.abs(pr_c - gt_c)
    vmin, vmax = -2.0, 1.0

    if mask_airfoil:
        try:
            x_np = x_phys.numpy() if isinstance(x_phys, torch.Tensor) else np.asarray(x_phys)
            surf_mask = None
            if x_np.shape[1] >= 5:
                surf_mask = (np.abs(x_np[:, 3:5]).sum(axis=1) > 1e-8)
                if x_np.shape[1] >= 3:
                    surf_mask = np.logical_or(surf_mask, x_np[:, 2] < 1e-6)
            elif x_np.shape[1] >= 3:
                surf_mask = x_np[:, 2] < 1e-6
            if surf_mask is not None and np.any(surf_mask):
                pts = xy[surf_mask]
                if pts.shape[0] >= 3:
                    c = pts.mean(axis=0)
                    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
                    poly = MplPath(pts[np.argsort(ang)], closed=True)
                    centers = np.mean(np.stack([tri.x[tri.triangles], tri.y[tri.triangles]], axis=-1), axis=1)
                    inside = poly.contains_points(centers, radius=-1e-6)
                    if inside is not None and inside.any():
                        tri.set_mask(inside.astype(bool))
        except Exception:
            pass

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    a1.tricontourf(tri, gt_c, levels=50, vmin=vmin, vmax=vmax, cmap=cmap)
    a2.tricontourf(tri, pr_c, levels=50, vmin=vmin, vmax=vmax, cmap=cmap)
    a3.tricontourf(tri, err, levels=50, cmap='magma')
    if show_mesh:
        for a in (a1, a2, a3):
            a.triplot(tri, color='k', lw=0.25, alpha=0.35)
    for a, t in zip((a1, a2, a3), ('Ground Truth', 'Prediction', 'Abs Error')):
        a.set_aspect('equal', 'box')
        a.set_title(t)
        a.set_xlabel('x'); a.set_ylabel('y')
    m1 = plt.cm.ScalarMappable(cmap=cmap); m1.set_clim(vmin, vmax)
    fig.colorbar(m1, ax=[a1, a2], fraction=0.046, pad=0.04, label='Cp')
    fig.colorbar(plt.cm.ScalarMappable(cmap='magma'), ax=a3, fraction=0.046, pad=0.04, label='abs error')

    if save:
        os.makedirs(save_dir, exist_ok=True)
        from datetime import datetime
        filename = os.path.join(save_dir, f"pred_vs_gt_channel{channel}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename, dpi=200)
        print(f"[Saved] {filename}")
    plt.show()


def plot_training_loss(history: dict, save_dir: str = "experiments/loss_signals"):
    """Plot loss curves from a Trainer history dict."""
    import matplotlib.pyplot as plt

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    keys = ["total_loss", "mse_loss", "continuity_loss", "momentum_loss", "bc_loss"]
    labels = {"total_loss": "Total Loss", "mse_loss": "MSE Loss",
              "continuity_loss": "Continuity Loss", "momentum_loss": "Momentum Loss", "bc_loss": "BC Loss"}

    train_logs = history.get("train", [])
    val_logs = history.get("val", [])
    if not train_logs:
        return

    epochs = list(range(1, len(train_logs) + 1))
    active_keys = [k for k in keys
                   if any(math.isfinite(log.get(k, float("nan"))) for log in train_logs + val_logs)]
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
        t_mask, v_mask = np.isfinite(train_vals), np.isfinite(val_vals)
        if np.any(t_mask):
            ax.plot(ep[t_mask], train_vals[t_mask], label="train", marker="o", linewidth=1.2)
        if np.any(v_mask):
            ax.plot(ep[v_mask], val_vals[v_mask], label="val", marker="x", linewidth=1.2)
        ax.set_title(labels.get(k, k)); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25); ax.legend()

    for idx in range(len(active_keys), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.tight_layout()
    out_path = save_path / "loss_signals_epoch_vs_loss.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot_training_loss] saved to {out_path}")


def plot_model_predictions(model: nn.Module, bundle: DataBundle, device: str = "cuda"):
    """Run a single val-graph prediction and plot pred-vs-gt (channel 2 = pressure)."""
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    graphs = bundle.val_graphs if bundle.val_graphs else bundle.train_graphs
    if not graphs:
        print("[plot_model_predictions] no graphs available")
        return

    dm_vis, y_pred_vis = predict_one_for_viz(graphs[0], model, bundle.x_scaler, bundle.y_scaler, dev)
    plot_pred_vs_gt(dm_vis, y_pred_vis, channel=2, show_mesh=False, denormalize=True,
                    y_scaler=bundle.y_scaler, x_scaler=bundle.x_scaler)
    print("[plot_model_predictions] saved to figures/")
