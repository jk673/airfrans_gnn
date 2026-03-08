#!/usr/bin/env python3
"""Declarative AirfRANS training script.

Usage:
    python scripts/train.py
"""

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
import torch
from src.pipeline import (
    load_airfrans_data, convert_to_pyg, build_model, build_physics_loss,
    Trainer, train_one_epoch, validate_one_epoch,
    save_model_checkpoint, log_training_metrics,
    plot_training_loss, plot_model_predictions,
)
from src.benchmark import run_benchmark_and_log_experiment


# ============================================================================
# Experiment Config — 여기만 수정하면 됩니다
# ============================================================================

@dataclass
class PhysicsRamp:
    weight: float = 0.0
    target: float = 0.0
    ramp_start_epoch: int = 50
    ramp_epochs: int = 30


@dataclass
class Config:
    # -- Data --
    task: str = "scarce"
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 4

    # -- Model --
    hidden_dim: int = 16
    num_layers: int = 14
    num_global_tokens: int = 0
    dropout: float = 0.1

    # -- Physics loss --
    continuity: PhysicsRamp = field(default_factory=lambda: PhysicsRamp(weight=0.0, target=0.01))
    momentum: PhysicsRamp = field(default_factory=lambda: PhysicsRamp(weight=0.0, target=0.05))
    bc: PhysicsRamp = field(default_factory=lambda: PhysicsRamp(weight=0.01, target=0.01))

    # -- Optimizer --
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # -- Scheduler --
    scheduler_T_max: int = 100

    # -- Training --
    num_epochs: int = 300
    device: str = "cuda"
    amp: bool = True
    live_plot: bool = True
    dashboard_refresh_every: int = 5

    # -- Experiment --
    notes: str = ""


cfg = Config()

# ============================================================================


# 1. Data
data = load_airfrans_data(task=cfg.task, seed=cfg.seed)
bundle = convert_to_pyg(data, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

# 2. Model
model = build_model({
    'type': 'EnhancedCFDModelWithGlobalContext',
    'input': {
        'node_dim': 7,
        'edge_dim': bundle.edge_dim,
        'hidden_dim': cfg.hidden_dim,
        'num_layers': cfg.num_layers,
        'num_global_tokens': cfg.num_global_tokens,
        'dropout': cfg.dropout,
    },
    'output': {'output_dim': 4},
})

# 3. Physics loss
criterion = build_physics_loss({
    'continuity': asdict(cfg.continuity),
    'momentum': asdict(cfg.momentum),
    'bc': asdict(cfg.bc),
}, steps_per_epoch=len(bundle.train_loader))

# 4. Optimizer & scheduler
optimizer = torch.optim.adamw.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.scheduler_T_max)

# 5. Train
training_start = time.time()
trainer = Trainer(model, optimizer, scheduler, criterion, device=cfg.device,
                  live_plot=cfg.live_plot, dashboard_refresh_every=cfg.dashboard_refresh_every,
                  amp=cfg.amp)
history = trainer.fit(
    bundle.train_loader, bundle.val_loader,
    num_epochs=cfg.num_epochs,
    routine={
        'train': train_one_epoch,
        'validate': validate_one_epoch,
        'save_checkpoint': save_model_checkpoint,
        'log_metrics': log_training_metrics,
    },
)
training_duration = time.time() - training_start

# 6. Post-training visualization
plot_training_loss(history)
plot_model_predictions(model, bundle)

# 7. Benchmark scoring
run_benchmark_and_log_experiment(
    model=model,
    data_bundle=bundle,
    scfg={'task': cfg.task, 'hidden': cfg.hidden_dim, 'layers': cfg.num_layers},
    device=torch.device(cfg.device),
    train_summary=history,
    training_duration_sec=training_duration,
    notes=cfg.notes,
)
