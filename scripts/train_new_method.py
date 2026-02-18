#!/usr/bin/env python3
"""Declarative AirfRANS training script (~55 lines).

Usage:
    python scripts/train_new_method.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.pipeline import (
    load_airfrans_data, convert_to_pyg, build_model, build_physics_loss,
    Trainer, train_one_epoch, validate_one_epoch,
    save_model_checkpoint, log_training_metrics,
    plot_training_loss, plot_model_predictions,
)

# 1. Data
data = load_airfrans_data(task='scarce', seed=42)
bundle = convert_to_pyg(data, batch_size=2)

# 2. Model (dict config -> nn.Module)
model = build_model({
    'type': 'EnhancedCFDModelWithGlobalContext',
    'input': {
        'node_dim': 7,
        'edge_dim': bundle.edge_dim,
        'hidden_dim': 128,
        'num_layers': 4,
        'num_global_tokens': 16,
    },
    'output': {'output_dim': 4},
})

# 3. Physics loss
criterion = build_physics_loss({
    'continuity': {'weight': 0.0, 'target': 0.15, 'ramp_start_epoch': 50, 'ramp_epochs': 30},
    'momentum':   {'weight': 0.0, 'target': 0.01, 'ramp_start_epoch': 50, 'ramp_epochs': 30},
    'bc':         {'weight': 0.01, 'ramp_start_epoch': 50, 'ramp_epochs': 30},
}, steps_per_epoch=len(bundle.train_loader))

# 4. Optimizer & scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 5. Train (live_plot=True -> browser dashboard auto-generated)
trainer = Trainer(model, optimizer, scheduler, criterion, device='cuda', live_plot=True)
history = trainer.fit(
    bundle.train_loader, bundle.val_loader,
    num_epochs=100,
    routine={
        'train': train_one_epoch,
        'validate': validate_one_epoch,
        'save_checkpoint': save_model_checkpoint,
        'log_metrics': log_training_metrics,
    },
)

# 6. Post-training visualization
plot_training_loss(history)
plot_model_predictions(model, bundle)
