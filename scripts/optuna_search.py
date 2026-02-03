#!/usr/bin/env python3
"""
scripts/optuna_search.py — Hyperparameter optimization with Optuna.
Converted from 02_optuna_training.ipynb.

Usage:
    python scripts/optuna_search.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training_common import (
    SmokeCfg, DataBundle,
    set_seed, collate_pyg,
    load_and_prepare_data, compute_loss_with_physics,
    create_lr_scheduler,
)
from src.navier_stokes_physics_loss import NavierStokesPhysicsLoss
from src.global_context_processor import EnhancedCFDModelWithGlobalContext

import optuna
from optuna.trial import TrialState


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def objective(trial, data_bundle: DataBundle, device):
    """Optuna objective function for hyperparameter optimization.
    Returns validation loss to minimize.
    """

    # 1) Model Architecture Hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    num_layers = trial.suggest_int('num_layers', 3, 10)
    dropout_p = trial.suggest_float('dropout', 0.0, 0.5, step=0.05)

    # 2) Training Hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1, 2, 4, 8])

    # 3) Optimizer Configuration
    beta1 = trial.suggest_float('beta1', 0.8, 0.99)
    beta2 = trial.suggest_float('beta2', 0.9, 0.999)
    eps = trial.suggest_float('eps', 1e-9, 1e-6, log=True)

    # 4) Physics Loss Hyperparameters
    continuity_weight = trial.suggest_float('continuity_weight', 0.001, 0.5, log=True)
    continuity_target_weight = trial.suggest_float(
        'continuity_target_weight', continuity_weight, 1.0)
    momentum_weight = trial.suggest_float('momentum_weight', 0.001, 0.5, log=True)
    momentum_target_weight = trial.suggest_float(
        'momentum_target_weight', momentum_weight, 1.0)
    bc_loss_weight = trial.suggest_float('bc_loss_weight', 0.001, 0.2, log=True)

    # 5) Curriculum Learning
    ramp_start_epoch = trial.suggest_int('ramp_start_epoch', 5, 20)
    ramp_epochs = trial.suggest_int('ramp_epochs', 10, 20)

    # 6) Global Context
    use_global_tokens = trial.suggest_categorical('use_global_tokens', [True, False])
    if use_global_tokens:
        num_global_tokens = trial.suggest_categorical('num_global_tokens', [2, 4, 8])
        attention_heads = trial.suggest_categorical('attention_heads', [2, 4, 8])
        attention_layers = trial.suggest_categorical('attention_layers', [2, 4, 8])
        use_cross_attention = trial.suggest_categorical('use_cross_attention', [True, False])
        positional_encoding = trial.suggest_categorical('positional_encoding', [True, False])
        global_pooling_type = trial.suggest_categorical(
            'global_pooling_type', ['mean', 'max', 'attention', 'set2set'])
    else:
        num_global_tokens = 0
        attention_heads = 4
        attention_layers = 2
        use_cross_attention = False
        positional_encoding = False
        global_pooling_type = 'mean'

    # 7) Learning Rate Scheduler
    lr_scheduler_type = trial.suggest_categorical(
        'lr_scheduler', ['cosine', 'cosine_warm_restarts', 'reduce_on_plateau', None])

    # Build config
    config = SmokeCfg(
        hidden=hidden_dim,
        layers=num_layers,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        betas=(beta1, beta2),
        eps=eps,
        continuity_loss_weight=continuity_weight,
        continuity_target_weight=continuity_target_weight,
        momentum_loss_weight=momentum_weight,
        momentum_target_weight=momentum_target_weight,
        bc_loss_weight=bc_loss_weight,
        ramp_start_epoch=ramp_start_epoch,
        ramp_epochs=ramp_epochs,
        use_global_tokens=use_global_tokens,
        num_global_tokens=num_global_tokens,
        attention_heads=attention_heads,
        attention_layers=attention_layers,
        use_cross_attention=use_cross_attention,
        positional_encoding=positional_encoding,
        global_pooling_type=global_pooling_type,
        lr_scheduler=lr_scheduler_type,
        epochs=20,
        wandb_mode='disabled',
    )

    # DataLoaders (batch size from trial)
    train_loader_trial = DataLoader(
        data_bundle.train_norm, batch_size=config.batch_size,
        shuffle=True, num_workers=0, collate_fn=collate_pyg)
    val_loader_trial = DataLoader(
        data_bundle.val_norm, batch_size=config.batch_size,
        shuffle=False, num_workers=0, collate_fn=collate_pyg)

    # Model
    model_trial = EnhancedCFDModelWithGlobalContext(
        node_feat_dim=7, edge_feat_dim=5,
        hidden_dim=config.hidden, output_dim=4,
        num_mp_layers=config.layers,
        dropout_p=dropout_p, config=config,
    ).to(device)

    # Optimizer
    optimizer_trial = torch.optim.AdamW(
        model_trial.parameters(), lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas, eps=config.eps)

    # LR Scheduler
    scheduler_trial = create_lr_scheduler(optimizer_trial, config)

    # Physics Loss
    loss_fn_trial = NavierStokesPhysicsLoss(
        data_loss_weight=getattr(config, 'data_loss_weight', 1.0),
        continuity_loss_weight=config.continuity_loss_weight,
        continuity_target_weight=config.continuity_target_weight,
        momentum_loss_weight=config.momentum_loss_weight,
        momentum_target_weight=config.momentum_target_weight,
        curriculum_ramp_steps=config.ramp_epochs * max(1, len(train_loader_trial)),
        ramp_start_step=config.ramp_start_epoch * max(1, len(train_loader_trial)),
        bc_loss_weight=config.bc_loss_weight,
        chord_length=getattr(config, 'chord_length', 1.0),
        dynamic_uref_from_data=getattr(config, 'dynamic_uref_from_data', False),
        dynamic_re_from_data=getattr(config, 'dynamic_re_from_data', False),
        nu_molecular=getattr(config, 'nu_molecular', 1.5e-5),
        use_huber_for_physics=getattr(config, 'use_huber_for_physics', False),
        huber_delta=getattr(config, 'huber_delta', 1.0),
        debug=False,
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5

    for epoch in range(config.epochs):
        model_trial.train()
        global_step = epoch * max(1, len(train_loader_trial))

        for batch in train_loader_trial:
            if batch is None:
                continue
            batch = batch.to(device)
            optimizer_trial.zero_grad()

            predictions = model_trial(batch)
            loss, _ = compute_loss_with_physics(
                predictions, batch.y, batch,
                loss_fn=loss_fn_trial, step=global_step)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_trial.parameters(), 1.0)
            optimizer_trial.step()
            global_step += 1

        # Validation
        model_trial.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader_trial:
                if batch is None:
                    continue
                batch = batch.to(device)
                predictions = model_trial(batch)
                _, loss_dict = compute_loss_with_physics(
                    predictions, batch.y, batch, loss_fn=loss_fn_trial)
                total = loss_dict.get('total_loss', loss_dict.get('loss', None))
                if total is None:
                    continue
                try:
                    val_losses.append(float(total))
                except Exception:
                    val_losses.append(float(total.item()))

        avg_val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else float('inf')

        # Scheduler step
        if scheduler_trial is not None:
            if isinstance(scheduler_trial, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_trial.step(avg_val_loss)
            else:
                scheduler_trial.step()

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                break

        # Report to Optuna (for pruning)
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Clean up
    del model_trial, optimizer_trial
    torch.cuda.empty_cache()

    return best_val_loss


# ---------------------------------------------------------------------------
# Visualization / analysis helpers
# ---------------------------------------------------------------------------

def print_study_results(study):
    """Print study results and top trials."""
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]

    print(f"\nStatistics:")
    print(f"  Completed trials: {len(completed_trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"  Best value: {study.best_value:.6f}")

    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    df = study.trials_dataframe()
    df_sorted = df.sort_values('value').head(5)
    print("\nTop 5 trials:")
    cols = ['number', 'value']
    for c in ['params_hidden_dim', 'params_lr', 'params_num_layers']:
        if c in df_sorted.columns:
            cols.append(c)
    print(df_sorted[cols])


def visualize_study(study):
    """Generate Optuna visualizations."""
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
    except Exception as e:
        print(f"Could not plot optimization history: {e}")

    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.show()
    except Exception as e:
        print(f"Could not plot param importances: {e}")

    try:
        fig = optuna.visualization.plot_parallel_coordinate(
            study,
            params=['hidden_dim', 'num_layers', 'lr', 'continuity_weight', 'momentum_weight'])
        fig.show()
    except Exception as e:
        print(f"Could not plot parallel coordinate: {e}")

    try:
        fig = optuna.visualization.plot_slice(
            study, params=['lr', 'hidden_dim', 'num_layers', 'batch_size'])
        fig.show()
    except Exception as e:
        print(f"Could not plot slice: {e}")


def build_config_from_best(study, epochs=100):
    """Build a SmokeCfg from the best trial parameters."""
    best_params = study.best_params
    return SmokeCfg(
        hidden=best_params['hidden_dim'],
        layers=best_params['num_layers'],
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
        batch_size=best_params['batch_size'],
        betas=(best_params['beta1'], best_params['beta2']),
        eps=best_params['eps'],
        continuity_loss_weight=best_params['continuity_weight'],
        continuity_target_weight=best_params['continuity_target_weight'],
        momentum_loss_weight=best_params['momentum_weight'],
        momentum_target_weight=best_params['momentum_target_weight'],
        bc_loss_weight=best_params['bc_loss_weight'],
        ramp_start_epoch=best_params['ramp_start_epoch'],
        ramp_epochs=best_params['ramp_epochs'],
        use_global_tokens=best_params['use_global_tokens'],
        num_global_tokens=best_params.get('num_global_tokens', 4),
        attention_heads=best_params.get('attention_heads', 4),
        lr_scheduler=best_params['lr_scheduler'],
        epochs=epochs,
        wandb_mode='online',
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    scfg = SmokeCfg()
    set_seed(scfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {device}')

    # --- Data (loaded once, shared across trials) ---
    data_bundle = load_and_prepare_data(scfg)

    # --- Optuna study ---
    study_name = "airfrans_gnn_optimization"
    storage_name = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    # Enqueue default trial
    study.enqueue_trial({
        'hidden_dim': 128, 'num_layers': 7, 'dropout': 0.1,
        'lr': 4e-4, 'weight_decay': 1e-2, 'batch_size': 2,
        'beta1': 0.9, 'beta2': 0.95, 'eps': 1e-8,
        'continuity_weight': 0.05, 'continuity_target_weight': 0.10,
        'momentum_weight': 0.05, 'momentum_target_weight': 0.10,
        'bc_loss_weight': 0.05,
        'ramp_start_epoch': 10, 'ramp_epochs': 10,
        'use_global_tokens': True, 'num_global_tokens': 4,
        'attention_heads': 4, 'lr_scheduler': 'cosine',
    })

    # Run optimization
    n_trials = 100
    study.optimize(
        lambda trial: objective(trial, data_bundle, device),
        n_trials=n_trials, timeout=None, n_jobs=1,
        gc_after_trial=True, show_progress_bar=True,
    )

    print("\n" + "=" * 50)
    print("Optimization Complete!")
    print("=" * 50)

    print_study_results(study)
    visualize_study(study)

    # Build best config for retraining
    final_config = build_config_from_best(study, epochs=100)
    print(f"\nBest config ready for retraining: {asdict(final_config)}")


if __name__ == '__main__':
    main()
