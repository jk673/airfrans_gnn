#!/usr/bin/env python3
"""
scripts/optuna_hpo.py — Hyperparameter optimization with Optuna for AirfRANS GNN.
Converted from 02_optuna_training.ipynb with CLI support and visualization.

Usage:
    # Basic HPO run with default settings
    python scripts/optuna_hpo.py

    # Custom study name and trials
    python scripts/optuna_hpo.py --study-name my-study --n-trials 50

    # Use SQLite storage for persistence
    python scripts/optuna_hpo.py --storage sqlite:///optuna.db --study-name gnn-hpo

    # Resume an existing study
    python scripts/optuna_hpo.py --storage sqlite:///optuna.db --study-name gnn-hpo --resume

    # Limit trial epochs and change pruning strategy
    python scripts/optuna_hpo.py --trial-epochs 15 --pruner median

    # Visualize only (no optimization)
    python scripts/optuna_hpo.py --visualize-only --storage sqlite:///optuna.db --study-name gnn-hpo

    # Export results to JSON
    python scripts/optuna_hpo.py --export-json results.json

Examples:
    # Run 30 trials with early stopping
    python scripts/optuna_hpo.py --n-trials 30 --early-stopping

    # Parallel optimization (CPU only, or use Ray for distributed)
    python scripts/optuna_hpo.py --n-jobs 2 --n-trials 50

    # Optimize with custom data limits
    python scripts/optuna_hpo.py --limit-train 150 --limit-val 15
"""

from __future__ import annotations

import argparse
import os
import sys
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from src.config import SmokeCfg
from src.data import DataBundle, collate_pyg, load_and_prepare_data
from src.training import set_seed, compute_loss_with_physics, create_lr_scheduler
from src.physics_loss import NavierStokesPhysicsLoss
from src.global_context_processor import EnhancedCFDModelWithGlobalContext

import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner, HyperbandPruner, PatientPruner
from optuna.samplers import TPESampler, RandomSampler


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, data_bundle: DataBundle, device: torch.device, args) -> float:
    """Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        data_bundle: Pre-loaded data bundle with loaders and scalers
        device: Torch device (cuda/cpu)
        args: CLI arguments

    Returns:
        Best validation loss achieved during trial
    """

    # 1) Model Architecture Hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    num_layers = trial.suggest_int('num_layers', 3, 14)
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
    bc_loss_weight = trial.suggest_float('bc_loss_weight', 0.001, 0.3, log=True)

    # 5) Curriculum Learning
    ramp_start_epoch = trial.suggest_int('ramp_start_epoch', 5, 20)
    ramp_epochs = trial.suggest_int('ramp_epochs', 10, 30)
    ramp_mode = trial.suggest_categorical('ramp_mode', ['linear', 'cosine'])

    # 6) Global Context & Attention
    use_global_tokens = trial.suggest_categorical('use_global_tokens', [True, False])
    if use_global_tokens:
        num_global_tokens = trial.suggest_categorical('num_global_tokens', [2, 4, 8])
        attention_heads = trial.suggest_categorical('attention_heads', [2, 4, 8])
        attention_layers = trial.suggest_int('attention_layers', 2, 8)
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
        ramp_mode=ramp_mode,
        use_global_tokens=use_global_tokens,
        num_global_tokens=num_global_tokens,
        attention_heads=attention_heads,
        attention_layers=attention_layers,
        use_cross_attention=use_cross_attention,
        positional_encoding=positional_encoding,
        global_pooling_type=global_pooling_type,
        lr_scheduler=lr_scheduler_type,
        epochs=args.trial_epochs,
        wandb_mode='disabled',
        amp=args.amp,
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
        node_feat_dim=7, edge_feat_dim=data_bundle.edge_dim,
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
    spe = max(1, len(train_loader_trial))
    loss_fn_trial = NavierStokesPhysicsLoss(
        data_loss_weight=getattr(config, 'data_loss_weight', 1.0),
        continuity_loss_weight=config.continuity_loss_weight,
        continuity_target_weight=config.continuity_target_weight,
        momentum_loss_weight=config.momentum_loss_weight,
        momentum_target_weight=config.momentum_target_weight,
        curriculum_ramp_steps=config.ramp_epochs * spe,
        ramp_start_step=config.ramp_start_epoch * spe,
        ramp_mode=config.ramp_mode,
        cont_ramp_start_step=getattr(config, 'cont_ramp_start_epoch', -1) * spe if getattr(config, 'cont_ramp_start_epoch', -1) >= 0 else -1,
        cont_curriculum_ramp_steps=getattr(config, 'cont_ramp_epochs', -1) * spe if getattr(config, 'cont_ramp_epochs', -1) >= 0 else -1,
        mom_ramp_start_step=getattr(config, 'mom_ramp_start_epoch', -1) * spe if getattr(config, 'mom_ramp_start_epoch', -1) >= 0 else -1,
        mom_curriculum_ramp_steps=getattr(config, 'mom_ramp_epochs', -1) * spe if getattr(config, 'mom_ramp_epochs', -1) >= 0 else -1,
        bc_ramp_start_step=getattr(config, 'bc_ramp_start_epoch', -1) * spe if getattr(config, 'bc_ramp_start_epoch', -1) >= 0 else -1,
        bc_curriculum_ramp_steps=getattr(config, 'bc_ramp_epochs', -1) * spe if getattr(config, 'bc_ramp_epochs', -1) >= 0 else -1,
        bc_loss_weight=config.bc_loss_weight,
        chord_length=getattr(config, 'chord_length', 1.0),
        dynamic_uref_from_data=getattr(config, 'dynamic_uref_from_data', True),
        dynamic_re_from_data=getattr(config, 'dynamic_re_from_data', True),
        nu_molecular=getattr(config, 'nu_molecular', 1.5e-5),
        use_huber_for_physics=getattr(config, 'use_huber_for_physics', True),
        huber_delta=getattr(config, 'huber_delta', 0.05),
        debug=False,
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = args.patience if args.early_stopping else config.epochs

    for epoch in range(config.epochs):
        # === Training ===
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

        # === Validation ===
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
                    val_losses.append(float(total.item()) if hasattr(total, 'item') else float(total))

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
            if args.early_stopping and patience_counter >= max_patience:
                print(f"  Trial {trial.number}: Early stopping at epoch {epoch}")
                break

        # Report to Optuna (for pruning)
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Clean up
    del model_trial, optimizer_trial, scheduler_trial, loss_fn_trial
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_loss


# ---------------------------------------------------------------------------
# Visualization and analysis
# ---------------------------------------------------------------------------

def print_study_results(study: optuna.Study):
    """Print study results and top trials."""
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]

    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Total trials: {len(study.trials)}")
    print(f"  Completed trials: {len(completed_trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Failed trials: {len(failed_trials)}")

    if len(completed_trials) == 0:
        print("\nNo completed trials to analyze.")
        return

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best value: {study.best_value:.6f}")

    print("\nBest parameters:")
    for key, value in sorted(study.best_params.items()):
        print(f"  {key:30s}: {value}")

    # Top trials
    df = study.trials_dataframe()
    df_completed = df[df['state'] == 'COMPLETE'].sort_values('value')

    if len(df_completed) > 0:
        print(f"\nTop {min(5, len(df_completed))} trials:")
        cols = ['number', 'value']
        param_cols = ['params_hidden_dim', 'params_lr', 'params_num_layers',
                      'params_batch_size', 'params_use_global_tokens']
        for c in param_cols:
            if c in df_completed.columns:
                cols.append(c)

        print(df_completed[cols].head(5).to_string(index=False))


def visualize_study(study: optuna.Study, save_dir: str | None = None):
    """Generate and optionally save Optuna visualizations."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plots = []

    # 1. Optimization history
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        plots.append(("optimization_history", fig))
        if not save_dir:
            fig.show()
    except Exception as e:
        print(f"Could not plot optimization history: {e}")

    # 2. Parameter importance
    try:
        fig = optuna.visualization.plot_param_importances(study)
        plots.append(("param_importances", fig))
        if not save_dir:
            fig.show()
    except Exception as e:
        print(f"Could not plot param importances: {e}")

    # 3. Parallel coordinate plot
    try:
        params = ['hidden_dim', 'num_layers', 'lr', 'continuity_weight', 'momentum_weight']
        # Filter params that exist in study
        available_params = [p for p in params if any(p in t.params for t in study.trials)]
        if available_params:
            fig = optuna.visualization.plot_parallel_coordinate(study, params=available_params)
            plots.append(("parallel_coordinate", fig))
            if not save_dir:
                fig.show()
    except Exception as e:
        print(f"Could not plot parallel coordinate: {e}")

    # 4. Slice plot
    try:
        params = ['lr', 'hidden_dim', 'num_layers', 'batch_size']
        available_params = [p for p in params if any(p in t.params for t in study.trials)]
        if available_params:
            fig = optuna.visualization.plot_slice(study, params=available_params)
            plots.append(("slice", fig))
            if not save_dir:
                fig.show()
    except Exception as e:
        print(f"Could not plot slice: {e}")

    # 5. Contour plot for key hyperparameters
    try:
        fig = optuna.visualization.plot_contour(study, params=['lr', 'hidden_dim'])
        plots.append(("contour_lr_hidden", fig))
        if not save_dir:
            fig.show()
    except Exception as e:
        print(f"Could not plot contour: {e}")

    # Save plots if directory specified
    if save_dir and plots:
        for name, fig in plots:
            filepath = os.path.join(save_dir, f"{name}.html")
            fig.write_html(filepath)
            print(f"Saved: {filepath}")


def export_results(study: optuna.Study, filepath: str):
    """Export study results to JSON."""
    results = {
        'study_name': study.study_name,
        'n_trials': len(study.trials),
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial_number': study.best_trial.number,
        'trials': []
    }

    for trial in study.trials:
        trial_data = {
            'number': trial.number,
            'state': trial.state.name,
            'value': trial.value,
            'params': trial.params,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            'duration': trial.duration.total_seconds() if trial.duration else None,
        }
        results['trials'].append(trial_data)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults exported to: {filepath}")


def build_config_from_best(study: optuna.Study, epochs: int = 100) -> SmokeCfg:
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
        ramp_mode=best_params.get('ramp_mode', 'linear'),
        use_global_tokens=best_params['use_global_tokens'],
        num_global_tokens=best_params.get('num_global_tokens', 4),
        attention_heads=best_params.get('attention_heads', 4),
        attention_layers=best_params.get('attention_layers', 2),
        use_cross_attention=best_params.get('use_cross_attention', True),
        positional_encoding=best_params.get('positional_encoding', False),
        global_pooling_type=best_params.get('global_pooling_type', 'attention'),
        lr_scheduler=best_params['lr_scheduler'],
        epochs=epochs,
        wandb_mode='online',
    )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter optimization for AirfRANS GNN with Optuna',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Study configuration
    parser.add_argument('--study-name', type=str, default='airfrans-gnn-hpo',
                        help='Name of the Optuna study')
    parser.add_argument('--storage', type=str, default=None,
                        help='Storage URL (e.g., sqlite:///optuna.db). If None, in-memory storage.')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume existing study (requires --storage and --study-name)')

    # Optimization parameters
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials to run')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs (use 1 for GPU)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds for the entire study')

    # Trial configuration
    parser.add_argument('--trial-epochs', type=int, default=20,
                        help='Number of epochs per trial')
    parser.add_argument('--early-stopping', action='store_true', default=False,
                        help='Enable early stopping for trials')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')

    # Pruning strategy
    parser.add_argument('--pruner', type=str, default='median',
                        choices=['median', 'hyperband', 'patient', 'none'],
                        help='Pruning strategy')
    parser.add_argument('--pruner-startup-trials', type=int, default=5,
                        help='Number of startup trials before pruning starts')
    parser.add_argument('--pruner-warmup-steps', type=int, default=5,
                        help='Number of warmup steps before pruning')

    # Sampler
    parser.add_argument('--sampler', type=str, default='tpe',
                        choices=['tpe', 'random'],
                        help='Sampling strategy')

    # Data configuration
    parser.add_argument('--task', type=str, default='scarce',
                        choices=['full', 'scarce', 'medium'],
                        help='AirfRANS task variant')
    parser.add_argument('--root', type=str, default='Dataset',
                        help='Path to AirfRANS dataset root')
    parser.add_argument('--limit-train', type=int, default=180,
                        help='Limit number of training graphs')
    parser.add_argument('--limit-val', type=int, default=20,
                        help='Limit number of validation graphs')

    # Output and visualization
    parser.add_argument('--visualize-only', action='store_true', default=False,
                        help='Only visualize existing study (no optimization)')
    parser.add_argument('--viz-dir', type=str, default='optuna_viz',
                        help='Directory to save visualization plots')
    parser.add_argument('--export-json', type=str, default=None,
                        help='Export results to JSON file')
    parser.add_argument('--save-best-config', type=str, default=None,
                        help='Save best config to JSON file')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, auto-detect if not specified)')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use automatic mixed precision')
    parser.add_argument('--gc-after-trial', action='store_true', default=True,
                        help='Run garbage collection after each trial')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    args = parse_args()
    set_seed(args.seed)

    # Device configuration
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 80)
    print('AirfRANS GNN Hyperparameter Optimization with Optuna')
    print('=' * 80)
    print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {device}')
    print(f'Study: {args.study_name}')
    print(f'Storage: {args.storage or "in-memory"}')
    print(f'Trials: {args.n_trials} | Trial epochs: {args.trial_epochs}')
    print(f'Pruner: {args.pruner} | Sampler: {args.sampler}')
    print(f'Task: {args.task} | Train graphs: {args.limit_train} | Val graphs: {args.limit_val}')
    print('=' * 80)

    # Create pruner
    if args.pruner == 'median':
        pruner = MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=args.pruner_warmup_steps,
            interval_steps=1
        )
    elif args.pruner == 'hyperband':
        pruner = HyperbandPruner()
    elif args.pruner == 'patient':
        pruner = PatientPruner(
            MedianPruner(n_startup_trials=args.pruner_startup_trials),
            patience=args.patience
        )
    else:
        pruner = None

    # Create sampler
    if args.sampler == 'tpe':
        sampler = TPESampler(seed=args.seed)
    elif args.sampler == 'random':
        sampler = RandomSampler(seed=args.seed)
    else:
        sampler = TPESampler(seed=args.seed)

    # Load or create study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='minimize',
        pruner=pruner,
        sampler=sampler,
        load_if_exists=args.resume or args.visualize_only,
    )

    if args.visualize_only:
        print("\nVisualization mode (no optimization)")
        print(f"Analyzing study: {args.study_name}")
        print(f"Total trials: {len(study.trials)}")

        print_study_results(study)
        visualize_study(study, save_dir=args.viz_dir)

        if args.export_json:
            export_results(study, args.export_json)

        if args.save_best_config and len(study.trials) > 0:
            best_config = build_config_from_best(study)
            with open(args.save_best_config, 'w') as f:
                json.dump(asdict(best_config), f, indent=2)
            print(f"\nBest config saved to: {args.save_best_config}")

        return

    # Load data (once, shared across trials)
    scfg = SmokeCfg(
        task=args.task,
        root=args.root,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        seed=args.seed,
    )
    data_bundle = load_and_prepare_data(scfg)

    # Enqueue default trial (baseline)
    if not args.resume:
        print("\nEnqueuing baseline trial...")
        study.enqueue_trial({
            'hidden_dim': 128, 'num_layers': 7, 'dropout': 0.1,
            'lr': 4e-4, 'weight_decay': 1e-2, 'batch_size': 2,
            'beta1': 0.9, 'beta2': 0.95, 'eps': 1e-8,
            'continuity_weight': 0.05, 'continuity_target_weight': 0.10,
            'momentum_weight': 0.05, 'momentum_target_weight': 0.10,
            'bc_loss_weight': 0.05,
            'ramp_start_epoch': 10, 'ramp_epochs': 10, 'ramp_mode': 'linear',
            'use_global_tokens': True, 'num_global_tokens': 4,
            'attention_heads': 4, 'attention_layers': 2,
            'use_cross_attention': True, 'positional_encoding': False,
            'global_pooling_type': 'attention',
            'lr_scheduler': 'cosine',
        })

    # Run optimization
    print("\nStarting optimization...")
    print("-" * 80)

    study.optimize(
        lambda trial: objective(trial, data_bundle, device, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        gc_after_trial=args.gc_after_trial,
        show_progress_bar=True,
    )

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 80)

    # Results
    print_study_results(study)

    # Visualizations
    if args.viz_dir:
        print(f"\nGenerating visualizations...")
        visualize_study(study, save_dir=args.viz_dir)

    # Export results
    if args.export_json:
        export_results(study, args.export_json)

    # Save best config
    if args.save_best_config:
        best_config = build_config_from_best(study)
        with open(args.save_best_config, 'w') as f:
            json.dump(asdict(best_config), f, indent=2)
        print(f"\nBest config saved to: {args.save_best_config}")

    # Print final best config for reference
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION (for full training)")
    print("=" * 80)
    best_config = build_config_from_best(study, epochs=100)
    for key, value in asdict(best_config).items():
        print(f"  {key:30s}: {value}")
    print("=" * 80)


if __name__ == '__main__':
    main()
