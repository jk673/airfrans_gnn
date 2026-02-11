# Optuna HPO Script Documentation

## Overview

`optuna_hpo.py` provides hyperparameter optimization for the AirfRANS GNN model using [Optuna](https://optuna.org/). It automatically searches for optimal combinations of:

- Model architecture (hidden dimensions, layers, dropout)
- Training hyperparameters (learning rate, weight decay, batch size)
- Physics loss weights and curriculum schedule
- Global context & attention settings
- Learning rate scheduler configuration

## Features

- **CLI Interface**: Comprehensive command-line arguments for all configuration
- **Persistent Storage**: SQLite or other database backends for resumable studies
- **Pruning Strategies**: MedianPruner, HyperbandPruner, or PatientPruner
- **Parallel Execution**: Multi-job support (CPU-only; use n_jobs=1 for GPU)
- **Visualization**: Automatic generation of optimization plots (history, importance, etc.)
- **Export Results**: JSON export of trials and best configurations
- **Early Stopping**: Optional early stopping for faster convergence
- **Resume Support**: Continue interrupted optimization runs

## Quick Start

### Basic Usage

Run 50 trials with default settings:

```bash
python scripts/optuna_hpo.py
```

### Persistent Storage

Use SQLite database to save study progress:

```bash
python scripts/optuna_hpo.py \
    --study-name my-gnn-hpo \
    --storage sqlite:///optuna_studies.db \
    --n-trials 100
```

### Resume an Existing Study

```bash
python scripts/optuna_hpo.py \
    --study-name my-gnn-hpo \
    --storage sqlite:///optuna_studies.db \
    --resume \
    --n-trials 50
```

### Visualize Only (No Optimization)

```bash
python scripts/optuna_hpo.py \
    --visualize-only \
    --study-name my-gnn-hpo \
    --storage sqlite:///optuna_studies.db \
    --viz-dir ./viz_output \
    --export-json results.json \
    --save-best-config best_config.json
```

## Command-Line Arguments

### Study Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--study-name` | str | `airfrans-gnn-hpo` | Name of the Optuna study |
| `--storage` | str | `None` | Storage URL (e.g., `sqlite:///optuna.db`). In-memory if None |
| `--resume` | flag | `False` | Resume existing study (requires `--storage`) |

### Optimization Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n-trials` | int | `50` | Number of trials to run |
| `--n-jobs` | int | `1` | Number of parallel jobs (use 1 for GPU) |
| `--timeout` | int | `None` | Timeout in seconds for the entire study |

### Trial Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--trial-epochs` | int | `20` | Number of epochs per trial |
| `--early-stopping` | flag | `False` | Enable early stopping for trials |
| `--patience` | int | `5` | Patience for early stopping |

### Pruning Strategy

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pruner` | str | `median` | Pruning strategy: `median`, `hyperband`, `patient`, `none` |
| `--pruner-startup-trials` | int | `5` | Number of startup trials before pruning |
| `--pruner-warmup-steps` | int | `5` | Number of warmup steps before pruning |

### Sampler

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sampler` | str | `tpe` | Sampling strategy: `tpe`, `random` |

### Data Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | `scarce` | AirfRANS task: `full`, `scarce`, `medium` |
| `--root` | str | `Dataset` | Path to AirfRANS dataset root |
| `--limit-train` | int | `180` | Limit number of training graphs |
| `--limit-val` | int | `20` | Limit number of validation graphs |

### Output and Visualization

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--visualize-only` | flag | `False` | Only visualize (no optimization) |
| `--viz-dir` | str | `optuna_viz` | Directory to save visualization plots |
| `--export-json` | str | `None` | Export results to JSON file |
| `--save-best-config` | str | `None` | Save best config to JSON file |

### Miscellaneous

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | int | `42` | Random seed |
| `--device` | str | `None` | Device: `cuda`, `cpu` (auto-detect if None) |
| `--amp` | flag | `False` | Use automatic mixed precision |
| `--gc-after-trial` | flag | `True` | Run garbage collection after each trial |

## Hyperparameter Search Space

The script optimizes the following hyperparameters:

### Model Architecture

- **hidden_dim**: Categorical choice from [64, 128, 256, 512]
- **num_layers**: Integer range [3, 14]
- **dropout**: Float range [0.0, 0.5] (step 0.05)

### Training

- **lr**: Log-uniform [1e-5, 1e-2]
- **weight_decay**: Log-uniform [1e-6, 1e-1]
- **batch_size**: Categorical [1, 2, 4, 8]
- **beta1**: Float [0.8, 0.99]
- **beta2**: Float [0.9, 0.999]
- **eps**: Log-uniform [1e-9, 1e-6]

### Physics Loss

- **continuity_weight**: Log-uniform [0.001, 0.5]
- **continuity_target_weight**: Float [continuity_weight, 1.0]
- **momentum_weight**: Log-uniform [0.001, 0.5]
- **momentum_target_weight**: Float [momentum_weight, 1.0]
- **bc_loss_weight**: Log-uniform [0.001, 0.3]

### Curriculum Learning

- **ramp_start_epoch**: Integer [5, 20]
- **ramp_epochs**: Integer [10, 30]
- **ramp_mode**: Categorical ['linear', 'cosine']

### Global Context & Attention

- **use_global_tokens**: Categorical [True, False]
- **num_global_tokens**: Categorical [2, 4, 8] (if enabled)
- **attention_heads**: Categorical [2, 4, 8] (if enabled)
- **attention_layers**: Integer [2, 8] (if enabled)
- **use_cross_attention**: Categorical [True, False] (if enabled)
- **positional_encoding**: Categorical [True, False] (if enabled)
- **global_pooling_type**: Categorical ['mean', 'max', 'attention', 'set2set'] (if enabled)

### LR Scheduler

- **lr_scheduler**: Categorical ['cosine', 'cosine_warm_restarts', 'reduce_on_plateau', None]

## Example Workflows

### 1. Quick Exploration (Small Dataset)

Fast exploration with fewer trials and shorter epochs:

```bash
python scripts/optuna_hpo.py \
    --n-trials 20 \
    --trial-epochs 10 \
    --limit-train 100 \
    --limit-val 10 \
    --early-stopping \
    --patience 3
```

### 2. Production HPO Run

Comprehensive search with persistence and visualization:

```bash
python scripts/optuna_hpo.py \
    --study-name production-hpo \
    --storage sqlite:///production_hpo.db \
    --n-trials 100 \
    --trial-epochs 20 \
    --early-stopping \
    --patience 5 \
    --pruner median \
    --viz-dir viz_production \
    --export-json production_results.json \
    --save-best-config production_best_config.json
```

### 3. Resume Interrupted Study

```bash
python scripts/optuna_hpo.py \
    --study-name production-hpo \
    --storage sqlite:///production_hpo.db \
    --resume \
    --n-trials 50
```

### 4. Analyze Completed Study

```bash
python scripts/optuna_hpo.py \
    --visualize-only \
    --study-name production-hpo \
    --storage sqlite:///production_hpo.db \
    --viz-dir analysis_plots \
    --export-json final_results.json \
    --save-best-config final_best_config.json
```

### 5. Aggressive Pruning for Fast Results

```bash
python scripts/optuna_hpo.py \
    --n-trials 50 \
    --trial-epochs 15 \
    --pruner hyperband \
    --early-stopping \
    --patience 3
```

## Output

### Console Output

The script prints:
1. Study configuration
2. Progress bar during optimization
3. Trial results (value, pruned/completed status)
4. Final statistics (completed, pruned, failed trials)
5. Best trial number and value
6. Best hyperparameters
7. Top 5 trials

### Visualization Files

When `--viz-dir` is specified, the following HTML plots are generated:

- **optimization_history.html**: Objective value vs. trial number
- **param_importances.html**: Hyperparameter importance ranking
- **parallel_coordinate.html**: Multi-dimensional parameter relationships
- **slice.html**: Individual parameter effects on objective
- **contour_lr_hidden.html**: 2D contour of learning rate vs. hidden dim

### JSON Export

When `--export-json` is specified, a JSON file contains:

```json
{
  "study_name": "airfrans-gnn-hpo",
  "n_trials": 50,
  "best_value": 0.012345,
  "best_params": { ... },
  "best_trial_number": 23,
  "trials": [
    {
      "number": 0,
      "state": "COMPLETE",
      "value": 0.023456,
      "params": { ... },
      "datetime_start": "2026-02-04T12:00:00",
      "datetime_complete": "2026-02-04T12:15:00",
      "duration": 900.0
    },
    ...
  ]
}
```

### Best Config JSON

When `--save-best-config` is specified, a JSON file contains the `SmokeCfg` dataclass with optimal hyperparameters ready for full training.

## Tips and Best Practices

### GPU Memory Management

1. **Use batch_size=1 or 2** for large models to avoid OOM
2. **Enable --gc-after-trial** (default) to free memory between trials
3. **Use --amp** for mixed precision training (reduces memory)
4. **Monitor GPU usage**: `watch -n 1 nvidia-smi`

### Optimization Strategy

1. **Start small**: Run 10-20 trials to verify the setup
2. **Use pruning**: MedianPruner or HyperbandPruner eliminate unpromising trials early
3. **Enable early stopping**: Saves time on trials that plateau
4. **Persistent storage**: Always use `--storage` for long runs
5. **Parallel jobs**: Only use `--n-jobs > 1` for CPU-only training

### Debugging

If trials fail:
1. Check data loading: Verify `--root` path and prebuilt graphs exist
2. Check GPU memory: Reduce `--limit-train` or `--trial-epochs`
3. Review first trial: The baseline trial may reveal issues
4. Use `--n-trials 1` to test a single trial

### Interpreting Results

- **Low variance in objective**: Search space may be too narrow
- **High pruning rate**: Increase `--pruner-startup-trials` or `--pruner-warmup-steps`
- **No improvement over baseline**: Expand search space or check data quality

## Integration with Training Pipeline

After finding the best hyperparameters:

```bash
# 1. Run HPO
python scripts/optuna_hpo.py \
    --study-name my-hpo \
    --storage sqlite:///hpo.db \
    --n-trials 50 \
    --save-best-config best_config.json

# 2. Use best config for full training
# (Option A: Manually copy parameters to scripts/train.py arguments)
python scripts/train.py \
    --hidden 256 \
    --layers 10 \
    --lr 3e-4 \
    --batch-size 2 \
    --epochs 200 \
    --continuity-target-weight 0.15 \
    --momentum-target-weight 0.15

# (Option B: Load JSON config programmatically in a custom script)
```

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution**:
- Reduce `--limit-train` and `--limit-val`
- Use `--batch-size` suggestion range [1, 2] only
- Enable `--amp`

### Issue: "No improvement over random search"

**Solution**:
- Increase `--n-trials` (50-100+)
- Check if baseline trial completes successfully
- Verify data preprocessing is correct

### Issue: "Study not found" on resume

**Solution**:
- Ensure `--storage` and `--study-name` match original run
- Check database file exists

### Issue: "All trials pruned"

**Solution**:
- Increase `--pruner-startup-trials` to 10
- Use `--pruner patient` or `--pruner none`
- Increase `--trial-epochs`

## Advanced: Custom Search Space

To customize the search space, edit the `objective()` function in `optuna_hpo.py`:

```python
# Example: Add new hyperparameter
gradient_clip_norm = trial.suggest_float('gradient_clip_norm', 0.5, 5.0)

# Example: Change existing range
lr = trial.suggest_float('lr', 1e-6, 5e-3, log=True)
```

## Performance Benchmarks

Typical runtime (NVIDIA RTX 3090, batch_size=2, trial_epochs=20):

- **Per trial**: 3-5 minutes
- **50 trials**: ~3-4 hours (with pruning)
- **100 trials**: ~6-8 hours (with pruning)

Pruning can reduce total time by 30-50% by eliminating poor trials early.

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Optuna Tutorials](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
- [AirfRANS Dataset](https://github.com/Extrality/AirfRANS)

## See Also

- `scripts/train.py` - Main training script
- `scripts/train_multiscale.py` - Multi-scale model training
- `src/training_common.py` - Shared training utilities
- `02_optuna_training.ipynb` - Original notebook
