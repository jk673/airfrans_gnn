# Training Scripts

This directory contains training scripts for the AirfRANS GNN project.

## Main Training Script: `train.py`

The main training script with full CLI argument support, converted from the `01_trainer.ipynb` notebook.

### Quick Start

```bash
# Basic training with default settings
python scripts/train.py

# Show all available options
python scripts/train.py --help
```

### Common Usage Examples

#### 1. Custom Hyperparameters
```bash
python scripts/train.py \
    --batch-size 4 \
    --epochs 200 \
    --lr 3e-4 \
    --hidden 256 \
    --layers 16
```

#### 2. Different Task Variant
```bash
# Train on full dataset
python scripts/train.py --task full

# Train on scarce dataset (default)
python scripts/train.py --task scarce
```

#### 3. Custom Learning Rate Scheduler
```bash
# Cosine annealing (default)
python scripts/train.py --lr-scheduler cosine --cosine-T-max 100

# Cosine with warm restarts
python scripts/train.py --lr-scheduler cosine_warm_restarts

# Reduce on plateau
python scripts/train.py --lr-scheduler reduce_on_plateau

# No scheduler (constant LR)
python scripts/train.py --lr-scheduler none
```

#### 4. Global Context Configuration
```bash
# Disable global context tokens
python scripts/train.py --no-global-tokens

# Custom global context settings
python scripts/train.py \
    --num-global-tokens 4 \
    --attention-heads 4 \
    --attention-layers 3 \
    --global-pooling-type set2set
```

#### 5. Physics Loss Configuration
```bash
# Custom physics loss weights
python scripts/train.py \
    --continuity-target-weight 0.3 \
    --momentum-target-weight 0.3 \
    --bc-loss-weight 0.2

# Custom curriculum ramp schedule
python scripts/train.py \
    --ramp-start-epoch 20 \
    --ramp-epochs 80 \
    --ramp-mode cosine
```

#### 6. Weights & Biases Configuration
```bash
# Custom W&B project and run name
python scripts/train.py \
    --wandb-project my-airfrans-project \
    --wandb-name baseline-run-v1 \
    --wandb-tags baseline physics-loss

# Offline mode
python scripts/train.py --wandb-mode offline

# Disabled W&B logging
python scripts/train.py --wandb-mode disabled

# Enable artifact uploads
python scripts/train.py --use-wandb-artifacts
```

#### 7. Mixed Precision Training
```bash
# Enable automatic mixed precision (AMP)
python scripts/train.py --amp
```

#### 8. Quick Testing Run
```bash
# Fast run for testing
python scripts/train.py \
    --epochs 5 \
    --batch-size 1 \
    --no-viz \
    --wandb-mode disabled
```

### Key Configuration Options

#### Model Architecture
- `--hidden`: Hidden dimension size (default: 128)
- `--layers`: Number of message passing layers (default: 14)
- `--dropout`: Dropout probability (default: 0.1)

#### Global Context & Attention
- `--use-global-tokens` / `--no-global-tokens`: Enable/disable global context
- `--num-global-tokens`: Number of global tokens (default: 2)
- `--attention-heads`: Number of attention heads (default: 2)
- `--attention-layers`: Number of attention layers (default: 2)
- `--global-pooling-type`: Pooling mechanism (attention/mean/max/set2set)

#### Training
- `--batch-size`: Batch size (default: 2)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 4e-4)
- `--weight-decay`: AdamW weight decay (default: 1e-2)
- `--amp`: Enable automatic mixed precision

#### Physics Loss
- `--data-loss-weight`: Weight for MSE loss (default: 1.0)
- `--continuity-loss-weight`: Initial continuity weight (default: 0.05)
- `--continuity-target-weight`: Target continuity weight (default: 0.20)
- `--momentum-loss-weight`: Initial momentum weight (default: 0.05)
- `--momentum-target-weight`: Target momentum weight (default: 0.20)
- `--bc-loss-weight`: Boundary condition weight (default: 0.1)
- `--ramp-start-epoch`: When to start curriculum ramp (default: 40)
- `--ramp-epochs`: Duration of curriculum ramp (default: 60)
- `--ramp-mode`: Ramp schedule (linear/cosine)

#### Checkpointing
- `--ckpt-dir`: Checkpoint directory (default: checkpoints)
- `--ckpt-interval`: Save checkpoint every N epochs (default: 5)

### Output

The script will:
1. Load and prepare the data with normalization
2. Initialize the model with specified architecture
3. Create optimizer and learning rate scheduler
4. Initialize physics-informed loss function
5. Train for the specified number of epochs
6. Save checkpoints to `--ckpt-dir`
7. Log metrics to Weights & Biases
8. Evaluate the model on validation data
9. Generate visualization plots (unless `--no-viz` is specified)

### Checkpoints

Checkpoints are saved to the directory specified by `--ckpt-dir` (default: `checkpoints/`):
- `best.pt`: Best model based on validation loss
- `epoch_N.pt`: Periodic checkpoints every `--ckpt-interval` epochs

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Scaler state dict (if using AMP)
- Training metadata (epoch, global_step, best_val, etc.)

## Evaluation Script: `eval_cp.py`

Standalone evaluation script for computing Cp (pressure coefficient) relative L2 error from a trained checkpoint.

### Quick Start

```bash
# Evaluate on validation set with surface nodes only
python scripts/eval_cp.py --checkpoint checkpoints/best.pt --split val --surface-only

# Evaluate on test set with all nodes
python scripts/eval_cp.py --checkpoint checkpoints/best.pt --split test

# Show all available options
python scripts/eval_cp.py --help
```

### Usage Examples

#### 1. Basic Evaluation
```bash
# Evaluate on validation split (default)
python scripts/eval_cp.py --checkpoint checkpoints/best.pt

# Evaluate on test split
python scripts/eval_cp.py --checkpoint checkpoints/best.pt --split test
```

#### 2. Surface vs All Nodes
```bash
# Surface nodes only (recommended for Cp evaluation)
python scripts/eval_cp.py --checkpoint checkpoints/best.pt --surface-only

# All nodes (volume + surface)
python scripts/eval_cp.py --checkpoint checkpoints/best.pt
```

#### 3. Custom Model Configuration
```bash
# Must match the configuration used during training
python scripts/eval_cp.py \
    --checkpoint checkpoints/best.pt \
    --hidden 256 \
    --layers 16 \
    --task scarce
```

#### 4. Verbose Output
```bash
# Print per-graph results
python scripts/eval_cp.py \
    --checkpoint checkpoints/best.pt \
    --split val \
    --surface-only \
    --verbose
```

### Key Options

- `--checkpoint`: Path to checkpoint file (required)
- `--split`: Evaluation split ('val' or 'test', default: 'val')
- `--surface-only`: Evaluate only on surface nodes (default: False)
- `--batch-size`: Batch size for evaluation (default: 1)
- `--task`: Task name, must match training (default: 'scarce')
- `--hidden`: Hidden dimension, must match training (default: 128)
- `--layers`: Number of MP layers, must match training (default: 14)
- `--device`: Device to use ('auto', 'cuda', 'cpu', default: 'auto')
- `--verbose`: Print per-graph results

### Output

The script reports:
- Per-graph statistics (mean, median, std, min, max)
- Global relative L2 error (aggregated over all nodes)
- Number of successfully evaluated graphs

Example output:
```
======================================================================
RESULTS: Cp Relative L2 Error
======================================================================
Checkpoint:    checkpoints/best.pt
Split:         val
Surface only:  True
Graphs:        18 / 18
----------------------------------------------------------------------
Per-graph statistics:
  Mean:        0.0342
  Median:      0.0287
  Std:         0.0156
  Min:         0.0145
  Max:         0.0698
----------------------------------------------------------------------
Global relative L2 (aggregated over all nodes):
  0.0335
======================================================================
```

## Other Scripts

- `train_multi_scale.py`: Training script for multi-scale model variant
- `optuna_search.py`: Hyperparameter optimization using Optuna

## Notes

- Default configuration follows the best practices from `01_trainer.ipynb`
- All CLI arguments have sensible defaults and can be overridden as needed
- Use `--help` to see the full list of available options
- For reproducibility, fix the random seed with `--seed`
