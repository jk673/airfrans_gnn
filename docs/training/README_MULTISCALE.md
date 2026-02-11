# Multi-Scale Training Scripts

This directory contains two training scripts for AirfRANS GNN:

## Scripts Overview

### 1. `train.py` - Standard Training
**Model**: `EnhancedCFDModelWithGlobalContext`
- Base message-passing GNN with global attention tokens
- Standard RANS physics loss (continuity, momentum, BC)
- Suitable for most training scenarios

**Usage**:
```bash
python scripts/train.py --batch-size 4 --epochs 200 --lr 3e-4
```

### 2. `train_multiscale.py` - Multi-Scale Training
**Model**: `UltraEnhancedCFDModel`
- Wraps base model + multi-scale graph convolutions
- Enhanced turbulence modeling physics loss
- Additional loss terms: turbulence production/dissipation, smoothness, wall functions
- Multi-scale convolutions for capturing multi-resolution features

**Usage**:
```bash
python scripts/train_multiscale.py --batch-size 4 --epochs 200 --lr 3e-4
```

## Key Differences

| Feature | train.py | train_multiscale.py |
|---------|----------|---------------------|
| Model | `EnhancedCFDModelWithGlobalContext` | `UltraEnhancedCFDModel` |
| Physics Loss | `NavierStokesPhysicsLoss` | `EnhancedPhysicsLoss` |
| Base Layers | 14 (default) | 7 (default) |
| Multi-Scale Layers | 0 | 3 (default) |
| Multi-Scale Scales | N/A | 3 (default) |
| Turbulence Terms | No | Yes (production, dissipation) |
| Smoothness Loss | No | Yes |
| Wall Function Loss | No | Yes |
| Checkpoint Dir | `checkpoints` | `checkpoints_multiscale` |
| W&B Project | `airfrans-gnn` | `airfrans-gnn-multiscale` |

## Multi-Scale Specific Arguments

```bash
--num-scales NUM_SCALES
    Number of scales in multi-scale convolutions (default: 3)

--num-multiscale-layers NUM_MULTISCALE_LAYERS
    Number of multi-scale conv layers (default: 3)

--use-spp / --no-spp
    Enable/disable spatial pyramid pooling (default: False)
    Usually disabled for node-level prediction tasks

--turbulence-loss-weight WEIGHT
    Weight for turbulence modeling loss (default: 0.05)

--rans-loss-weight WEIGHT
    Weight for RANS-specific loss terms (default: 0.05)

--smoothness-weight WEIGHT
    Weight for smoothness loss (default: 0.01)

--wall-function-weight WEIGHT
    Weight for wall function loss (default: 0.02)

--use-adaptive-weights
    Enable adaptive weighting for physics loss terms (default: False)
```

## Architecture Comparison

### Standard Model (train.py)
```
Input (7D node features + 5D edge features)
  ↓
Node/Edge Encoders
  ↓
14x Message-Passing Layers
  ↓
Global Context Processor (optional)
  - Attention-based global tokens
  - Cross-attention between local/global
  ↓
Output Decoder (4D: u, v, p, nu_t)
```

### Multi-Scale Model (train_multiscale.py)
```
Input (7D node features + 5D edge features)
  ↓
Base Model (EnhancedCFDModelWithGlobalContext)
  - 7x Message-Passing Layers
  - Global Context Processor
  ↓ (outputs 4D predictions)
  │
  └──> Enhanced Processing Branch:
       - Node Encoder → hidden_dim
       - 3x Multi-Scale Graph Convs (3 scales each)
       - Residual connections
       - Spatial Pyramid Pooling (optional)
       ↓
Combined Features (4D base + hidden_dim enhanced)
  ↓
Enhanced Output Head
  ↓
Final Output (4D: u, v, p, nu_t)
```

## Code Reuse Strategy

Both scripts maximize code reuse from shared modules:

**Shared from `src/training_common.py`**:
- `SmokeCfg` configuration dataclass
- `load_and_prepare_data()` - data loading pipeline
- `train_epoch()` - training loop
- `run_epoch()` - validation loop
- `create_lr_scheduler()` - LR scheduler factory
- `init_wandb()` - W&B initialization
- `StandardScaler`, `NormalizedDataset`

**Shared from `scripts/train.py`**:
- `train_with_scheduler()` - main training loop with checkpointing
- `evaluate_model()` - evaluation utilities
- `plot_pred_vs_gt()` - visualization
- Force coefficient computation

**Multi-Scale Specific**:
- `MultiScaleCfg` - extends `SmokeCfg` with multi-scale parameters
- Imports `UltraEnhancedCFDModel` instead of `EnhancedCFDModelWithGlobalContext`
- Imports `EnhancedPhysicsLoss` instead of `NavierStokesPhysicsLoss`
- Additional CLI arguments for turbulence modeling

## When to Use Which Script

**Use `train.py` when**:
- You want a simpler, faster model
- You don't need multi-scale features
- Standard RANS physics constraints are sufficient
- You're doing initial experiments or ablation studies

**Use `train_multiscale.py` when**:
- You need multi-resolution feature extraction
- You want enhanced turbulence modeling
- You need additional physics constraints (smoothness, wall functions)
- You're willing to trade computational cost for potentially better accuracy

## Example Commands

### Standard Training
```bash
# Quick test
python scripts/train.py --epochs 10 --batch-size 2 --wandb-mode disabled

# Full training
python scripts/train.py --epochs 100 --batch-size 4 --lr 4e-4 \
    --continuity-target-weight 0.2 --momentum-target-weight 0.2 \
    --wandb-project my-project --wandb-name baseline-run

# With custom scheduler
python scripts/train.py --lr-scheduler cosine --cosine-T-max 100 \
    --ramp-start-epoch 30 --ramp-epochs 80
```

### Multi-Scale Training
```bash
# Quick test
python scripts/train_multiscale.py --epochs 10 --batch-size 2 --wandb-mode disabled

# Full training with custom scales
python scripts/train_multiscale.py --epochs 100 --batch-size 4 --lr 4e-4 \
    --num-scales 4 --num-multiscale-layers 4 \
    --turbulence-production-weight 0.02 \
    --wandb-project my-project --wandb-name multiscale-run

# Ablation: disable turbulence terms
python scripts/train_multiscale.py \
    --turbulence-loss-weight 0.0 \
    --rans-loss-weight 0.0 \
    --smoothness-weight 0.0 \
    --wall-function-weight 0.0

# Enable adaptive weighting
python scripts/train_multiscale.py --use-adaptive-weights
```

## Implementation Notes

1. **Code Organization**: Both scripts follow the same structure (imports, config, CLI parsing, main) to maintain consistency

2. **Minimal Duplication**: The multi-scale script only defines differences (new config class, different model/loss imports, additional CLI args)

3. **Backward Compatibility**: All shared code remains unchanged, so improvements benefit both scripts

4. **W&B Organization**: Different default projects/tags help organize experiments by model variant

5. **Checkpoint Isolation**: Separate checkpoint directories prevent overwriting between variants
