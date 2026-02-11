# Training Script Usage Examples

Quick reference for common training scenarios with `scripts/train.py`.

## Basic Examples

### 1. Default Training
```bash
python scripts/train.py
```
Uses all default settings from SmokeCfg:
- Task: scarce
- Batch size: 2
- Epochs: 100
- Hidden dim: 128
- Layers: 14
- LR: 4e-4

### 2. Quick Test Run
```bash
python scripts/train.py \
    --epochs 5 \
    --batch-size 1 \
    --no-viz \
    --wandb-mode disabled
```
Fast iteration for debugging.

### 3. Production Training
```bash
python scripts/train.py \
    --task full \
    --batch-size 4 \
    --epochs 200 \
    --hidden 256 \
    --layers 16 \
    --lr 3e-4 \
    --wandb-name production-run-v1 \
    --wandb-tags production baseline
```

## Architecture Variants

### 4. Larger Model
```bash
python scripts/train.py \
    --hidden 256 \
    --layers 20 \
    --num-global-tokens 4 \
    --attention-heads 4
```

### 5. No Global Context (Baseline)
```bash
python scripts/train.py \
    --no-global-tokens \
    --wandb-name baseline-no-global
```

### 6. Different Pooling Strategies
```bash
# Set2Set pooling
python scripts/train.py --global-pooling-type set2set

# Mean pooling
python scripts/train.py --global-pooling-type mean

# Max pooling
python scripts/train.py --global-pooling-type max
```

## Learning Rate Schedules

### 7. Cosine Annealing
```bash
python scripts/train.py \
    --lr-scheduler cosine \
    --cosine-T-max 150 \
    --cosine-eta-min 1e-7
```

### 8. Warm Restarts
```bash
python scripts/train.py \
    --lr-scheduler cosine_warm_restarts
```

### 9. Constant Learning Rate
```bash
python scripts/train.py --lr-scheduler none
```

## Physics Loss Tuning

### 10. Aggressive Physics Weights
```bash
python scripts/train.py \
    --continuity-target-weight 0.5 \
    --momentum-target-weight 0.5 \
    --bc-loss-weight 0.3 \
    --ramp-start-epoch 20 \
    --ramp-epochs 100
```

### 11. Early Physics Ramp
```bash
python scripts/train.py \
    --ramp-start-epoch 10 \
    --ramp-epochs 40 \
    --ramp-mode cosine
```

### 12. Data-Only Training (No Physics)
```bash
python scripts/train.py \
    --continuity-target-weight 0.0 \
    --momentum-target-weight 0.0 \
    --bc-loss-weight 0.0
```

## Hardware Optimization

### 13. Mixed Precision Training
```bash
python scripts/train.py --amp
```

### 14. Larger Batches
```bash
python scripts/train.py \
    --batch-size 8 \
    --lr 6e-4
```
Note: Scale LR with batch size.

## Weights & Biases Integration

### 15. Offline Training
```bash
python scripts/train.py --wandb-mode offline
```

### 16. Custom Project & Tags
```bash
python scripts/train.py \
    --wandb-project airfrans-ablation \
    --wandb-name exp-001-baseline \
    --wandb-tags ablation baseline no-physics
```

### 17. With Artifacts
```bash
python scripts/train.py \
    --use-wandb-artifacts \
    --wandb-name artifact-test
```

## Ablation Studies

### 18. Effect of Global Context
```bash
# Without global context
python scripts/train.py --no-global-tokens --wandb-name ablation-no-global

# With global context (default)
python scripts/train.py --wandb-name ablation-with-global
```

### 19. Effect of Attention Layers
```bash
# 1 layer
python scripts/train.py --attention-layers 1 --wandb-name attn-1layer

# 2 layers (default)
python scripts/train.py --attention-layers 2 --wandb-name attn-2layer

# 4 layers
python scripts/train.py --attention-layers 4 --wandb-name attn-4layer
```

### 20. Effect of Hidden Dimension
```bash
for hidden in 64 128 256 512; do
    python scripts/train.py \
        --hidden $hidden \
        --wandb-name ablation-hidden-$hidden
done
```

## Reproducibility

### 21. Fixed Seed
```bash
python scripts/train.py --seed 42
```

### 22. CPU-Only Training
```bash
python scripts/train.py --device cpu
```

### 23. Specific GPU
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --device cuda
```

## Complete Example (Recommended Setup)

```bash
python scripts/train.py \
    --task scarce \
    --batch-size 4 \
    --epochs 150 \
    --hidden 256 \
    --layers 16 \
    --lr 3e-4 \
    --weight-decay 1e-2 \
    --lr-scheduler cosine \
    --cosine-T-max 120 \
    --num-global-tokens 4 \
    --attention-heads 4 \
    --attention-layers 2 \
    --continuity-target-weight 0.25 \
    --momentum-target-weight 0.25 \
    --bc-loss-weight 0.15 \
    --ramp-start-epoch 30 \
    --ramp-epochs 80 \
    --ramp-mode cosine \
    --use-huber-physics \
    --huber-delta 0.05 \
    --amp \
    --wandb-name optimal-run-v1 \
    --wandb-tags production optimized \
    --seed 42
```

## Tips

1. **Monitor training**: Use W&B to track metrics in real-time
2. **Start small**: Use `--epochs 5` for quick iteration
3. **Scale gradually**: Increase model size and batch size together
4. **Physics loss**: Start with lower weights and ramp up
5. **Checkpoints**: Default saves best model and periodic checkpoints
6. **Visualization**: Use `--no-viz` for headless servers
7. **Reproducibility**: Always set `--seed` for reproducible experiments
