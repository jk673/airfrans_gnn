# Optuna HPO Quick Reference

## One-Line Commands

### Quick Test
```bash
python scripts/optuna_hpo.py --n-trials 5 --trial-epochs 5 --limit-train 50 --limit-val 5
```

### Standard Run
```bash
python scripts/optuna_hpo.py --study-name my-hpo --storage sqlite:///hpo.db --n-trials 50 --viz-dir viz
```

### Resume Study
```bash
python scripts/optuna_hpo.py --study-name my-hpo --storage sqlite:///hpo.db --resume --n-trials 30
```

### Visualize Only
```bash
python scripts/optuna_hpo.py --visualize-only --study-name my-hpo --storage sqlite:///hpo.db --viz-dir viz
```

---

## Common Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `--n-trials N` | Number of trials | `--n-trials 50` |
| `--trial-epochs N` | Epochs per trial | `--trial-epochs 20` |
| `--early-stopping` | Enable early stop | `--early-stopping --patience 5` |
| `--storage URL` | Database storage | `--storage sqlite:///hpo.db` |
| `--resume` | Resume study | `--resume` |
| `--visualize-only` | No optimization | `--visualize-only` |
| `--viz-dir DIR` | Save plots | `--viz-dir my_plots` |
| `--export-json FILE` | Export results | `--export-json results.json` |
| `--save-best-config FILE` | Save best params | `--save-best-config best.json` |
| `--amp` | Mixed precision | `--amp` |

---

## Pruner Strategies

| Strategy | When to Use | Command |
|----------|-------------|---------|
| `median` | Default, balanced | `--pruner median` |
| `hyperband` | Aggressive, fast | `--pruner hyperband` |
| `patient` | Conservative | `--pruner patient --patience 7` |
| `none` | No pruning | `--pruner none` |

---

## Sampler Strategies

| Strategy | When to Use | Command |
|----------|-------------|---------|
| `tpe` | Default, smart | `--sampler tpe` |
| `random` | Baseline comparison | `--sampler random` |

---

## Resource Management

### Low GPU Memory
```bash
--limit-train 100 --limit-val 10 --amp
```

### Fast Results
```bash
--n-trials 30 --trial-epochs 15 --early-stopping --pruner hyperband
```

### High Quality
```bash
--n-trials 100 --trial-epochs 25 --pruner patient --patience 7
```

---

## Output Files

| File | Content | Command |
|------|---------|---------|
| `*.db` | SQLite study database | `--storage sqlite:///hpo.db` |
| `results.json` | All trial results | `--export-json results.json` |
| `best.json` | Best config for training | `--save-best-config best.json` |
| `viz/*.html` | Visualization plots | `--viz-dir viz` |

---

## Hyperparameters Optimized

### Model
- hidden_dim: [64, 128, 256, 512]
- num_layers: [3-14]
- dropout: [0.0-0.5]

### Training
- lr: [1e-5, 1e-2]
- weight_decay: [1e-6, 1e-1]
- batch_size: [1, 2, 4, 8]

### Physics Loss
- continuity_weight: [0.001, 0.5]
- momentum_weight: [0.001, 0.5]
- bc_loss_weight: [0.001, 0.3]

### Attention
- use_global_tokens: [True, False]
- num_global_tokens: [2, 4, 8]
- attention_heads: [2, 4, 8]
- attention_layers: [2-8]

---

## Typical Runtimes

| Configuration | Time per Trial | Total (50 trials) |
|---------------|----------------|-------------------|
| Small (limit=100, epochs=10) | 1-2 min | 50-100 min |
| Medium (limit=180, epochs=20) | 3-5 min | 150-250 min |
| Large (limit=500, epochs=25) | 8-12 min | 400-600 min |

*With GPU, pruning enabled*

---

## Workflow

1. **Test setup**: `--n-trials 5 --trial-epochs 5`
2. **Quick search**: `--n-trials 30 --trial-epochs 15 --early-stopping`
3. **Deep search**: `--n-trials 100 --trial-epochs 20 --storage sqlite:///hpo.db`
4. **Analyze**: `--visualize-only --viz-dir viz --export-json results.json`
5. **Train**: Use best params with `scripts/train.py`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error | `--limit-train 100 --amp` |
| All trials pruned | `--pruner-startup-trials 10` |
| Study not found | Check `--storage` and `--study-name` |
| Slow trials | `--early-stopping --patience 3` |
| No improvement | Increase `--n-trials` |

---

## Advanced

### Compare Strategies
```bash
# TPE + Median
python scripts/optuna_hpo.py --study-name tpe-median --storage sqlite:///cmp.db --sampler tpe --pruner median --n-trials 50

# Random + Hyperband
python scripts/optuna_hpo.py --study-name rand-hband --storage sqlite:///cmp.db --sampler random --pruner hyperband --n-trials 50
```

### Multi-Study Analysis
```bash
for study in study1 study2 study3; do
    python scripts/optuna_hpo.py --visualize-only --study-name $study --storage sqlite:///multi.db --viz-dir viz_$study
done
```

---

## Best Practices

1. Always use `--storage` for runs > 30 min
2. Start with `--n-trials 5` to test
3. Enable `--early-stopping` for efficiency
4. Save results: `--export-json` and `--save-best-config`
5. Visualize periodically during long runs
6. Use `--amp` on memory-constrained GPUs
7. Monitor with `watch nvidia-smi`

---

## Documentation

- Full docs: `scripts/README_OPTUNA_HPO.md`
- Examples: `scripts/EXAMPLES_OPTUNA.md`
- Help: `python scripts/optuna_hpo.py --help`
