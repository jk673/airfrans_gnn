# Optuna HPO Examples

Practical examples for running hyperparameter optimization with `optuna_hpo.py`.

## Example 1: Quick Test Run (5 trials)

Test the setup with a minimal run:

```bash
python scripts/optuna_hpo.py \
    --n-trials 5 \
    --trial-epochs 5 \
    --limit-train 50 \
    --limit-val 5
```

**Expected runtime**: ~5-10 minutes
**Use case**: Verify setup before longer runs

---

## Example 2: Standard HPO Run

Recommended configuration for most users:

```bash
python scripts/optuna_hpo.py \
    --study-name standard-hpo \
    --storage sqlite:///optuna_standard.db \
    --n-trials 50 \
    --trial-epochs 20 \
    --early-stopping \
    --patience 5 \
    --limit-train 180 \
    --limit-val 20 \
    --viz-dir viz_standard \
    --export-json standard_results.json \
    --save-best-config standard_best.json
```

**Expected runtime**: ~3-4 hours (with GPU)
**Output**:
- SQLite database: `optuna_standard.db`
- Visualizations: `viz_standard/*.html`
- Results: `standard_results.json`
- Best config: `standard_best.json`

---

## Example 3: Aggressive Pruning (Fast Results)

Get quick results with aggressive trial pruning:

```bash
python scripts/optuna_hpo.py \
    --study-name fast-hpo \
    --storage sqlite:///optuna_fast.db \
    --n-trials 30 \
    --trial-epochs 15 \
    --pruner hyperband \
    --early-stopping \
    --patience 3 \
    --limit-train 150 \
    --limit-val 15
```

**Expected runtime**: ~1.5-2 hours
**Use case**: Quick exploration before detailed search

---

## Example 4: Resume Interrupted Study

Continue an interrupted optimization run:

```bash
# Original run (interrupted)
python scripts/optuna_hpo.py \
    --study-name my-hpo \
    --storage sqlite:///my_hpo.db \
    --n-trials 100

# Resume from where it stopped
python scripts/optuna_hpo.py \
    --study-name my-hpo \
    --storage sqlite:///my_hpo.db \
    --resume \
    --n-trials 50
```

**Note**: Total trials = original + resume trials

---

## Example 5: Analyze Existing Study

Generate visualizations without running new trials:

```bash
python scripts/optuna_hpo.py \
    --visualize-only \
    --study-name standard-hpo \
    --storage sqlite:///optuna_standard.db \
    --viz-dir analysis_plots \
    --export-json final_analysis.json \
    --save-best-config final_best_config.json
```

**Output**:
- 5 HTML visualization files in `analysis_plots/`
- Complete results in `final_analysis.json`
- Best configuration in `final_best_config.json`

---

## Example 6: Minimal Resource Usage

Optimize on limited GPU memory:

```bash
python scripts/optuna_hpo.py \
    --n-trials 40 \
    --trial-epochs 15 \
    --limit-train 100 \
    --limit-val 10 \
    --early-stopping \
    --patience 5 \
    --amp
```

**Tips**:
- `--amp`: Mixed precision reduces memory by ~40%
- Small train/val sets avoid OOM errors
- Trials suggest batch_size in [1, 2] automatically

---

## Example 7: Patient Pruner (Conservative)

Use patient pruner to avoid premature trial termination:

```bash
python scripts/optuna_hpo.py \
    --study-name patient-hpo \
    --storage sqlite:///optuna_patient.db \
    --n-trials 60 \
    --trial-epochs 25 \
    --pruner patient \
    --patience 7 \
    --pruner-startup-trials 10 \
    --pruner-warmup-steps 8
```

**Use case**: When trials show high variance in validation loss

---

## Example 8: Random Sampler Baseline

Compare TPE sampler against random search:

```bash
# Random search
python scripts/optuna_hpo.py \
    --study-name random-baseline \
    --storage sqlite:///optuna_random.db \
    --sampler random \
    --n-trials 50 \
    --trial-epochs 20

# TPE search (default)
python scripts/optuna_hpo.py \
    --study-name tpe-search \
    --storage sqlite:///optuna_tpe.db \
    --sampler tpe \
    --n-trials 50 \
    --trial-epochs 20
```

**Analysis**: Compare best values to verify TPE effectiveness

---

## Example 9: Parallel Trials (CPU Only)

Run multiple trials in parallel (CPU training only):

```bash
python scripts/optuna_hpo.py \
    --study-name parallel-hpo \
    --storage sqlite:///optuna_parallel.db \
    --n-trials 50 \
    --n-jobs 4 \
    --device cpu \
    --trial-epochs 15
```

**Warning**: GPU doesn't support parallel trials due to memory constraints

---

## Example 10: Full Pipeline (HPO → Training)

Complete workflow from optimization to final training:

```bash
# Step 1: Run HPO
python scripts/optuna_hpo.py \
    --study-name production \
    --storage sqlite:///production_hpo.db \
    --n-trials 100 \
    --trial-epochs 20 \
    --early-stopping \
    --save-best-config best_params.json \
    --viz-dir production_viz

# Step 2: Train final model with best hyperparameters
# (Manually extract parameters from best_params.json or console output)
python scripts/train.py \
    --hidden 256 \
    --layers 10 \
    --lr 3.2e-4 \
    --weight-decay 0.008 \
    --batch-size 2 \
    --epochs 200 \
    --continuity-target-weight 0.15 \
    --momentum-target-weight 0.12 \
    --bc-loss-weight 0.08 \
    --use-global-tokens \
    --num-global-tokens 4 \
    --attention-heads 4 \
    --attention-layers 6 \
    --lr-scheduler cosine \
    --wandb-project production-training
```

---

## Example 11: Study Comparison

Compare multiple optimization strategies:

```bash
# Strategy 1: Default TPE + Median Pruner
python scripts/optuna_hpo.py \
    --study-name strategy1-tpe-median \
    --storage sqlite:///comparison.db \
    --sampler tpe \
    --pruner median \
    --n-trials 50

# Strategy 2: Random + Hyperband Pruner
python scripts/optuna_hpo.py \
    --study-name strategy2-random-hyperband \
    --storage sqlite:///comparison.db \
    --sampler random \
    --pruner hyperband \
    --n-trials 50

# Strategy 3: TPE + Patient Pruner
python scripts/optuna_hpo.py \
    --study-name strategy3-tpe-patient \
    --storage sqlite:///comparison.db \
    --sampler tpe \
    --pruner patient \
    --n-trials 50

# Analyze each
for study in strategy1-tpe-median strategy2-random-hyperband strategy3-tpe-patient; do
    python scripts/optuna_hpo.py \
        --visualize-only \
        --study-name $study \
        --storage sqlite:///comparison.db \
        --viz-dir viz_$study \
        --export-json results_$study.json
done
```

---

## Example 12: Custom Data Configuration

Optimize on different task or data subset:

```bash
# Scarce task (default)
python scripts/optuna_hpo.py \
    --task scarce \
    --root Dataset \
    --n-trials 50

# Full task
python scripts/optuna_hpo.py \
    --task full \
    --root Dataset \
    --limit-train 500 \
    --limit-val 50 \
    --n-trials 30
```

---

## Expected Outputs

### Console Output Example

```
================================================================================
AirfRANS GNN Hyperparameter Optimization with Optuna
================================================================================
PyTorch: 2.8.0+cu128 | CUDA: True | Device: cuda
Study: standard-hpo
Storage: sqlite:///optuna_standard.db
Trials: 50 | Trial epochs: 20
Pruner: median | Sampler: tpe
Task: scarce | Train graphs: 180 | Val graphs: 20
================================================================================

[I 2026-02-04 12:00:00,000] A new study created in RDB with name: standard-hpo

Trial 0:  100%|██████████| 20/20 [02:45<00:00,  8.25s/it]
[I 2026-02-04 12:02:45,000] Trial 0 finished with value: 0.0234 and parameters: {...}

Trial 1:  45%|████▌     | 9/20 [01:15<01:30,  8.23s/it]
[I 2026-02-04 12:04:00,000] Trial 1 pruned.

...

================================================================================
OPTIMIZATION COMPLETE!
================================================================================

Statistics:
  Completed trials: 35
  Pruned trials: 15
  Failed trials: 0
  Best trial: #23
  Best value: 0.012345

Best parameters:
  attention_heads                : 4
  attention_layers               : 6
  batch_size                     : 2
  bc_loss_weight                 : 0.0234
  ...
```

### Generated Files

After a typical run, you'll have:

```
optuna_standard.db              # SQLite database with all trials
standard_results.json           # JSON export of results
standard_best.json              # Best config for training
viz_standard/
├── optimization_history.html   # Objective value over trials
├── param_importances.html      # Most important hyperparameters
├── parallel_coordinate.html    # Multi-dimensional view
├── slice.html                  # Individual parameter effects
└── contour_lr_hidden.html      # 2D contour plot
```

---

## Tips for Success

1. **Start small**: Always test with `--n-trials 5 --trial-epochs 5` first
2. **Use storage**: Always specify `--storage` for runs longer than 30 minutes
3. **Monitor GPU**: Use `watch nvidia-smi` in another terminal
4. **Save results**: Use `--export-json` and `--save-best-config`
5. **Visualize often**: Run `--visualize-only` periodically during long runs
6. **Patience matters**: Good hyperparameters may not appear in first 10 trials

---

## Next Steps

After finding optimal hyperparameters:

1. **Verify**: Re-run best trial manually to confirm results
2. **Full training**: Use `scripts/train.py` with best parameters
3. **Ensemble**: Train multiple models with top-5 configurations
4. **Fine-tune**: Narrow search space around best parameters and re-run

---

## Troubleshooting Common Issues

### All trials fail immediately

```bash
# Debug with single trial and small data
python scripts/optuna_hpo.py \
    --n-trials 1 \
    --limit-train 10 \
    --limit-val 2
```

### Out of memory errors

```bash
# Reduce memory usage
python scripts/optuna_hpo.py \
    --limit-train 100 \
    --limit-val 10 \
    --amp
```

### No pruning happening

```bash
# Lower pruning thresholds
python scripts/optuna_hpo.py \
    --pruner median \
    --pruner-startup-trials 3 \
    --pruner-warmup-steps 3
```
