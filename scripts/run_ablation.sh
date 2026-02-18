#!/bin/bash
# ==============================================================================
# Physics Loss Ablation Study
#
# Runs 4 experiments with progressively more physics loss terms:
#   1) MSE only
#   2) MSE + Momentum
#   3) MSE + Momentum + Continuity
#   4) MSE + Momentum + Continuity + BC
#
# Each physics term targets ~10% of total loss (target_weight=0.1, data=1.0).
# Ramp schedule from train_default.json: start epoch 50, ramp over 30 epochs.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train.py"

# Shared args — edit these to your needs
COMMON_ARGS=(
  --no-viz
  --epochs 200
)

# Uncomment to enable W&B logging:
# COMMON_ARGS+=(--enable-wandb --wandb-project airfrans-gnn)

echo "========================================================"
echo " Physics Loss Ablation Study (4 runs)"
echo "========================================================"

# --------------------------------------------------------------------------
# 1) MSE only — all physics losses disabled
# --------------------------------------------------------------------------
echo ""
echo "[1/4] MSE only"
echo "--------------------------------------------------------"
python "$TRAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  --wandb-name "ablation-1-mse-only" \
  --ckpt-dir checkpoints/ablation_1_mse_only \
  --data-loss-weight 1.0 \
  --continuity-loss-weight 0 --continuity-target-weight 0 \
  --momentum-loss-weight 0  --momentum-target-weight 0 \
  --bc-loss-weight 0

# --------------------------------------------------------------------------
# 2) MSE + Momentum loss
#    momentum: 0 → 0.1 over epochs 50–80 (linear ramp)
# --------------------------------------------------------------------------
echo ""
echo "[2/4] MSE + Momentum"
echo "--------------------------------------------------------"
python "$TRAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  --wandb-name "ablation-2-mse-mom" \
  --ckpt-dir checkpoints/ablation_2_mse_mom \
  --data-loss-weight 1.0 \
  --continuity-loss-weight 0 --continuity-target-weight 0 \
  --momentum-loss-weight 0  --momentum-target-weight 0.1 \
  --bc-loss-weight 0 \
  --mom-ramp-start-epoch 50 --mom-ramp-epochs 30

# --------------------------------------------------------------------------
# 3) MSE + Momentum + Continuity
#    momentum:   0 → 0.1 over epochs 50–80
#    continuity: 0 → 0.1 over epochs 50–80
# --------------------------------------------------------------------------
echo ""
echo "[3/4] MSE + Momentum + Continuity"
echo "--------------------------------------------------------"
python "$TRAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  --wandb-name "ablation-3-mse-mom-cont" \
  --ckpt-dir checkpoints/ablation_3_mse_mom_cont \
  --data-loss-weight 1.0 \
  --continuity-loss-weight 0 --continuity-target-weight 0.1 \
  --momentum-loss-weight 0  --momentum-target-weight 0.1 \
  --bc-loss-weight 0 \
  --cont-ramp-start-epoch 50 --cont-ramp-epochs 30 \
  --mom-ramp-start-epoch 50  --mom-ramp-epochs 30

# --------------------------------------------------------------------------
# 4) MSE + Momentum + Continuity + BC
#    momentum:   0 → 0.1 over epochs 50–80
#    continuity: 0 → 0.1 over epochs 50–80
#    BC:         0 → 0.1 over epochs 50–80
# --------------------------------------------------------------------------
echo ""
echo "[4/4] MSE + Momentum + Continuity + BC"
echo "--------------------------------------------------------"
python "$TRAIN_SCRIPT" "${COMMON_ARGS[@]}" \
  --wandb-name "ablation-4-mse-mom-cont-bc" \
  --ckpt-dir checkpoints/ablation_4_mse_mom_cont_bc \
  --data-loss-weight 1.0 \
  --continuity-loss-weight 0 --continuity-target-weight 0.1 \
  --momentum-loss-weight 0  --momentum-target-weight 0.1 \
  --bc-loss-weight 0.1 \
  --cont-ramp-start-epoch 50 --cont-ramp-epochs 30 \
  --mom-ramp-start-epoch 50  --mom-ramp-epochs 30 \
  --bc-ramp-start-epoch 50   --bc-ramp-epochs 30

echo ""
echo "========================================================"
echo " All 4 ablation runs complete!"
echo " Checkpoints saved under checkpoints/ablation_*"
echo "========================================================"
