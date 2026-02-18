#!/usr/bin/env python3
"""
scripts/train.py — Thin entry point for AirfRANS GNN training.

All logic has been factored into src/ modules; this file only contains main().

Usage:
    python scripts/train.py [OPTIONS]
    python scripts/train.py --help

Examples:
    # Basic training with default settings
    python scripts/train.py

    # Train with custom hyperparameters
    python scripts/train.py --batch-size 4 --epochs 200 --lr 3e-4 --hidden 256

    # Train with different task and scheduler
    python scripts/train.py --task full --lr-scheduler cosine --cosine-T-max 100

    # Disable global context tokens
    python scripts/train.py --no-global-tokens

    # Train with custom physics loss weights
    python scripts/train.py --continuity-target-weight 0.3 --momentum-target-weight 0.3
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data import load_and_prepare_data, collate_pyg
from src.training import create_lr_scheduler, init_wandb, set_seed
from src.config import parse_args, create_config_from_args
from src.ddp_utils import (
    _is_ddp,
    _ddp_rank,
    _ddp_local_rank,
    _ddp_world_size,
    _is_main_process,
    setup_ddp,
    cleanup_ddp,
    _unwrap_model,
)
from src.training import train_with_scheduler
from src.benchmark import run_benchmark_and_log_experiment
from src.diagnostics import plot_inlet_bc_velocity
from src.prediction import predict_one_for_viz, evaluate_model
from src.visualization import plot_pred_vs_gt
from src.physics_loss import NavierStokesPhysicsLoss
from src.global_context_processor import EnhancedCFDModelWithGlobalContext


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    from dotenv import load_dotenv
    load_dotenv()

    # Parse command line arguments
    args = parse_args()
    scfg = create_config_from_args(args)

    # --- DDP setup ---
    use_ddp = _is_ddp()
    if use_ddp:
        setup_ddp()
        device = torch.device(f'cuda:{_ddp_local_rank()}')
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rank = _ddp_rank()
    is_main = _is_main_process()

    set_seed(scfg.seed)

    if is_main:
        print('=' * 80)
        print('AirfRANS GNN Training')
        print('=' * 80)
        print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {device}')
        if use_ddp:
            print(f'DDP: world_size={_ddp_world_size()}, rank={rank}, local_rank={_ddp_local_rank()}')
        print(f'Task: {scfg.task}')
        print(f'Batch size: {scfg.batch_size} | Epochs: {scfg.epochs} | LR: {scfg.lr}')
        print(f'Hidden: {scfg.hidden} | Layers: {scfg.layers}')
        print(f'Global tokens: {scfg.use_global_tokens} (num={scfg.num_global_tokens})')
        print(f'LR scheduler: {scfg.lr_scheduler}')
        print('=' * 80)
    training_start = time.time()

    # --- Data ---
    data_bundle = load_and_prepare_data(scfg)

    # Override train DataLoader with DistributedSampler for DDP
    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            data_bundle.train_norm,
            num_replicas=_ddp_world_size(),
            rank=rank,
            shuffle=True,
            seed=scfg.seed,
        )
        data_bundle.train_loader = DataLoader(
            data_bundle.train_norm,
            batch_size=scfg.batch_size,
            sampler=train_sampler,
            num_workers=0,
            collate_fn=collate_pyg,
        )

    # --- Model ---
    node_dim = 7
    edge_dim = data_bundle.edge_dim

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = EnhancedCFDModelWithGlobalContext(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        hidden_dim=scfg.hidden,
        output_dim=4,
        num_mp_layers=scfg.layers,
        dropout_p=scfg.dropout,
        config=scfg,
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[_ddp_local_rank()], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=scfg.lr,
        weight_decay=scfg.weight_decay, betas=scfg.betas, eps=scfg.eps,
    )
    if is_main:
        print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # --- Physics loss ---
    steps_per_epoch = len(data_bundle.train_loader)
    loss_fn = NavierStokesPhysicsLoss(
        data_loss_weight=scfg.data_loss_weight,
        continuity_loss_weight=scfg.continuity_loss_weight,
        continuity_target_weight=scfg.continuity_target_weight,
        momentum_loss_weight=scfg.momentum_loss_weight,
        momentum_target_weight=scfg.momentum_target_weight,
        ramp_mode=scfg.ramp_mode,
        # Per-component ramp schedule (epoch → step conversion)
        cont_ramp_start_step=scfg.cont_ramp_start_epoch * steps_per_epoch,
        cont_curriculum_ramp_steps=scfg.cont_ramp_epochs * steps_per_epoch,
        mom_ramp_start_step=scfg.mom_ramp_start_epoch * steps_per_epoch,
        mom_curriculum_ramp_steps=scfg.mom_ramp_epochs * steps_per_epoch,
        bc_ramp_start_step=scfg.bc_ramp_start_epoch * steps_per_epoch,
        bc_curriculum_ramp_steps=scfg.bc_ramp_epochs * steps_per_epoch,
        bc_loss_weight=scfg.bc_loss_weight,
        chord_length=scfg.chord_length,
        dynamic_uref_from_data=scfg.dynamic_uref_from_data,
        dynamic_re_from_data=scfg.dynamic_re_from_data,
        nu_molecular=scfg.nu_molecular,
        uinf_from=scfg.uinf_from,
        use_huber_for_physics=scfg.use_huber_for_physics,
        huber_delta=scfg.huber_delta,
        use_perimeter_norm_for_div=scfg.use_perimeter_norm_for_div,
        div_area_floor_factor=scfg.div_area_floor_factor,
        div_min_degree=scfg.div_min_degree,
        debug=scfg.physics_debug,
        debug_level=scfg.physics_debug_level,
        debug_every=scfg.physics_debug_every,
    )
    if is_main:
        _cont_start = scfg.cont_ramp_start_epoch if scfg.cont_ramp_start_epoch >= 0 else scfg.ramp_start_epoch
        _mom_start  = scfg.mom_ramp_start_epoch  if scfg.mom_ramp_start_epoch  >= 0 else scfg.ramp_start_epoch
        _bc_start   = scfg.bc_ramp_start_epoch   if scfg.bc_ramp_start_epoch   >= 0 else scfg.ramp_start_epoch
        _cont_dur   = scfg.cont_ramp_epochs if scfg.cont_ramp_epochs >= 0 else scfg.ramp_epochs
        _mom_dur    = scfg.mom_ramp_epochs  if scfg.mom_ramp_epochs  >= 0 else scfg.ramp_epochs
        _bc_dur     = scfg.bc_ramp_epochs   if scfg.bc_ramp_epochs   >= 0 else scfg.ramp_epochs
        print(f"Physics loss initialized: cont {scfg.continuity_loss_weight:.3f}->{scfg.continuity_target_weight:.3f} (ep {_cont_start}+{_cont_dur}), "
              f"mom {scfg.momentum_loss_weight:.3f}->{scfg.momentum_target_weight:.3f} (ep {_mom_start}+{_mom_dur}), "
              f"bc {scfg.bc_loss_weight:.3f} (ep {_bc_start}+{_bc_dur})")

    # --- LR Scheduler ---
    # 이부분에서 심플하게 어떤 scheduler를 쓸지 간단하게 선택할 수 있도록 만들어보자. 예를 들어 cosine, step, plateau 등등.
    lr_scheduler = create_lr_scheduler(optimizer, scfg)

    # --- W&B (rank 0 only) ---
    if not is_main:
        scfg.wandb_mode = "disabled"
        scfg.enable_wandb = False
    init_wandb(scfg, loss_fn)

    # --- Train ---
    train_summary = train_with_scheduler(
        model, optimizer, lr_scheduler,
        data_bundle.train_loader, data_bundle.val_loader,
        scfg, device, loss_fn,
        train_sampler=train_sampler,
        is_main=is_main,
    )
    training_duration = time.time() - training_start

    # --- Post-training: evaluate, visualize, benchmark (rank 0 only) ---
    if is_main:
        eval_model = _unwrap_model(model)

        evaluate_model(eval_model, device, data_bundle, scfg)

        try:
            plot_inlet_bc_velocity(eval_model, data_bundle, device, scfg)
        except Exception as exc:
            print(f"[inlet BC plot] Failed: {exc}")

        if not args.no_viz:
            val_edges = data_bundle.val_graphs
            train_edges = data_bundle.train_graphs
            d_vis = val_edges[0] if len(val_edges) > 0 else (train_edges[0] if len(train_edges) > 0 else None)
            if d_vis is not None:
                dm_vis_n, y_pred_vis_n = predict_one_for_viz(
                    d_vis, eval_model, data_bundle.x_scaler, data_bundle.y_scaler, device)
                plot_pred_vs_gt(
                    dm_vis_n, y_pred_vis_n,
                    channel=2, show_mesh=False, denormalize=True,
                    y_scaler=data_bundle.y_scaler, x_scaler=data_bundle.x_scaler,
                )

        run_benchmark_and_log_experiment(
            model=eval_model,
            data_bundle=data_bundle,
            scfg=scfg,
            device=device,
            train_summary=train_summary,
            training_duration_sec=training_duration,
            notes=f"task={scfg.task}, hidden={scfg.hidden}, layers={scfg.layers}, lr={scfg.lr}, scheduler={scfg.lr_scheduler}",
        )

        print('\n' + '=' * 80)
        print('Training complete!')
        print('=' * 80)
        print(f'Best checkpoint saved to: {os.path.join(scfg.ckpt_dir, "best.pt")}')

    # --- DDP cleanup ---
    if use_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    main()
