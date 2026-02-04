"""Training utilities for AirfRANS GNN."""

import os
import torch
import wandb
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def create_lr_scheduler(optimizer, config):
    """
    Configuration에 따라 적절한 LR scheduler를 생성합니다.

    Args:
        optimizer: PyTorch optimizer instance
        config: Configuration object with scheduler settings

    Returns:
        LR scheduler or None if no scheduler is configured
    """
    if config.lr_scheduler is None:
        print("🚫 Learning rate scheduler: None (constant LR)")
        return None

    elif config.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.cosine_T_max,
            eta_min=config.cosine_eta_min
        )
        print(f"📊 Learning rate scheduler: CosineAnnealingLR")
        print(f"   T_max: {config.cosine_T_max}, eta_min: {config.cosine_eta_min}")
        return scheduler

    elif config.lr_scheduler == 'cosine_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.wr_T_0,
            T_mult=config.wr_T_mult,
            eta_min=config.wr_eta_min
        )
        print(f"🔄 Learning rate scheduler: CosineAnnealingWarmRestarts")
        print(f"   T_0: {config.wr_T_0}, T_mult: {config.wr_T_mult}, eta_min: {config.wr_eta_min}")
        return scheduler

    elif config.lr_scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # validation loss를 minimize
            factor=config.rop_factor,
            patience=config.rop_patience,
            min_lr=config.rop_min_lr,
        )
        print(f"📉 Learning rate scheduler: ReduceLROnPlateau")
        print(f"   factor: {config.rop_factor}, patience: {config.rop_patience}, min_lr: {config.rop_min_lr}")
        return scheduler

    else:
        print(f"❌ Unknown scheduler: {config.lr_scheduler}, using None")
        return None


def simulate_lr_schedule(config, create_lr_scheduler_fn, num_epochs=20):
    """
    LR 스케줄 변화를 시뮬레이션합니다.

    Args:
        config: Configuration object with scheduler settings
        create_lr_scheduler_fn: Function to create scheduler
        num_epochs: Number of epochs to simulate

    Returns:
        List of learning rates for each epoch
    """
    # 임시 optimizer 생성
    temp_param = torch.nn.Parameter(torch.randn(1))
    temp_opt = torch.optim.AdamW([temp_param], lr=config.lr)
    temp_scheduler = create_lr_scheduler_fn(temp_opt, config)

    lrs = []
    val_losses = [1.0, 0.9, 0.85, 0.8, 0.85, 0.82, 0.81, 0.80, 0.82, 0.79,
                  0.78, 0.77, 0.78, 0.76, 0.75, 0.76, 0.74, 0.73, 0.74, 0.72]

    for epoch in range(num_epochs):
        lrs.append(temp_opt.param_groups[0]['lr'])

        if temp_scheduler is not None:
            if config.lr_scheduler == 'reduce_on_plateau':
                # ReduceLROnPlateau의 경우 validation loss 필요
                val_loss = val_losses[epoch] if epoch < len(val_losses) else val_losses[-1]
                temp_scheduler.step(val_loss)
            else:
                # 다른 스케줄러는 epoch만 필요
                temp_scheduler.step()

    return lrs


def train_with_scheduler(model, optim, scheduler, train_loader, val_loader,
                         scfg, device, scaler=None, physics_loss_fn=None,
                         train_epoch_fn=None, run_epoch_fn=None, get_lr_fn=None,
                         channel_names=None):
    """
    Enhanced training loop with learning rate scheduling and W&B artifact management.

    Args:
        model: PyTorch model to train
        optim: Optimizer instance
        scheduler: LR scheduler or None
        train_loader: Training data loader
        val_loader: Validation data loader
        scfg: Configuration object
        device: Device to train on (cpu/cuda)
        scaler: GradScaler for AMP or None
        physics_loss_fn: Physics loss function or None
        train_epoch_fn: Function to run training epoch
        run_epoch_fn: Function to run validation epoch
        get_lr_fn: Function to get current learning rate
        channel_names: List of channel names for logging

    Returns:
        Dictionary with training history and artifact statistics
    """
    scaler = GradScaler(enabled=(scfg.amp and torch.cuda.is_available()))
    global_step = 0
    best_val = float('inf')

    # Artifact 관리 설정
    USE_WANDB_ARTIFACTS = getattr(scfg, "use_wandb_artifacts", False)
    ARTIFACT_SAVE_BEST_ONLY = getattr(scfg, "artifact_save_best_only", True)
    ARTIFACT_SAVE_INTERVAL = getattr(scfg, "artifact_save_interval", 20)

    EPOCHS = getattr(scfg, "epochs", 50)
    ckpt_dir = getattr(scfg, "ckpt_dir", "checkpoints")
    ckpt_interval = max(1, getattr(scfg, "ckpt_interval", 5))

    # Artifact 히스토리 추적
    artifact_history = {
        'best_uploaded': False,
        'last_periodic_epoch': -1,
        'total_artifacts': 0
    }

    # Default channel names if not provided
    if channel_names is None:
        channel_names = []

    for epoch in range(EPOCHS):
        # Train epoch
        train_total, train_logs, global_step = train_epoch_fn(
            train_loader, model, optim, device, scaler,
            desc=f"train[{epoch}]", loss_fn=physics_loss_fn,
            global_step_start=global_step,
            scheduler=scheduler,
            scheduler_step_mode=("step" if getattr(scfg, "scheduler_step_per_batch", False) else "epoch"),
            log_every_n_steps=getattr(scfg, "log_every_n_steps", 25)
        )

        # Validation epoch
        val_total, val_logs = run_epoch_fn(val_loader, model, device, loss_fn=physics_loss_fn)

        # === wandb epoch-level logging ===
        log_epoch = {
            "epoch": epoch,
            "train/total_epoch": train_logs['total_loss'],
            "train/mse_epoch": train_logs['mse_loss'],
            "train/continuity_epoch": train_logs.get('continuity_loss', float('nan')),
            "train/momentum_epoch": train_logs.get('momentum_loss', float('nan')),
            "train/bc_epoch": train_logs.get('bc_loss', float('nan')),
            "val/total_epoch": val_logs['total_loss'],
            "val/mse_epoch": val_logs['mse_loss'],
            "val/continuity_epoch": val_logs.get('continuity_loss', float('nan')),
            "val/momentum_epoch": val_logs.get('momentum_loss', float('nan')),
            "val/bc_epoch": val_logs.get('bc_loss', float('nan')),
        }

        # Log per-channel MSE losses
        for cn in channel_names:
            log_epoch[f"train/mse_{cn}_epoch"] = train_logs.get(f'mse_{cn}', float('nan'))
            log_epoch[f"val/mse_{cn}_epoch"] = val_logs.get(f'mse_{cn}', float('nan'))

        # Log curriculum weights if available
        if 'cont_weight_used' in train_logs:
            log_epoch["weight/cont_used_epoch"] = train_logs['cont_weight_used']
        if 'mom_weight_used' in train_logs:
            log_epoch["weight/mom_used_epoch"] = train_logs['mom_weight_used']

        # Log learning rate
        lr_now = get_lr_fn(optim) if get_lr_fn else None
        if lr_now is not None:
            log_epoch["lr_epoch"] = lr_now

        wandb.log(log_epoch, step=global_step, commit=True)

        # === Learning Rate Scheduler Step ===
        if scheduler is not None and not getattr(scfg, "scheduler_step_per_batch", False):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_total)
            else:
                scheduler.step()

        # === Checkpoint 저장 (로컬 파일시스템) ===
        os.makedirs(ckpt_dir, exist_ok=True)

        # Best model 저장
        is_best = val_total < best_val
        if is_best:
            best_val = val_total
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                "scaler": (scaler.state_dict() if scaler is not None else None),
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
                "train_logs": train_logs,
                "val_logs": val_logs
            }, best_path)

            # W&B Artifact 업로드 (조건부)
            if USE_WANDB_ARTIFACTS:
                try:
                    art = wandb.Artifact(
                        name=f"model-best",
                        type="model",
                        description=f"Best model at epoch {epoch} with val_loss={val_total:.4f}",
                        metadata={
                            "epoch": epoch,
                            "val_loss": val_total,
                            "train_loss": train_total,
                            "best_val": best_val
                        }
                    )
                    art.add_file(best_path)
                    wandb.run.log_artifact(art)
                    artifact_history['best_uploaded'] = True
                    artifact_history['total_artifacts'] += 1
                    print(f"  📤 W&B Artifact uploaded: best model (epoch {epoch})")
                except Exception as e:
                    print(f"  ⚠️ Failed to upload W&B artifact: {e}")

        # Periodic checkpoint 저장
        if (epoch + 1) % ckpt_interval == 0:
            ep_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                "scaler": (scaler.state_dict() if scaler is not None else None),
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val
            }, ep_path)

            # Periodic artifact 업로드 (매우 제한적으로)
            if USE_WANDB_ARTIFACTS and not ARTIFACT_SAVE_BEST_ONLY:
                if (epoch + 1) % ARTIFACT_SAVE_INTERVAL == 0:
                    try:
                        art = wandb.Artifact(
                            name=f"model-checkpoint",
                            type="model",
                            description=f"Checkpoint at epoch {epoch+1}",
                            metadata={
                                "epoch": epoch + 1,
                                "val_loss": val_total,
                                "train_loss": train_total
                            }
                        )
                        art.add_file(ep_path)
                        wandb.run.log_artifact(art, aliases=[f"epoch-{epoch+1}"])
                        artifact_history['last_periodic_epoch'] = epoch + 1
                        artifact_history['total_artifacts'] += 1
                        print(f"  📤 W&B Artifact uploaded: checkpoint (epoch {epoch+1})")
                    except Exception as e:
                        print(f"  ⚠️ Failed to upload periodic artifact: {e}")

        # Print epoch summary
        print(f"Epoch {epoch:3d} | Train: total={train_total:.4f} mse={train_logs['mse_loss']:.4f} "
              f"cont={train_logs.get('continuity_loss', 0):.2e} mom={train_logs.get('momentum_loss', 0):.2e} "
              f"bc={train_logs.get('bc_loss', 0):.2e} | "
              f"Val: total={val_total:.4f} bc={val_logs.get('bc_loss', 0):.2e}"
              f" {'[BEST]' if is_best else ''}")

    # === Training 완료 후 최종 artifact ===
    if USE_WANDB_ARTIFACTS:
        # 최종 모델 저장
        final_path = os.path.join(ckpt_dir, "final.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": (scheduler.state_dict() if scheduler is not None else None),
            "scaler": (scaler.state_dict() if scaler is not None else None),
            "epoch": EPOCHS - 1,
            "global_step": global_step,
            "best_val": best_val
        }, final_path)

        try:
            art = wandb.Artifact(
                name=f"model-final",
                type="model",
                description=f"Final model after {EPOCHS} epochs",
                metadata={
                    "total_epochs": EPOCHS,
                    "best_val": best_val,
                    "total_artifacts": artifact_history['total_artifacts']
                }
            )
            art.add_file(final_path)
            art.add_file(os.path.join(ckpt_dir, "best.pt"), name="best.pt")
            wandb.run.log_artifact(art, aliases=["latest", "final"])
            print(f"\n📤 Final W&B Artifact uploaded with {artifact_history['total_artifacts']+1} total artifacts")
        except Exception as e:
            print(f"⚠️ Failed to upload final artifact: {e}")

    # 종료
    wandb.finish()

    return {
        'lr_history': [],
        'train_total_loss': [],
        'train_continuity_loss': [],
        'train_bc_loss': [],
        'val_total_loss': [],
        'val_continuity_loss': [],
        'val_bc_loss': [],
        'artifacts_uploaded': artifact_history['total_artifacts']
    }


def run_training_experiment(model, train_loader, val_loader, scfg, device, loss_fn,
                           train_epoch_fn, run_epoch_fn, get_lr_fn, channel_names,
                           config_updates=None, physics_config_updates=None):
    """
    설정을 업데이트하고 physics loss와 함께 훈련을 실행합니다.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        scfg: Configuration object
        device: Device to train on (cpu/cuda)
        loss_fn: Loss function (physics loss function)
        train_epoch_fn: Function to run training epoch
        run_epoch_fn: Function to run validation epoch
        get_lr_fn: Function to get current learning rate
        channel_names: List of channel names for logging
        config_updates: Dictionary of model configuration changes (예: {'lr_scheduler': 'cosine'})
        physics_config_updates: Dictionary of physics loss configuration changes

    Returns:
        Dictionary with training history
    """
    # 설정 업데이트
    if config_updates:
        for key, value in config_updates.items():
            setattr(scfg, key, value)
        print(f"🔧 Model configuration updated: {config_updates}")

    # 새로운 optimizer와 scheduler 생성
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scfg.lr,
        weight_decay=scfg.weight_decay,
        betas=scfg.betas,
        eps=scfg.eps
    )

    scheduler = create_lr_scheduler(optimizer, scfg)

    # GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler() if scfg.amp and torch.cuda.is_available() else None

    # 훈련 실행
    history = train_with_scheduler(
        model, optimizer, scheduler, train_loader, val_loader,
        scfg, device, scaler, physics_loss_fn=loss_fn,
        train_epoch_fn=train_epoch_fn, run_epoch_fn=run_epoch_fn,
        get_lr_fn=get_lr_fn, channel_names=channel_names
    )

    return history
