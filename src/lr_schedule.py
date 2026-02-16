"""Utilities for learning-rate schedulers."""

import torch


def create_lr_scheduler(
    optimizer,
    config=None,
    *,
    lr_scheduler=None,
    cosine_T_max=80,
    cosine_eta_min=1e-6,
    wr_T_0=10,
    wr_T_mult=1,
    wr_eta_min=1e-6,
    rop_factor=0.1,
    rop_patience=10,
    rop_min_lr=1e-7,
):
    """
    Create a learning-rate scheduler.

    Args:
        optimizer: PyTorch optimizer instance.
        config: Optional config-like object with scheduler attributes.
        lr_scheduler: Optional override for scheduler name.
        cosine_T_max, cosine_eta_min: CosineAnnealingLR params.
        wr_T_0, wr_T_mult, wr_eta_min: CosineAnnealingWarmRestarts params.
        rop_factor, rop_patience, rop_min_lr: ReduceLROnPlateau params.
    """
    if config is not None:
        if lr_scheduler is None:
            lr_scheduler = getattr(config, "lr_scheduler", None)
        cosine_T_max = getattr(config, "cosine_T_max", cosine_T_max)
        cosine_eta_min = getattr(config, "cosine_eta_min", cosine_eta_min)
        wr_T_0 = getattr(config, "wr_T_0", wr_T_0)
        wr_T_mult = getattr(config, "wr_T_mult", wr_T_mult)
        wr_eta_min = getattr(config, "wr_eta_min", wr_eta_min)
        rop_factor = getattr(config, "rop_factor", rop_factor)
        rop_patience = getattr(config, "rop_patience", rop_patience)
        rop_min_lr = getattr(config, "rop_min_lr", rop_min_lr)

    if lr_scheduler is None or lr_scheduler in ("", "none", "None"):
        print("🚫 Learning rate scheduler: None (constant LR)")
        return None

    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_T_max,
            eta_min=cosine_eta_min,
        )
        print("📊 Learning rate scheduler: CosineAnnealingLR")
        print(f"   T_max: {cosine_T_max}, eta_min: {cosine_eta_min}")
        return scheduler

    if lr_scheduler == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=wr_T_0,
            T_mult=wr_T_mult,
            eta_min=wr_eta_min,
        )
        print("🔄 Learning rate scheduler: CosineAnnealingWarmRestarts")
        print(f"   T_0: {wr_T_0}, T_mult: {wr_T_mult}, eta_min: {wr_eta_min}")
        return scheduler

    if lr_scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=rop_factor,
            patience=rop_patience,
            min_lr=rop_min_lr,
        )
        print("📉 Learning rate scheduler: ReduceLROnPlateau")
        print(f"   factor: {rop_factor}, patience: {rop_patience}, min_lr: {rop_min_lr}")
        return scheduler

    print(f"❌ Unknown scheduler: {lr_scheduler}, using None")
    return None


def simulate_lr_schedule(config, create_lr_scheduler_fn, num_epochs=20):
    """
    Simulate learning-rate scheduler progression across epochs.

    Args:
        config: Configuration object.
        create_lr_scheduler_fn: Function to create scheduler.
        num_epochs: Number of epochs to simulate.
    """
    temp_param = torch.nn.Parameter(torch.randn(1))
    temp_opt = torch.optim.AdamW([temp_param], lr=config.lr)
    temp_scheduler = create_lr_scheduler_fn(temp_opt, config)

    lrs = []
    val_losses = [1.0, 0.9, 0.85, 0.8, 0.85, 0.82, 0.81, 0.80, 0.82, 0.79,
                  0.78, 0.77, 0.78, 0.76, 0.75, 0.76, 0.74, 0.73, 0.74, 0.72]

    for epoch in range(num_epochs):
        lrs.append(temp_opt.param_groups[0]["lr"])

        if temp_scheduler is not None:
            if config.lr_scheduler == "reduce_on_plateau":
                val_loss = val_losses[epoch] if epoch < len(val_losses) else val_losses[-1]
                temp_scheduler.step(val_loss)
            else:
                temp_scheduler.step()

    return lrs
