"""
Prediction helper functions for AirfRANS GNN model inference.

This module provides utilities for making single-sample predictions with proper
normalization handling, surface/volume masking, and MSE computation across
different channel outputs (u, v, pressure, turbulent viscosity).

Functions:
    - predict_one_local: Single-sample prediction in normalized space (for evaluation metrics)
    - predict_one_for_viz: Single-sample prediction optimized for visualization
    - surface_volume_masks_from_orig: Extract surface and volume node masks
    - mse_per_channel: Compute per-channel MSE with optional masking

Notes:
    - All predictions are returned in normalized space for consistency with training metrics
    - Visualization functions can optionally denormalize for physical interpretation
    - Functions require model, x_scaler, y_scaler, device to be in scope when called
"""

import contextlib
import torch
from torch_geometric.data import Data

from src.utils import get_surface_mask, ensure_edge_features, with_pos2


@torch.no_grad()
def surface_volume_masks_from_orig(d: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract surface and volume node masks from a data object.

    Surface nodes are identified by non-zero wall normal vectors (components 3:5 of x).
    Volume nodes have zero normal vectors.

    Args:
        d (Data): PyG Data object with node features x of shape (N, >=5).
                  Assumes x[:, 3:5] contains wall normal components (nx, ny).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Boolean tensors
            - surf: (N,) bool tensor, True for surface nodes
            - vol: (N,) bool tensor, True for volume nodes
    """
    x = d.x
    if x is not None and x.size(1) >= 5:
        nxy = x[:, 3:5]
        surf = (nxy.abs().sum(dim=1) > 0)
    else:
        # Fallback to utility function
        surf = get_surface_mask(d)
    vol = ~surf
    return surf, vol


@torch.no_grad()
def predict_one_local(
    d: Data,
    model: torch.nn.Module,
    x_scaler,
    y_scaler,
    device: torch.device,
    amp_enabled: bool = False,
) -> tuple[Data, torch.Tensor]:
    """
    Make a prediction for a single sample in normalized space.

    This function ensures the input graph has proper node/edge features, normalizes
    node features, runs inference, and returns predictions in normalized space for
    evaluation metrics computation (MSE, physics loss evaluation, etc.).

    Workflow:
        1. Clone input data
        2. Ensure pos2 appended to node features (7D total)
        3. Validate and ensure edge features exist (5D)
        4. Normalize node features using x_scaler
        5. Forward pass on device with optional AMP
        6. Return normalized data and predictions on CPU

    Args:
        d (Data): Input PyG Data object with:
            - x: (N, 5) node features [vx, vy, wall_dist, nx, ny]
            - y: (N, 4) target values [u, v, p_over_rho, nu_t]
            - edge_index: (2, E) edge connectivity (must exist)
            - edge_attr: (E, 5) edge attributes or None (will be computed)
            - pos: (N, 3) node coordinates [x, y, 0] (optional, computed from x[:,:3] if missing)
        model: Trained GNN model in eval mode
        x_scaler: Fitted StandardScaler for node features
        y_scaler: Fitted StandardScaler for targets
        device: Computation device (torch.device)
        amp_enabled: Enable automatic mixed precision (default: False)

    Returns:
        tuple[Data, torch.Tensor]: Normalized data and predictions
            - dm_norm: Data object with normalized x, y; includes normalization parameters
            - y_pred_norm: (N, 4) predictions in normalized space on CPU

    Raises:
        AssertionError: If edge_index is missing from input graph

    Notes:
        - Both input and output are in normalized space for metric computation
        - Normalization parameters stored in dm_norm for potential denormalization
        - AMP context applied only if amp_enabled=True and CUDA available
    """
    dm = Data(**{k: v for k, v in d})

    # Ensure pos2 appended (7D node features)
    if dm.x.size(1) == 5 or not getattr(dm, "pos2_appended", False):
        dm = with_pos2(dm)

    # Validate edge existence
    assert hasattr(dm, "edge_index") and dm.edge_index is not None, "edge_index missing"

    # Ensure 5D edge features
    dm = ensure_edge_features(dm, want_dim=5)
    dm.x_orig = dm.x

    # Build normalized copy for evaluation
    dm_norm = Data(**{k: v for k, v in dm})
    dm_norm.x = x_scaler.transform(dm.x)
    dm_norm.y = y_scaler.transform(dm.y)
    dm_norm.x_orig = dm.x
    dm_norm.x_norm_params = {
        "mean": x_scaler.mean.clone(),
        "scale": x_scaler.std.clone(),
    }
    dm_norm.y_norm_params = {
        "mean": y_scaler.mean.clone(),
        "scale": y_scaler.std.clone(),
    }

    # Forward pass on device (clone to keep dm_norm on CPU)
    dm_run = dm_norm.clone().to(device)
    if amp_enabled and torch.cuda.is_available():
        with torch.amp.autocast(device_type="cuda"):
            y_pred_norm = model(dm_run).detach().cpu()
    else:
        y_pred_norm = model(dm_run).detach().cpu()

    return dm_norm, y_pred_norm


@torch.no_grad()
def predict_one_for_viz(
    d: Data,
    model: torch.nn.Module,
    x_scaler,
    y_scaler,
    device: torch.device,
) -> tuple[Data, torch.Tensor]:
    """
    Make a prediction for visualization with float32 enforcement.

    This function prepares a single graph for visualization, ensuring proper
    node/edge features, normalizing inputs, and returning predictions. Float32
    is explicitly used to avoid dtype conflicts from automatic mixed precision
    during model inference.

    Workflow:
        1. Clone input data
        2. Ensure pos2 appended to node features (7D)
        3. Validate and ensure edge features exist (5D)
        4. Normalize node features
        5. Move to device and ensure float32 tensors
        6. Forward pass with model.eval()
        7. Return normalized data and CPU predictions

    Args:
        d (Data): Input PyG Data object with same structure as predict_one_local
        model: Trained GNN model (will set to eval mode)
        x_scaler: Fitted StandardScaler for node features
        y_scaler: Fitted StandardScaler for targets
        device: Computation device (torch.device)

    Returns:
        tuple[Data, torch.Tensor]: Normalized data and predictions
            - dm_norm: Data object with normalized x, y
            - y_pred_norm: (N, 4) predictions in normalized space on CPU

    Notes:
        - Forces float32 for x and edge_attr to prevent AMP dtype issues
        - Suitable for visualization workflows; returns normalized predictions
        - Can be denormalized using y_scaler.inverse() for physical interpretation
    """
    dm = Data(**{k: v for k, v in d})

    # Ensure pos2 appended
    if dm.x.size(1) == 5 or not getattr(dm, "pos2_appended", False):
        dm = with_pos2(dm)

    # Validate and ensure edges
    assert hasattr(dm, "edge_index") and dm.edge_index is not None, "edge_index missing"
    dm = ensure_edge_features(dm, want_dim=5)

    # Normalize
    dm_norm = Data(**{k: v for k, v in dm})
    dm_norm.x = x_scaler.transform(dm.x)
    dm_norm.y = y_scaler.transform(dm.y)

    # Move to device with float32 enforcement
    dm_run = dm_norm.to(device)
    model.eval()

    # Force float32 to avoid AMP dtype conflicts
    if hasattr(dm_run, "x"):
        dm_run.x = dm_run.x.float()
    if hasattr(dm_run, "edge_attr"):
        dm_run.edge_attr = dm_run.edge_attr.float()

    y_pred_norm = model(dm_run).detach().cpu()

    return dm_norm, y_pred_norm


@torch.no_grad()
def mse_per_channel(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> list[float]:
    """
    Compute mean squared error per output channel.

    Optionally applies a boolean mask to select a subset of nodes (e.g., surface
    or volume nodes only) before computing MSE. All tensors are moved to match
    the device of y_pred.

    Args:
        y_pred (torch.Tensor): Predictions of shape (N, C) where C is number of channels
        y_true (torch.Tensor): Ground truth of shape (N, C)
        mask (torch.Tensor | None): Boolean mask of shape (N,) to select nodes.
                                     If None, all nodes used. Default: None

    Returns:
        list[float]: Per-channel MSE values, one per output channel.
                     Returns [NaN] * C if mask selects zero nodes.

    Example:
        >>> # Compute MSE on surface nodes only
        >>> surf_mask = surface_volume_masks_from_orig(data)[0]
        >>> mse_surf = mse_per_channel(y_pred, y_true, mask=surf_mask)
        >>> print(mse_surf)  # [mse_u, mse_v, mse_p, mse_nu_t]
    """
    dev = y_pred.device
    y_t = y_true.to(dev)

    if mask is not None:
        m = mask.to(dev)
        y_p = y_pred[m]
        y_t = y_t[m]
    else:
        y_p = y_pred

    err = (y_p - y_t) ** 2

    # Handle empty mask case
    if err.numel() == 0:
        return [float("nan")] * y_true.size(1)

    # Compute mean per channel
    return [float(err[:, i].mean().item()) for i in range(y_true.size(1))]
