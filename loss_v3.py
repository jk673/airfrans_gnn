import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Any


class ComprehensivePhysicsLoss(nn.Module):
    """
    Comprehensive physics-informed loss with y+ matching term.
    """

    def __init__(self, loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()

        self.default_weights = {
            'relative_l2': 1.0,
            'yplus_matching': 0.0,  # Start at 0, gradually increase
            'pressure_wss_consistency': 0.0,  # Start at 0
            'wall_law': 0.0,  # Start at 0
        }

        self.loss_weights = loss_weights or self.default_weights.copy()

        # Physical constants
        self.mu = 3.75e-7  # Dynamic viscosity [Pa·s]
        self.rho = 1.0     # Density
        self.nu = self.mu / self.rho  # Kinematic viscosity

        # Wall thresholds
        self.wall_threshold = 0.09
        self.near_wall_width = 0.02
        
        # Debug storage
        self._last_wall_debug = None
        self._last_yplus_debug = None

    def denormalize_predictions(self, predictions: torch.Tensor, y_norm_params: Dict) -> torch.Tensor:
        """Denormalize predictions to physical scale"""
        device = predictions.device

        # Handle different input types for norm params
        if isinstance(y_norm_params['mean'], (np.ndarray, list)):
            y_mean = torch.tensor(y_norm_params['mean'], dtype=torch.float32, device=device)
            y_scale = torch.tensor(y_norm_params['scale'], dtype=torch.float32, device=device)
        elif isinstance(y_norm_params['mean'], torch.Tensor):
            y_mean = y_norm_params['mean'].to(device)
            y_scale = y_norm_params['scale'].to(device)
        else:
            y_mean = torch.tensor(y_norm_params['mean'], dtype=torch.float32, device=device)
            y_scale = torch.tensor(y_norm_params['scale'], dtype=torch.float32, device=device)

        # Ensure correct shape
        if len(predictions.shape) == 2 and len(y_mean.shape) == 1:
            y_mean = y_mean.unsqueeze(0)
            y_scale = y_scale.unsqueeze(0)

        return predictions * y_scale + y_mean

    def compute_relative_l2_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute relative L2 error"""
        error_norm = torch.norm(predictions - targets, p=2, dim=-1)
        target_norm = torch.norm(targets, p=2, dim=-1) + 1e-8
        relative_error = error_norm / target_norm
        return relative_error.mean()

    def compute_yplus_matching_loss(self, data: Any, predictions_physical: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute y+ matching loss using ground truth y+ values.
        Returns dict with loss value and debug metrics.
        """
        device = predictions_physical.device
        
        # Check if we have ground truth y+ values
        if not hasattr(data, 'y_plus') or data.y_plus is None:
            return {
                'loss': torch.tensor(0.0, device=device),
                'mae': torch.tensor(0.0, device=device),
                'mape': torch.tensor(0.0, device=device),
                'pred_mean': torch.tensor(0.0, device=device),
                'gt_mean': torch.tensor(0.0, device=device),
            }
        
        # Get wall distance (physical units)
        if hasattr(data, 'x_norm_params') and data.x_norm_params is not None:
            x_phys = self.denormalize_predictions(data.x.to(device), data.x_norm_params)
            wall_distance = x_phys[:, 3]
        else:
            wall_distance = data.x[:, 3]
        
        # Extract wall shear stress components from predictions
        tau_x = predictions_physical[:, 1]
        tau_y = predictions_physical[:, 2] 
        tau_z = predictions_physical[:, 3]
        
        # Compute wall shear stress magnitude
        tau_mag = torch.sqrt(tau_x**2 + tau_y**2 + tau_z**2 + 1e-12)
        
        # Compute friction velocity: u_tau = sqrt(tau_w / rho)
        u_tau = torch.sqrt(tau_mag / self.rho)
        
        # Compute predicted y+: y+ = y * u_tau / nu
        y_plus_pred = wall_distance * u_tau / self.nu
        
        # Get ground truth y+
        y_plus_gt = data.y_plus.to(device).view(-1)
        
        # Focus on near-wall region where y+ matters most
        # Typically y+ < 300 is the region of interest for wall models
        near_wall_mask = (y_plus_gt < 300) & (y_plus_gt > 0) & torch.isfinite(y_plus_gt)
        
        if not torch.any(near_wall_mask):
            # Fallback to all valid nodes
            near_wall_mask = (y_plus_gt > 0) & torch.isfinite(y_plus_gt)
        
        if not torch.any(near_wall_mask):
            return {
                'loss': torch.tensor(0.0, device=device),
                'mae': torch.tensor(0.0, device=device),
                'mape': torch.tensor(0.0, device=device),
                'pred_mean': torch.tensor(0.0, device=device),
                'gt_mean': torch.tensor(0.0, device=device),
            }
        
        # Extract values for near-wall nodes
        y_plus_pred_nw = y_plus_pred[near_wall_mask]
        y_plus_gt_nw = y_plus_gt[near_wall_mask]
        
        # Compute loss using relative error (more stable for varying scales)
        # Use log-space comparison to handle wide range of y+ values
        log_pred = torch.log(y_plus_pred_nw + 1.0)
        log_gt = torch.log(y_plus_gt_nw + 1.0)
        
        # Primary loss: MSE in log space
        yplus_loss = F.mse_loss(log_pred, log_gt)
        
        # Additional metrics for debugging
        mae = (y_plus_pred_nw - y_plus_gt_nw).abs().mean()
        mape = ((y_plus_pred_nw - y_plus_gt_nw).abs() / (y_plus_gt_nw + 1e-8)).mean()
        
        # Store debug info
        self._last_yplus_debug = {
            'num_nodes': near_wall_mask.sum().item(),
            'pred_mean': y_plus_pred_nw.mean().item(),
            'gt_mean': y_plus_gt_nw.mean().item(),
            'mae': mae.item(),
            'mape': mape.item() * 100,  # Convert to percentage
        }
        
        return {
            'loss': yplus_loss,
            'mae': mae,
            'mape': mape,
            'pred_mean': y_plus_pred_nw.mean(),
            'gt_mean': y_plus_gt_nw.mean(),
        }

    def compute_wall_law_loss(self, data: Any, predictions_physical: torch.Tensor,
                             targets_physical: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute wall-law loss (simplified version)"""
        device = predictions_physical.device
        
        # Get wall distance
        if hasattr(data, 'x_norm_params') and data.x_norm_params is not None:
            x_phys = self.denormalize_predictions(data.x.to(device), data.x_norm_params)
            wall_distance = x_phys[:, 3]
        else:
            wall_distance = data.x[:, 3]
        
        # Find wall nodes
        wall_mask = wall_distance < self.wall_threshold
        if not torch.any(wall_mask):
            return torch.tensor(0.0, device=device)
        
        # If we have targets, just match WSS at wall
        if targets_physical is not None:
            pred_wss = torch.sqrt(predictions_physical[wall_mask, 1]**2 + 
                                predictions_physical[wall_mask, 2]**2 + 
                                predictions_physical[wall_mask, 3]**2 + 1e-8)
            
            target_wss = torch.sqrt(targets_physical[wall_mask, 1]**2 + 
                                  targets_physical[wall_mask, 2]**2 + 
                                  targets_physical[wall_mask, 3]**2 + 1e-8)
            
            # Relative error
            loss = ((pred_wss - target_wss)**2 / (target_wss**2 + 1e-8)).mean()
            return loss
        
        return torch.tensor(0.0, device=device)

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    data: Any) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive physics loss with y+ matching
        """
        device = predictions.device
        loss_components = {}

        # 1. Relative L2 Loss (always active, in normalized space)
        relative_l2_loss = self.compute_relative_l2_loss(predictions, targets)
        loss_components['relative_l2'] = relative_l2_loss

        # Get denormalized predictions for physics calculations
        predictions_physical = predictions
        targets_physical = targets
        
        if hasattr(data, 'y_norm_params') and data.y_norm_params is not None:
            predictions_physical = self.denormalize_predictions(predictions, data.y_norm_params)
            try:
                targets_physical = self.denormalize_predictions(targets, data.y_norm_params)
            except Exception:
                targets_physical = None

        # 2. Y+ Matching Loss (NEW)
        yplus_result = self.compute_yplus_matching_loss(data, predictions_physical)
        loss_components['yplus_matching'] = yplus_result['loss']
        
        # Add y+ debug metrics
        loss_components['yplus_mae'] = yplus_result['mae']
        loss_components['yplus_mape'] = yplus_result['mape']
        loss_components['yplus_pred_mean'] = yplus_result['pred_mean']
        loss_components['yplus_gt_mean'] = yplus_result['gt_mean']

        # 3. Pressure-WSS Consistency (simplified)
        pressure_wss_loss = torch.tensor(0.0, device=device)
        if hasattr(data, 'edge_index') and data.edge_index.numel() > 0:
            edge_index = data.edge_index.to(device)
            row, col = edge_index
            
            # Pressure gradient
            p_coeff = predictions_physical[:, 0]
            dp = p_coeff[col] - p_coeff[row]
            
            # Use positions for distance
            if hasattr(data, 'pos') and data.pos is not None:
                pos = data.pos.to(device)
            else:
                pos = data.x[:, :3].to(device)
            
            dx = torch.norm(pos[col] - pos[row], dim=1) + 1e-8
            pressure_gradient = dp / dx
            
            # WSS magnitude
            tau = predictions_physical[:, 1:]
            wss_mag = torch.sqrt((tau**2).sum(dim=1) + 1e-8)
            wss_mean = (wss_mag[row] + wss_mag[col]) / 2.0
            
            # Simple consistency: adverse pressure gradient should correlate with low WSS
            # Normalize both to similar scales
            pg_norm = torch.tanh(pressure_gradient / 100.0)
            wss_norm = torch.tanh(wss_mean / 1.0)
            
            # Penalize positive pressure gradient with positive WSS
            pressure_wss_loss = F.relu(pg_norm * wss_norm).mean()
        
        loss_components['pressure_wss_consistency'] = pressure_wss_loss

        # 4. Wall Law (simplified)
        wall_law_loss = self.compute_wall_law_loss(data, predictions_physical, targets_physical)
        loss_components['wall_law'] = wall_law_loss

        # 5. Compute Total Loss
        total_loss = torch.tensor(0.0, device=device)
        
        for loss_name, loss_value in loss_components.items():
            if loss_name in self.loss_weights and isinstance(loss_value, torch.Tensor):
                weight = self.loss_weights.get(loss_name, 0.0)
                if weight > 0:
                    total_loss = total_loss + weight * loss_value

        loss_components['total_loss'] = total_loss

        # Debug information
        if self._last_yplus_debug is not None:
            loss_components['debug_yplus_nodes'] = torch.tensor(
                self._last_yplus_debug['num_nodes'], device=device, dtype=torch.float32
            )
            loss_components['debug_yplus_pred_mean'] = torch.tensor(
                self._last_yplus_debug['pred_mean'], device=device, dtype=torch.float32
            )
            loss_components['debug_yplus_gt_mean'] = torch.tensor(
                self._last_yplus_debug['gt_mean'], device=device, dtype=torch.float32
            )
            loss_components['debug_yplus_mape_pct'] = torch.tensor(
                self._last_yplus_debug['mape'], device=device, dtype=torch.float32
            )

        return loss_components