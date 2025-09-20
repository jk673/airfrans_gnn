import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EnhancedPhysicsLoss(nn.Module):
    """Enhanced physics loss with turbulence modeling and stability terms"""
    
    def __init__(self, 
                 # Basic weights
                 data_loss_weight=1.0,
                 continuity_loss_weight=0.1,
                 momentum_loss_weight=0.1,
                 bc_loss_weight=0.05,
                 
                 # Curriculum learning targets
                 continuity_target_weight=None,  # Target weight after ramp
                 momentum_target_weight=None,    # Target weight after ramp
                 
                 # Turbulence terms
                 turbulence_loss_weight=0.05,
                 rans_loss_weight=0.05,
                 # Stability terms  
                 smoothness_weight=0.01,
                 wall_function_weight=0.02,
                 
                 # Curriculum learning parameters
                 curriculum_ramp_steps=1000,
                 ramp_start_step=0,
                 ramp_mode='linear',
                 
                 # Physical parameters
                 nu_molecular=1.5e-5,
                 chord_length=1.0,
                 # Control parameters
                 use_adaptive_weights=False,
                 debug=False):
        
        super().__init__()
        
        # Store initial weights
        self.data_loss_weight = data_loss_weight
        self.continuity_loss_weight_init = continuity_loss_weight
        self.momentum_loss_weight_init = momentum_loss_weight
        self.bc_loss_weight = bc_loss_weight
        self.turbulence_loss_weight = turbulence_loss_weight
        self.rans_loss_weight = rans_loss_weight
        self.smoothness_weight = smoothness_weight
        self.wall_function_weight = wall_function_weight
        
        # Curriculum learning targets
        self.continuity_target_weight = continuity_target_weight or continuity_loss_weight
        self.momentum_target_weight = momentum_target_weight or momentum_loss_weight
        
        # Curriculum learning schedule
        self.curriculum_ramp_steps = curriculum_ramp_steps
        self.ramp_start_step = ramp_start_step
        self.ramp_mode = ramp_mode
        
        # Physical parameters
        self.nu_molecular = nu_molecular
        self.chord_length = chord_length
        
        # Control
        self.use_adaptive_weights = use_adaptive_weights
        self.debug = debug
        
        # Track current step
        self.current_step = 0
        
    def get_current_weights(self, step):
        """Calculate current weights based on curriculum learning schedule"""
        if step is None:
            step = self.current_step
        
        if step < self.ramp_start_step:
            # Before ramp: use initial weights
            cont_weight = self.continuity_loss_weight_init
            mom_weight = self.momentum_loss_weight_init
        elif step >= self.ramp_start_step + self.curriculum_ramp_steps:
            # After ramp: use target weights
            cont_weight = self.continuity_target_weight
            mom_weight = self.momentum_target_weight
        else:
            # During ramp: interpolate
            progress = (step - self.ramp_start_step) / self.curriculum_ramp_steps
            
            if self.ramp_mode == 'linear':
                alpha = progress
            elif self.ramp_mode == 'cosine':
                alpha = 0.5 * (1 - np.cos(np.pi * progress))
            else:
                alpha = progress  # fallback to linear
            
            cont_weight = (1 - alpha) * self.continuity_loss_weight_init + alpha * self.continuity_target_weight
            mom_weight = (1 - alpha) * self.momentum_loss_weight_init + alpha * self.momentum_target_weight
        
        return cont_weight, mom_weight
        
    def forward(self, predictions, targets, data, step=None):
        """
        Enhanced physics loss computation with curriculum learning
        
        Args:
            predictions: [N, 4] (u, v, p, nu_t)
            targets: [N, 4] ground truth
            data: PyG Data object with edge_index, edge_attr_dxdy, bc_mask_dict
            step: current training step
        """
        device = predictions.device
        losses = {}
        
        # Update current step
        if step is not None:
            self.current_step = step
        
        # Get current weights based on curriculum schedule
        continuity_loss_weight, momentum_loss_weight = self.get_current_weights(step)
        
        # Store used weights for logging
        losses['cont_weight_used'] = continuity_loss_weight
        losses['mom_weight_used'] = momentum_loss_weight
        
        # 1. Data loss (MSE)
        mse_loss = F.mse_loss(predictions, targets)
        losses['mse_loss'] = mse_loss
        
        # Extract flow variables
        u_pred, v_pred, p_pred, nu_t_pred = predictions.unbind(dim=-1)
        u_true, v_true, p_true, nu_t_true = targets.unbind(dim=-1)
        
        # 2. Continuity equation loss
        if hasattr(data, 'edge_attr_dxdy') and data.edge_attr_dxdy is not None:
            div_loss = self._compute_divergence_loss(
                u_pred, v_pred, data.edge_index, data.edge_attr_dxdy
            )
            losses['continuity_loss'] = div_loss
        else:
            losses['continuity_loss'] = torch.tensor(0.0, device=device)
        
        # 3. Momentum equation loss (simplified RANS)
        if hasattr(data, 'edge_attr_dxdy') and data.edge_attr_dxdy is not None:
            momentum_loss = self._compute_momentum_loss(
                u_pred, v_pred, p_pred, nu_t_pred,
                data.edge_index, data.edge_attr_dxdy
            )
            losses['momentum_loss'] = momentum_loss
        else:
            losses['momentum_loss'] = torch.tensor(0.0, device=device)
        
        # 4. Turbulence modeling losses
        turb_losses = self._compute_turbulence_losses(
            nu_t_pred, u_pred, v_pred, data
        )
        losses.update(turb_losses)
        
        # 5. Boundary condition losses
        bc_loss = self._compute_bc_loss(predictions, data)
        losses['bc_loss'] = bc_loss
        
        # 6. Smoothness regularization
        smooth_loss = self._compute_smoothness_loss(
            predictions, data.edge_index
        )
        losses['smoothness_loss'] = smooth_loss
        
        # 7. Wall function loss
        if hasattr(data, 'bc_mask_dict') and 'wall' in data.bc_mask_dict:
            wall_loss = self._compute_wall_function_loss(
                u_pred, v_pred, nu_t_pred, data.bc_mask_dict['wall']
            )
            losses['wall_function_loss'] = wall_loss
        else:
            losses['wall_function_loss'] = torch.tensor(0.0, device=device)
        
        # Combine all losses with curriculum-adjusted weights
        total_loss = (
            self.data_loss_weight * losses['mse_loss'] +
            continuity_loss_weight * losses['continuity_loss'] +  # Use curriculum weight
            momentum_loss_weight * losses['momentum_loss'] +      # Use curriculum weight
            self.turbulence_loss_weight * losses.get('turbulence_production_loss', 0) +
            self.rans_loss_weight * losses.get('turbulence_dissipation_loss', 0) +
            self.bc_loss_weight * losses['bc_loss'] +
            self.smoothness_weight * losses['smoothness_loss'] +
            self.wall_function_weight * losses['wall_function_loss']
        )
        
        losses['total_loss'] = total_loss
        
        if self.debug and step is not None and step % 100 == 0:
            print(f"Step {step}: cont_weight={continuity_loss_weight:.4f}, mom_weight={momentum_loss_weight:.4f}")
        
        return losses
    

    def _compute_divergence_loss(self, u, v, edge_index, edge_attr_dxdy):
        """Compute divergence loss (continuity equation)"""
        device = u.device
        
        # Handle edge_attr_dxdy with different dimensions
        if edge_attr_dxdy.shape[-1] == 5:
            # If edge_attr has 5 dimensions, extract dx, dy from first two
            dx = edge_attr_dxdy[:, 0]
            dy = edge_attr_dxdy[:, 1]
        elif edge_attr_dxdy.shape[-1] == 2:
            # Original case: exactly 2 dimensions
            dx, dy = edge_attr_dxdy.unbind(dim=-1)
        else:
            # Fallback: compute approximate derivatives
            print(f"Warning: Unexpected edge_attr_dxdy shape {edge_attr_dxdy.shape}, using approximate derivatives")
            dx = torch.ones(edge_attr_dxdy.shape[0], device=device) * 0.01
            dy = torch.ones(edge_attr_dxdy.shape[0], device=device) * 0.01
        
        row, col = edge_index
        
        # Compute du/dx and dv/dy using edge attributes
        du_dx = (u[col] - u[row]) / (dx + 1e-8)
        dv_dy = (v[col] - v[row]) / (dy + 1e-8)
        
        # Aggregate derivatives at each node
        num_nodes = u.size(0)
        div_u = torch.zeros(num_nodes, device=device)
        div_v = torch.zeros(num_nodes, device=device)
        
        # Use scatter_add to accumulate contributions
        div_u.scatter_add_(0, row, du_dx)
        div_v.scatter_add_(0, row, dv_dy)
        
        # Count edges per node for normalization
        edge_count = torch.zeros(num_nodes, device=device)
        edge_count.scatter_add_(0, row, torch.ones_like(du_dx))
        edge_count = edge_count.clamp(min=1)
        
        # Normalize and compute divergence
        divergence = (div_u + div_v) / edge_count
        
        # Compute loss (divergence should be zero)
        div_loss = torch.mean(divergence ** 2)
        
        return div_loss
    


    def _compute_momentum_loss(self, u, v, p, nu_t, edge_index, edge_attr_dxdy):
        """Compute simplified momentum equation loss"""
        device = u.device
        
        # Handle edge_attr_dxdy with different dimensions
        if edge_attr_dxdy.shape[-1] == 5:
            # If edge_attr has 5 dimensions, extract dx, dy from first two
            dx = edge_attr_dxdy[:, 0]
            dy = edge_attr_dxdy[:, 1]
        elif edge_attr_dxdy.shape[-1] == 2:
            # Original case: exactly 2 dimensions
            dx, dy = edge_attr_dxdy.unbind(dim=-1)
        else:
            # Fallback: compute approximate derivatives
            dx = torch.ones(edge_attr_dxdy.shape[0], device=device) * 0.01
            dy = torch.ones(edge_attr_dxdy.shape[0], device=device) * 0.01
        
        row, col = edge_index
        
        # Compute pressure gradients
        dp_dx = (p[col] - p[row]) / (dx + 1e-8)
        dp_dy = (p[col] - p[row]) / (dy + 1e-8)
        
        # Compute velocity Laplacians (simplified)
        du2_dx2 = (u[col] - u[row]) / (dx ** 2 + 1e-8)
        dv2_dy2 = (v[col] - v[row]) / (dy ** 2 + 1e-8)
        
        # Aggregate at nodes
        num_nodes = u.size(0)
        pressure_grad_x = torch.zeros(num_nodes, device=device)
        pressure_grad_y = torch.zeros(num_nodes, device=device)
        laplacian_u = torch.zeros(num_nodes, device=device)
        laplacian_v = torch.zeros(num_nodes, device=device)
        
        pressure_grad_x.scatter_add_(0, row, dp_dx)
        pressure_grad_y.scatter_add_(0, row, dp_dy)
        laplacian_u.scatter_add_(0, row, du2_dx2)
        laplacian_v.scatter_add_(0, row, dv2_dy2)
        
        # Normalize
        edge_count = torch.zeros(num_nodes, device=device)
        edge_count.scatter_add_(0, row, torch.ones_like(dp_dx))
        edge_count = edge_count.clamp(min=1)
        
        pressure_grad_x /= edge_count
        pressure_grad_y /= edge_count
        laplacian_u /= edge_count
        laplacian_v /= edge_count
        
        # Effective viscosity
        nu_eff = self.nu_molecular + nu_t
        
        # Simplified momentum residuals (steady-state, no convective terms for stability)
        residual_x = pressure_grad_x + nu_eff * laplacian_u
        residual_y = pressure_grad_y + nu_eff * laplacian_v
        
        # Loss
        momentum_loss = torch.mean(residual_x ** 2 + residual_y ** 2)
        
        return momentum_loss

    

    def _compute_turbulence_losses(self, nu_t, u, v, data):
        """Compute turbulence-related losses"""
        device = nu_t.device
        losses = {}
        
        # 1. Positivity constraint for eddy viscosity
        negative_nu_t = F.relu(-nu_t)
        losses['turbulence_positivity_loss'] = torch.mean(negative_nu_t ** 2)
        
        # 2. Turbulence production and dissipation balance (simplified)
        if hasattr(data, 'edge_index') and hasattr(data, 'edge_attr_dxdy'):
            edge_index = data.edge_index
            edge_attr_dxdy = data.edge_attr_dxdy
            
            # Handle edge_attr_dxdy with different dimensions
            if edge_attr_dxdy.shape[-1] >= 2:
                # Extract dx, dy from first two dimensions
                dx = edge_attr_dxdy[:, 0]
                dy = edge_attr_dxdy[:, 1]
            else:
                # Fallback: compute approximate derivatives
                print(f"Warning: Unexpected edge_attr_dxdy shape {edge_attr_dxdy.shape} in turbulence losses")
                dx = torch.ones(edge_attr_dxdy.shape[0], device=device) * 0.01
                dy = torch.ones(edge_attr_dxdy.shape[0], device=device) * 0.01
            
            row, col = edge_index
            
            # Compute strain rate tensor components
            du_dx = (u[col] - u[row]) / (dx + 1e-8)
            du_dy = (u[col] - u[row]) / (dy + 1e-8)  
            dv_dx = (v[col] - v[row]) / (dx + 1e-8)
            dv_dy = (v[col] - v[row]) / (dy + 1e-8)
            
            # Production term: P = nu_t * S^2 where S is strain rate magnitude
            strain_rate_sq = 2 * (du_dx ** 2 + dv_dy ** 2) + (du_dy + dv_dx) ** 2
            
            # Aggregate at nodes
            num_nodes = nu_t.size(0)
            production = torch.zeros(num_nodes, device=device)
            production.scatter_add_(0, row, nu_t[row] * strain_rate_sq)
            
            # Count edges for normalization
            edge_count = torch.zeros(num_nodes, device=device)
            edge_count.scatter_add_(0, row, torch.ones_like(strain_rate_sq))
            edge_count = edge_count.clamp(min=1)
            production = production / edge_count
            
            # Dissipation term (simplified: assume equilibrium)
            k_turb = nu_t ** 2 / (0.09 + 1e-8)  # Rough estimate from nu_t
            dissipation = 0.09 * k_turb ** 1.5 / (1.0 + 1e-8)  # Simplified epsilon
            
            # Balance loss
            imbalance = production - dissipation
            losses['turbulence_production_loss'] = torch.mean(imbalance ** 2)
            
            # Additional physical constraint: limit nu_t magnitude
            nu_t_max = 100 * self.nu_molecular  # Maximum eddy viscosity ratio
            excess_nu_t = F.relu(nu_t - nu_t_max)
            losses['turbulence_dissipation_loss'] = torch.mean(excess_nu_t ** 2)
        else:
            losses['turbulence_production_loss'] = torch.tensor(0.0, device=device)
            losses['turbulence_dissipation_loss'] = torch.tensor(0.0, device=device)
        
        return losses

    
    def _compute_bc_loss(self, predictions, data):
        """Boundary condition losses"""
        if not hasattr(data, 'bc_mask_dict'):
            return torch.tensor(0.0, device=predictions.device)
        
        bc_losses = []
        u_pred, v_pred, p_pred, nu_t_pred = predictions.unbind(dim=-1)
        
        # Wall BC: no-slip (u=v=0)
        if 'wall' in data.bc_mask_dict:
            wall_mask = data.bc_mask_dict['wall']
            wall_loss = (u_pred[wall_mask]**2 + v_pred[wall_mask]**2).mean()
            bc_losses.append(wall_loss)
        
        # Inlet BC: prescribed velocity
        if 'inlet' in data.bc_mask_dict and hasattr(data, 'x'):
            inlet_mask = data.bc_mask_dict['inlet']
            u_inlet_target = data.x[inlet_mask, 0]  # Assuming x contains inlet velocity
            v_inlet_target = data.x[inlet_mask, 1]
            inlet_loss = F.mse_loss(u_pred[inlet_mask], u_inlet_target) + \
                        F.mse_loss(v_pred[inlet_mask], v_inlet_target)
            bc_losses.append(inlet_loss)
        
        # Outlet BC: zero gradient (soft constraint)
        if 'outlet' in data.bc_mask_dict and hasattr(data, 'edge_index'):
            outlet_mask = data.bc_mask_dict['outlet']
            # Find edges connected to outlet nodes
            row, col = data.edge_index
            outlet_edges = torch.isin(row, outlet_mask.nonzero().squeeze())
            if outlet_edges.any():
                du = (u_pred[col[outlet_edges]] - u_pred[row[outlet_edges]])**2
                dv = (v_pred[col[outlet_edges]] - v_pred[row[outlet_edges]])**2
                dp = (p_pred[col[outlet_edges]] - p_pred[row[outlet_edges]])**2
                outlet_loss = (du + dv + dp).mean()
                bc_losses.append(outlet_loss)
        
        return sum(bc_losses) / len(bc_losses) if bc_losses else torch.tensor(0.0)
    
    def _compute_smoothness_loss(self, predictions, edge_index):
        """Smoothness regularization"""
        row, col = edge_index
        
        # Differences between connected nodes
        diff = predictions[col] - predictions[row]
        smooth_loss = (diff**2).mean()
        
        return smooth_loss
    
    def _compute_wall_function_loss(self, u, v, nu_t, wall_mask):
        """Wall function for near-wall turbulence"""
        if not wall_mask.any():
            return torch.tensor(0.0, device=u.device)
        
        # Near-wall velocity magnitude
        u_mag = torch.sqrt(u[wall_mask]**2 + v[wall_mask]**2)
        
        # Simplified wall function: nu_t should decay near wall
        # y+ dependent profile (simplified)
        nu_t_wall = nu_t[wall_mask]
        
        # nu_t should be small near wall
        wall_loss = nu_t_wall.mean()
        
        return wall_loss