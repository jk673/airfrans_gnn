import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import NeighborLoader, GraphSAINTRandomWalkSampler, ClusterLoader, ClusterData
from torch_geometric.data import Data
import os
import gc
import time
from tqdm import tqdm
try:
    import wandb  # optional
except Exception:
    class _WandbStub:
        def init(self, *a, **k):
            print("[wandb] not installed; proceeding without logging")
            return self
        def log(self, *a, **k):
            pass
        def finish(self):
            pass
    wandb = _WandbStub()
import contextlib
import contextlib
import numpy as np
from math import ceil

from loss_v3 import ComprehensivePhysicsLoss

# ---- Normalization helpers ----
def _ensure_norm_params_on_device(norm_params, device):
    """Coerce a {'mean','scale'|'std'} dict to float32 tensors on device.

    Accepts values as numpy arrays, lists, or tensors. Returns a new dict
    with 'mean' and 'scale' torch.Tensors on the requested device.
    """
    if norm_params is None:
        raise ValueError("Normalization params are required but got None")
    mean = norm_params.get('mean')
    scale = norm_params.get('scale', norm_params.get('std', None))
    if mean is None or scale is None:
        raise ValueError("Normalization params must include 'mean' and 'scale' (or 'std')")
    mean_t = mean if isinstance(mean, torch.Tensor) else torch.as_tensor(mean, dtype=torch.float32)
    scale_t = scale if isinstance(scale, torch.Tensor) else torch.as_tensor(scale, dtype=torch.float32)
    return {
        'mean': mean_t.to(device=device, dtype=torch.float32),
        'scale': scale_t.to(device=device, dtype=torch.float32)
    }


class EnhancedMeshGraphNetsProcessor(nn.Module):
    """Enhanced MeshGraphNets with global token integration"""

    def __init__(self, latent_size=128, num_layers=10, dropout=0.1,
                 num_global_tokens=2, use_global_tokens=True):
        super().__init__()
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.use_global_tokens = use_global_tokens
        self.num_global_tokens = num_global_tokens

        if use_global_tokens:
            # Transformer-style global token mechanism
            self.global_token_transformer = GlobalTokenTransformer(
                hidden_dim=latent_size,
                num_global_tokens=num_global_tokens,
                n_heads=4,
                n_layers=1,
                dropout=dropout
            )

        self.edge_models = nn.ModuleList()
        self.node_models = nn.ModuleList()

        for _ in range(num_layers):
            self.edge_models.append(nn.Sequential(
                nn.Linear(latent_size * 3, latent_size * 2),
                nn.LayerNorm(latent_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_size * 2, latent_size),
                nn.LayerNorm(latent_size)
            ))

            node_input_size = latent_size * 2
            if use_global_tokens:
                # Include per-node global context (H)
                node_input_size += latent_size

            self.node_models.append(nn.Sequential(
                nn.Linear(node_input_size, latent_size * 2),
                nn.LayerNorm(latent_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_size * 2, latent_size),
                nn.LayerNorm(latent_size)
            ))

    def forward(self, x, edge_index, edge_attr, face_areas=None, batch=None):
        device = x.device
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        if face_areas is not None:
            face_areas = face_areas.to(device)

        for i in range(self.num_layers):
            row, col = edge_index
            edge_input = torch.cat([x[row], x[col], edge_attr], dim=1)
            edge_attr_new = self.edge_models[i](edge_input)
            edge_attr = edge_attr + edge_attr_new

            x_residual = x
            num_nodes = x.size(0)
            x_aggregated = torch.zeros_like(x)
            x_aggregated.index_add_(0, col, edge_attr)

            ones = torch.ones(edge_index.size(1), 1, device=device, dtype=x.dtype)
            count = torch.zeros(num_nodes, 1, device=device, dtype=x.dtype)
            count.index_add_(0, col, ones)
            count = count.clamp(min=1)
            x_aggregated = x_aggregated / count

            node_input = [x_residual, x_aggregated]

            if self.use_global_tokens:
                # Compute transformer-style global tokens and per-node context
                # Note: face_areas no longer required here; tokens are learned and attend to nodes
                _tokens, global_context = self.global_token_transformer(x_residual)
                node_input.append(global_context)

            node_input = torch.cat(node_input, dim=1)
            x_update = self.node_models[i](node_input)
            x = x_residual + x_update

        return x, edge_attr


class EnhancedCFDSurrogateModel(nn.Module):
    """Enhanced CFD Surrogate Model optimized for multi-GPU subgraph training"""

    def __init__(self, node_feat_dim=5, edge_feat_dim=12, hidden_dim=128,
                 output_dim=4, num_mp_layers=10, dropout_p=0.1,
                 num_global_tokens=2, use_global_tokens=True):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.use_global_tokens = use_global_tokens

        self.encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Enhanced edge feature encoder for 12D input
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim * 2),  # 12 -> 256
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim * 2, hidden_dim),     # 256 -> 128
            nn.LayerNorm(hidden_dim)
        )

        self.processor = EnhancedMeshGraphNetsProcessor(
            latent_size=hidden_dim,
            num_layers=num_mp_layers,
            dropout=dropout_p,
            num_global_tokens=num_global_tokens,
            use_global_tokens=use_global_tokens
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        device = data.x.device
        
        # Validation
        if not hasattr(data, 'x') or data.x is None:
            raise ValueError("Node features (data.x) must be provided")
        if data.x.shape[1] != self.node_feat_dim:
            raise ValueError(f"Expected {self.node_feat_dim}D node features, got {data.x.shape[1]}D")

        # Ensure device consistency
        if data.edge_index.device != device:
            data.edge_index = data.edge_index.to(device)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.device != device:
            data.edge_attr = data.edge_attr.to(device)
        if hasattr(data, 'y') and data.y is not None and data.y.device != device:
            data.y = data.y.to(device)

        # Ensure pos field exists
        if not hasattr(data, 'pos') or data.pos is None:
            # Use denormalized positions if available
            data.pos = data.x[:, :3].clone().requires_grad_(True)
        else:
            if data.pos.device != device:
                data.pos = data.pos.to(device)
            if not data.pos.requires_grad:
                data.pos = data.pos.requires_grad_(True)

        # Encode node features
        x = self.encoder(data.x)
        
        # Process edge features
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            # If pre-computed edge features exist
            if data.edge_attr.shape[1] == self.edge_feat_dim:
                edge_attr = self.edge_encoder(data.edge_attr)
            else:
                # Recompute if dimension mismatch
                edge_attr = self._create_enhanced_edge_features(data, device)
                
        else:
            # Recompute if dimension mismatch
            edge_attr = self._create_enhanced_edge_features(data, device)

        # Extract face areas for global tokens
        face_areas = None
        if self.use_global_tokens and data.x.shape[1] >= 5:
            # Denormalize areas if using global tokens
            if hasattr(data, 'x_norm_params') and data.x_norm_params is not None:
                xnp = _ensure_norm_params_on_device(data.x_norm_params, device)
                area_mean = xnp['mean'][4]
                area_scale = xnp['scale'][4]
                face_areas = data.x[:, 4] * area_scale + area_mean
            else:
                face_areas = data.x[:, 4]
            
        # Message passiong
        x, edge_attr = self.processor(x, data.edge_index, edge_attr, face_areas)

        # Decode to output
        out = self.decoder(x)
        
        return out

    def _create_edge_features(self, data, device):
        """Create edge features from node positions and properties"""
        row, col = data.edge_index

        pos_i = data.x[row, :3]
        pos_j = data.x[col, :3]
        wall_dist_i = data.x[row, 3:4]
        wall_dist_j = data.x[col, 3:4]
        area_i = data.x[row, 4:5]
        area_j = data.x[col, 4:5]

        edge_vec = pos_j - pos_i
        edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
        edge_vec_normalized = edge_vec / (edge_length + 1e-8)
        
        edge_features = torch.cat([
            edge_vec_normalized,
            edge_length,
            wall_dist_j - wall_dist_i,
            (area_i + area_j) / 2.0
        ], dim=1)
        
        edge_attr = self.edge_encoder(edge_features)
        return edge_attr

    def _create_enhanced_edge_features(self, data, device):
        """Create enhanced edge features with proper denormalization"""
        row, col = data.edge_index
        
        # Step 1: Denormalize node features for physical edge calculation
        if hasattr(data, 'x_norm_params'):
            # Extract normalization parameters - 이미 tensor인지 확인
            if isinstance(data.x_norm_params['mean'], torch.Tensor):
                x_mean = data.x_norm_params['mean'].to(device).detach()
                x_scale = data.x_norm_params['scale'].to(device).detach()
            else:
                # numpy array나 list인 경우
                x_mean = torch.tensor(data.x_norm_params['mean'], dtype=torch.float32, device=device)
                x_scale = torch.tensor(data.x_norm_params['scale'], dtype=torch.float32, device=device)
            
            # Denormalize positions (first 3 features)
            pos_i_norm = data.x[row, :3]
            pos_j_norm = data.x[col, :3]
            pos_i = pos_i_norm * x_scale[:3] + x_mean[:3]
            pos_j = pos_j_norm * x_scale[:3] + x_mean[:3]
            
            # Denormalize wall distance (4th feature)
            wall_dist_i_norm = data.x[row, 3:4]
            wall_dist_j_norm = data.x[col, 3:4]
            wall_dist_i = wall_dist_i_norm * x_scale[3:4] + x_mean[3:4]
            wall_dist_j = wall_dist_j_norm * x_scale[3:4] + x_mean[3:4]
            
            # Denormalize area (5th feature)
            area_i_norm = data.x[row, 4:5]
            area_j_norm = data.x[col, 4:5]
            area_i = area_i_norm * x_scale[4:5] + x_mean[4:5]
            area_j = area_j_norm * x_scale[4:5] + x_mean[4:5]
        else:
            # If no norm params, assume data is already in physical scale
            pos_i = data.x[row, :3]
            pos_j = data.x[col, :3]
            wall_dist_i = data.x[row, 3:4]
            wall_dist_j = data.x[col, 3:4]
            area_i = data.x[row, 4:5]
            area_j = data.x[col, 4:5]
        
        # Step 2: Calculate physical edge features
        
        # Edge vector and length in physical space
        edge_vec = pos_j - pos_i
        edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
        edge_vec_normalized = edge_vec / (edge_length + 1e-8)
        
        # Relative position (normalized by edge length)
        relative_pos = edge_vec / edge_length.clamp(min=1e-8)
        
        # Wall distance gradient (physical)
        wall_grad = (wall_dist_j - wall_dist_i) / edge_length.clamp(min=1e-8)
        
        # Area ratio (physical)
        area_ratio = (area_j / area_i.clamp(min=1e-8)).clamp(0.1, 10.0)
        
        # Step 3: Combine edge features
        import numpy as np
        edge_features = torch.cat([
            edge_vec_normalized,                    # 3D: unit direction vector
            edge_length,                            # 1D: physical edge length
            relative_pos,                           # 3D: relative position
            wall_grad,                              # 1D: wall distance gradient
            area_ratio.log(),                       # 1D: log area ratio
            torch.sin(edge_vec_normalized * np.pi), # 3D: periodic features
        ], dim=1)  # Total: 12D
        
        return self.edge_encoder(edge_features)
    

class GlobalTokenTransformer(nn.Module):
    """Learned global tokens updated via Transformer-style attention.

    Mechanics:
    - Start from learned token embeddings (T x H)
    - Cross-attention: tokens attend over node features to aggregate global context
    - Feed-forward + residuals and layer norms
    - Optional repetition of the above for multiple layers
    - Finally, nodes attend over the refined tokens to obtain a per-node global context (N x H)
    """

    def __init__(self, hidden_dim: int, num_global_tokens: int = 2, n_heads: int = 4, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_global_tokens
        self.n_layers = max(1, int(n_layers))
        self.dropout = nn.Dropout(dropout)

        # Learned global tokens (sequence length = num_tokens)
        self.token_embed = nn.Parameter(torch.randn(self.num_tokens, hidden_dim) * 0.02)

        # Build layers: token<-node cross-attn + FFN
        self.attn_tok_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=False)
            for _ in range(self.n_layers)
        ])
        self.ln_tok_1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.ff_tok = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            ) for _ in range(self.n_layers)
        ])
        self.ln_tok_2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])

        # Node->token cross-attn to produce per-node global context
        self.attn_node_from_tok = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=False)

    def forward(self, node_x: torch.Tensor):
        """Compute refined tokens and per-node global context.

        Args:
            node_x: [N, H] node embeddings

        Returns:
            tokens: [T, H] refined token embeddings
            node_global_context: [N, H] per-node global context derived from tokens
        """
        N, H = node_x.shape
        # (L, B, E) format for MHA where B=1 (single graph)
        nodes = node_x.unsqueeze(1)  # [N, 1, H]
        tokens = self.token_embed.unsqueeze(1)  # [T, 1, H]

        # T layers of token refinement by attending to nodes
        for l in range(self.n_layers):
            # tokens query nodes (Q=tokens, K=nodes, V=nodes)
            tok_attn, _ = self.attn_tok_layers[l](query=tokens, key=nodes, value=nodes)
            tokens = tokens + self.dropout(tok_attn)
            # layer norm + FFN
            tokens = self.ln_tok_1[l](tokens.squeeze(1)).unsqueeze(1)
            tok_ff = self.ff_tok[l](tokens.squeeze(1))
            tokens = tokens + self.dropout(tok_ff.unsqueeze(1))
            tokens = self.ln_tok_2[l](tokens.squeeze(1)).unsqueeze(1)

        # Nodes attend to refined tokens to get a global context per node
        node_ctx, _ = self.attn_node_from_tok(query=nodes, key=tokens, value=tokens)
        node_ctx = node_ctx.squeeze(1)  # [N, H]
        return tokens.squeeze(1), node_ctx
