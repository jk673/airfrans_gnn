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

# ---- Basic graph sanity checks (prevent CUDA device asserts) ----
def _validate_pyg_graph(data: Data, require_y: bool = True):
    """Return a list of human-readable error strings if the graph is invalid.

    Checks:
    - edge_index dtype(long), shape(2, E), and index bounds within [0, N)
    - presence of x, minimal dims
    - optional y presence and shape compatibility
    - NaNs/Infs in x, edge_index, y
    """
    errs = []
    try:
        if not hasattr(data, 'x') or data.x is None:
            errs.append("Missing node features 'x'.")
        else:
            if data.x.dim() != 2:
                errs.append(f"x should be 2D [N,F], got shape {tuple(data.x.shape)}")
            if torch.isnan(data.x).any() or torch.isinf(data.x).any():
                errs.append("x contains NaNs/Infs")
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            errs.append("Missing 'edge_index'.")
        else:
            ei = data.edge_index
            if ei.dtype != torch.long:
                errs.append(f"edge_index dtype must be long, got {ei.dtype}")
            if ei.dim() != 2 or ei.size(0) != 2:
                errs.append(f"edge_index must have shape [2, E], got {tuple(ei.shape)}")
            else:
                # Prefer explicit num_nodes if available, else derive from x
                N = None
                try:
                    N = int(getattr(data, 'num_nodes', 0))
                    if N == 0 and hasattr(data, 'x') and data.x is not None:
                        N = int(data.x.size(0))
                except Exception:
                    if hasattr(data, 'x') and data.x is not None:
                        N = int(data.x.size(0))
                if N is not None:
                    row, col = ei[0], ei[1]
                    if (row < 0).any() or (col < 0).any():
                        errs.append("edge_index has negative indices")
                    if (row >= N).any() or (col >= N).any():
                        errs.append(f"edge_index out of bounds for N={N}")
            if torch.isnan(ei).any() or torch.isinf(ei).any():
                errs.append("edge_index contains NaNs/Infs")
        # Optional pos checks
        if hasattr(data, 'pos') and data.pos is not None:
            try:
                if data.pos.dim() != 2 or data.pos.size(1) < 3:
                    errs.append(f"pos should be [N,>=3], got {tuple(data.pos.shape)}")
                else:
                    Nx = int(data.x.size(0)) if hasattr(data, 'x') and data.x is not None else None
                    Np = int(data.pos.size(0))
                    if Nx is not None and Np != Nx:
                        errs.append(f"pos N ({Np}) != x N ({Nx})")
                if torch.isnan(data.pos).any() or torch.isinf(data.pos).any():
                    errs.append("pos contains NaNs/Infs")
            except Exception:
                pass
        if require_y:
            if not hasattr(data, 'y') or data.y is None:
                errs.append("Missing ground-truth 'y'.")
            else:
                y = data.y
                if not isinstance(y, torch.Tensor):
                    try:
                        y = torch.as_tensor(y)
                    except Exception:
                        errs.append("y is not a tensor and cannot be converted")
                        y = None
                if isinstance(y, torch.Tensor):
                    if torch.isnan(y).any() or torch.isinf(y).any():
                        errs.append("y contains NaNs/Infs")
                    # Check y length matches x length
                    try:
                        Nx = int(data.x.size(0)) if hasattr(data, 'x') and data.x is not None else None
                        Ny = int(y.size(0))
                        if Nx is not None and Ny != Nx:
                            errs.append(f"y N ({Ny}) != x N ({Nx})")
                    except Exception:
                        pass
    except Exception as e:
        errs.append(f"Validation exception: {e}")
    return errs

# ============= Copy all necessary classes from notebook =============

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


class MultiGPUSubgraphModelWrapper:
    """Wrapper for multi-GPU subgraph training"""
    
    def __init__(self, model, num_gpus=1, use_ddp=False, rank=0):
        self.num_gpus = num_gpus
        self.use_ddp = use_ddp
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        
        if num_gpus > 1 and use_ddp:
            self.model = DDP(self.model, device_ids=[rank], output_device=rank)
            print(f"Rank {rank}: Using DistributedDataParallel")
        else:
            print(f"Using single GPU: {self.device}")
    
    def forward(self, data):
        data = data.to(self.device)
        return self.model(data)
    
    def get_base_model(self):
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def state_dict(self):
        return self.get_base_model().state_dict()
    
    def load_state_dict(self, state_dict):
        self.get_base_model().load_state_dict(state_dict)
    
    def parameters(self):
        return self.model.parameters()


# Update the MultiGPUCFDGraphSampler class in ddp_training.py

class MultiGPUCFDGraphSampler:
    """CFD Graph Sampler optimized for multi-GPU training with NeighborLoader"""
    
    def __init__(self, sampling_method='neighbor', num_gpus=1, rank=0, **kwargs):
        self.sampling_method = sampling_method
        self.num_gpus = num_gpus
        self.rank = rank
        self.kwargs = kwargs
        self.use_ddp = False
        
        # Cache for loader reuse optimization
        self.loader_cache = {}
        self.max_cache_size = 3
        
        self.default_neighbor_params = {
            'num_neighbors': [25, 20, 15],
            'batch_size': 64,
            'shuffle': True,
            'drop_last': True,
            'num_workers': 0,  # Set to 0 for DDP to avoid issues
            'pin_memory': False,  # CRITICAL: Set to False for DDP to avoid pinning GPU tensors
            'persistent_workers': False
        }

        # GraphSAINT defaults (does not require METIS)
        self.default_saint_params = {
            'batch_size_nodes': 20000,
            'walk_length': 2,
            'num_steps': 1,
            'shuffle': True,
            'num_workers': 0,
        }

        # Cluster defaults (requires METIS backend via pyg-lib or torch-sparse)
        self.default_cluster_params = {
            'cluster_size': 50000,   # target nodes per cluster
            'batch_size': 1,         # clusters per batch
            'shuffle': True,
        }

        # Manual cluster defaults (no METIS): spatial bin + chunk
        self.default_manual_cluster_params = {
            'cluster_size': 50000,
            'batch_size': 1,
            'shuffle': True,
            'grid_axis_bins': None,   # if None, auto from N and cluster_size
            'hop_halo': 1,            # include k-hop neighbors around clusters
            'max_nodes': 0,           # cap after halo; 0 or None means no cap
            'cap_strategy': 'prioritize_core',  # or 'uniform'
        }

    @staticmethod
    def _verify_metis_or_raise():
        try:
            from pyg_lib.partition import metis as _  # noqa: F401
            return
        except Exception:
            pass
        try:
            from torch_sparse import partition as _p
            if hasattr(_p, 'metis'):
                return
        except Exception:
            pass
        raise ImportError("METIS backend not available. Install compatible pyg-lib/torch-sparse wheels and restart the process.")

    @staticmethod
    def _make_spatial_clusters(data: Data, target_size: int, grid_axis_bins: int | None = None):
        """Partition nodes into spatially local clusters without METIS.

        Strategy: quantize 3D coordinates into a regular grid, compute a grid code per node,
        sort by code (and x for stability), then chunk sequentially by target_size.
        Returns a list of 1D LongTensors of node indices.
        """
        x = data.x
        if x is None or x.size(1) < 3:
            raise ValueError("data.x with at least 3D (positions) is required for manual clustering")

        # Use physical positions if available; else normalized positions
        pos = x[:, :3].detach().cpu()
        if hasattr(data, 'x_norm_params') and data.x_norm_params is not None:
            try:
                mean = torch.as_tensor(data.x_norm_params['mean'][:3], dtype=pos.dtype)
                scale = torch.as_tensor(data.x_norm_params['scale'][:3], dtype=pos.dtype)
                pos = pos * scale + mean
            except Exception:
                # Fallback silently to normalized pos if norm params malformed
                pass

        n = int(pos.size(0))
        if n <= target_size:
            return [torch.arange(n, dtype=torch.long)]

        # Determine number of grid bins per axis
        if grid_axis_bins is None or (isinstance(grid_axis_bins, int) and grid_axis_bins <= 0):
            # total bins ~= n / target_size -> per-axis bins ~ cube-root
            total_bins = max(1, n // max(1, int(target_size)))
            grid_axis_bins = max(1, int(round(total_bins ** (1/3))))

        mins = pos.min(dim=0).values
        maxs = pos.max(dim=0).values
        spans = (maxs - mins).clamp(min=1e-9)
        rel = (pos - mins) / spans  # in [0,1]
        # Quantize
        q = (rel * grid_axis_bins).floor().clamp(0, grid_axis_bins - 1).to(torch.long)
        ix, iy, iz = q[:, 0], q[:, 1], q[:, 2]
        code = ix + grid_axis_bins * (iy + grid_axis_bins * iz)

        # Sort by grid code then x for stability
        sort_keys = torch.stack([code, pos[:, 0].argsort().argsort()], dim=1)  # secondary via rank
        # Flatten to single key by lexicographic: stable via tuple sort not trivial; do two-stage sort
        idx = torch.argsort(code, stable=True)

        # Chunk into clusters of ~target_size
        clusters = []
        for start in range(0, n, target_size):
            end = min(n, start + target_size)
            clusters.append(idx[start:end].clone())
        return clusters

    @staticmethod
    def _build_subgraph(data: Data, node_idx: torch.Tensor) -> Data:
        """Create a subgraph Data from original Data and a 1D LongTensor of node indices.
        Includes n_id mapping to original indices. Edge features are recomputed by the model.
        """
        node_idx = node_idx.to(torch.long)
        if not hasattr(data, 'x') or data.x is None:
            raise ValueError("manual_cluster requires data.x to build subgraphs")
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            raise ValueError("manual_cluster requires data.edge_index to build subgraphs")

        num_nodes = int(data.x.size(0))
        edge_index = data.edge_index
        edge_index_cpu = edge_index.cpu() if edge_index.is_cuda else edge_index

        # Map old -> new indices
        mapping = torch.full((num_nodes,), -1, dtype=torch.long)
        mapping[node_idx] = torch.arange(node_idx.size(0), dtype=torch.long)
        row, col = edge_index_cpu[0], edge_index_cpu[1]
        new_row = mapping[row]
        new_col = mapping[col]
        mask = (new_row >= 0) & (new_col >= 0)
        new_edge_index = torch.stack([new_row[mask], new_col[mask]], dim=0)

        sub = Data(
            x=data.x[node_idx],
            edge_index=new_edge_index,
        )
        # Carry optional attributes if present and cheap
        if hasattr(data, 'pos') and data.pos is not None:
            sub.pos = data.pos[node_idx]
        # n_id enables fetching y from parent
        sub.n_id = node_idx
        return sub
    
    def create_loader(self, data: Data, **override_params):
        """Create appropriate loader for multi-GPU training, with norm/flow propagation."""

        # Transform to ensure x/y norm params and flow dir propagate to sampled batches
        def _transform(batch):
            if getattr(batch, 'x_norm_params', None) is None and getattr(data, 'x_norm_params', None) is not None:
                batch.x_norm_params = data.x_norm_params
            if getattr(batch, 'y_norm_params', None) is None and getattr(data, 'y_norm_params', None) is not None:
                batch.y_norm_params = data.y_norm_params
            if getattr(batch, 'Vel_direction', None) is None:
                batch.Vel_direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            return batch

        if self.sampling_method == 'neighbor':
            params = {**self.default_neighbor_params, **self.kwargs, **override_params}

            # For DDP, ensure safe defaults
            if self.use_ddp:
                params['pin_memory'] = False
                params['num_workers'] = 0

            effective_batch_size = params['batch_size']
            total_nodes = getattr(data, 'num_nodes', None)
            if total_nodes is None:
                total_nodes = data.x.size(0) if hasattr(data, 'x') and data.x is not None else 0
            input_nodes = torch.arange(int(total_nodes))

            if self.num_gpus > 1 and self.use_ddp:
                nodes_per_rank = len(input_nodes) // self.num_gpus
                start_idx = self.rank * nodes_per_rank
                end_idx = (self.rank + 1) * nodes_per_rank if self.rank < self.num_gpus - 1 else len(input_nodes)
                input_nodes = input_nodes[start_idx:end_idx]

            loader = NeighborLoader(
                data,
                num_neighbors=params['num_neighbors'],
                input_nodes=input_nodes,
                batch_size=effective_batch_size,
                shuffle=params['shuffle'],
                drop_last=params['drop_last'],
                num_workers=params['num_workers'],
                pin_memory=params['pin_memory'],
                persistent_workers=params.get('persistent_workers', False),
            )

        elif self.sampling_method == 'graphsaint':
            params = {**self.default_saint_params, **self.kwargs, **override_params}

            # DDP-friendly: keep num_workers 0
            if self.use_ddp:
                params['num_workers'] = 0

            # Optional: Partition steps across ranks to avoid duplicate work (opt-in)
            saint_steps = int(params.get('num_steps', 1))
            if self.use_ddp and self.num_gpus > 1 and saint_steps > 1 and params.get('partition_steps_across_ranks', False):
                base = max(1, saint_steps // self.num_gpus)
                rem = saint_steps - base * (self.num_gpus - 1)
                params['num_steps'] = base if self.rank < self.num_gpus - 1 else max(1, rem)

            loader = GraphSAINTRandomWalkSampler(
                data,
                batch_size=params['batch_size_nodes'],
                walk_length=params['walk_length'],
                num_steps=params['num_steps'],
                shuffle=params['shuffle'],
                num_workers=params.get('num_workers', 0),
            )

        elif self.sampling_method == 'cluster':
            params = {**self.default_cluster_params, **self.kwargs, **override_params}
            # Verify METIS backend availability
            self._verify_metis_or_raise()

            # Compute number of parts from target cluster size
            num_nodes = int(getattr(data, 'num_nodes', 0) or (data.x.size(0) if hasattr(data, 'x') and data.x is not None else 0))
            num_parts = max(1, num_nodes // max(1, int(params['cluster_size'])))
            clusters = ClusterData(data, num_parts=num_parts, recursive=False, save_dir=None)

            loader = ClusterLoader(
                clusters,
                batch_size=params['batch_size'],
                shuffle=params['shuffle'],
            )

        elif self.sampling_method == 'manual_cluster':
            params = {**self.default_manual_cluster_params, **self.kwargs, **override_params}

            # Build spatial clusters deterministically
            node_clusters = self._make_spatial_clusters(
                data,
                target_size=int(params.get('cluster_size', 50000)),
                grid_axis_bins=params.get('grid_axis_bins', None),
            )

            # Optionally merge multiple clusters per batch by union of node sets
            batch_size = int(params.get('batch_size', 1))
            shuffle = bool(params.get('shuffle', True))

            # Define a lightweight iterable loader
            class _ManualClusterLoader:
                def __init__(self, parent_data, clusters, batch_size, shuffle, hop_halo=1, max_nodes=0, cap_strategy='prioritize_core'):
                    self.parent_data = parent_data
                    self.clusters = clusters
                    self.batch_size = max(1, batch_size)
                    self.shuffle = shuffle
                    self.hop_halo = int(max(0, hop_halo))
                    self.max_nodes = int(max(0, max_nodes or 0))
                    self.cap_strategy = cap_strategy
                    # Preload CPU tensors for expansion
                    self.N = int(parent_data.x.size(0))
                    self.edge_index_cpu = parent_data.edge_index.cpu()

                def __len__(self):
                    return ceil(len(self.clusters) / self.batch_size)

                def __iter__(self):
                    order = torch.arange(len(self.clusters))
                    if self.shuffle:
                        order = order[torch.randperm(len(order))]
                    # Yield batches
                    for i in range(0, len(order), self.batch_size):
                        sel = order[i:i + self.batch_size]
                        # Union nodes across selected clusters
                        core_nodes = torch.unique(torch.cat([self.clusters[j] for j in sel.tolist()], dim=0))
                        nodes = core_nodes
                        # k-hop halo expansion to reduce boundary cuts
                        if self.hop_halo > 0:
                            nodes = self._expand_khop(nodes, self.hop_halo)
                        # Optional cap to avoid OOM
                        if self.max_nodes > 0 and nodes.numel() > self.max_nodes:
                            nodes = self._cap_nodes(nodes, core_nodes)
                        yield MultiGPUCFDGraphSampler._build_subgraph(self.parent_data, nodes)

                def _expand_khop(self, start_nodes: torch.Tensor, k: int) -> torch.Tensor:
                    nodes = start_nodes
                    for _ in range(k):
                        included = torch.zeros(self.N, dtype=torch.bool)
                        included[nodes] = True
                        row, col = self.edge_index_cpu[0], self.edge_index_cpu[1]
                        nbrs_from_row = col[included[row]]
                        nbrs_from_col = row[included[col]]
                        nodes = torch.unique(torch.cat([nodes, nbrs_from_row, nbrs_from_col], dim=0))
                    return nodes

                def _cap_nodes(self, nodes: torch.Tensor, core_nodes: torch.Tensor) -> torch.Tensor:
                    if self.cap_strategy == 'uniform':
                        return nodes[:self.max_nodes]
                    # prioritize_core: keep all core, then fill with halo until cap
                    core_set = torch.zeros(self.N, dtype=torch.bool)
                    core_set[core_nodes] = True
                    halo = nodes[~core_set[nodes]]
                    keep_halo = max(0, self.max_nodes - core_nodes.numel())
                    halo = halo[:keep_halo]
                    return torch.cat([core_nodes, halo], dim=0)

            loader = _ManualClusterLoader(
                data,
                node_clusters,
                batch_size,
                shuffle,
                hop_halo=int(params.get('hop_halo', 1)),
                max_nodes=int(params.get('max_nodes', 0) or 0),
                cap_strategy=params.get('cap_strategy', 'prioritize_core'),
            )

        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

        return loader


# ============= Training functions =============

def train_epoch_multigpu_subgraph(model_wrapper, train_graphs, sampler, optimizer, 
                                physics_loss, epoch, accumulation_steps=1,
                                num_graphs_per_epoch=5, batches_per_graph=25):
    """Training epoch with multi-GPU subgraph sampling"""
    
    model_wrapper.train()
    total_loss = 0
    loss_components_sum = {}
    successful_batches = 0
    
    # Robust GradScaler across torch versions (prefer torch.amp.GradScaler)
    scaler = None
    try:
        AmpGradScaler = getattr(torch.amp, 'GradScaler', None)
        if AmpGradScaler is not None:
            # New API prefers positional device type
            scaler = AmpGradScaler('cuda', enabled=True)
    except Exception:
        pass
    if scaler is None:
        # Fallback for older torch versions
        CudaAmpGradScaler = getattr(getattr(torch, 'cuda', object), 'amp', object).__dict__.get('GradScaler', None)
        scaler = CudaAmpGradScaler(enabled=True) if CudaAmpGradScaler is not None else None

    # Pre-determine graph sequence for all GPUs (ensures consistency)
    if model_wrapper.use_ddp:
        # Rank 0 samples indices and broadcasts to others for strict alignment
        if model_wrapper.rank == 0:
            torch.manual_seed(epoch + 42)
            sampled = torch.randperm(len(train_graphs))[:num_graphs_per_epoch].tolist()
        else:
            sampled = None
        obj_list = [sampled]
        dist.broadcast_object_list(obj_list, src=0)
        graph_indices = torch.tensor(obj_list[0], dtype=torch.long)
    else:
        graph_indices = torch.randperm(len(train_graphs))[:num_graphs_per_epoch]
    
    # Warm up GPU if first epoch
    if epoch == 0 and model_wrapper.rank == 0:
        print("Warming up GPU for optimal performance...")
        torch.cuda.empty_cache()
    
    optimizer.zero_grad()
    
    for graph_idx_enum, graph_idx in enumerate(graph_indices):
        # Time graph transition for monitoring
        transition_start = time.time()
        
        current_graph = train_graphs[graph_idx.item()]
        train_loader = sampler.create_loader(current_graph)
        
        if model_wrapper.rank == 0:
            transition_time = time.time() - transition_start
            print(f"Graph {graph_idx_enum+1} transition time: {transition_time:.2f}s")
        
        # Compute total steps safely (GraphSAINT implements __len__)
        try:
            loader_len = len(train_loader)
        except Exception:
            loader_len = None

        total_steps = min(batches_per_graph, loader_len) if loader_len is not None else batches_per_graph

        if model_wrapper.rank == 0:
            progress_bar = tqdm(
                enumerate(train_loader), 
                total=total_steps,
                desc=f'Epoch {epoch+1}, Graph {graph_idx_enum+1}/{num_graphs_per_epoch}'
            )
        else:
            progress_bar = enumerate(train_loader)
        
        batch_count = 0
        for batch_idx, batch in progress_bar:
            if batch_count >= total_steps:
                break
            
            try:
                batch = batch.to(model_wrapper.device, non_blocking=True)
                
                # FIXED: 확실한 norm_params 전파
                if hasattr(current_graph, 'x_norm_params'):
                    batch.x_norm_params = current_graph.x_norm_params
                else:
                    raise ValueError("Graph missing x_norm_params!")
                    
                if hasattr(current_graph, 'y_norm_params'):
                    batch.y_norm_params = current_graph.y_norm_params  
                else:
                    raise ValueError("Graph missing y_norm_params!")

                # Ensure Vel_direction is present
                if not hasattr(batch, 'Vel_direction') or batch.Vel_direction is None:
                    batch.Vel_direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=model_wrapper.device)
                
                # Ensure params are on the model/device
                batch.x_norm_params = _ensure_norm_params_on_device(batch.x_norm_params, model_wrapper.device)
                batch.y_norm_params = _ensure_norm_params_on_device(batch.y_norm_params, model_wrapper.device)
                
                # Use new autocast API if available; otherwise fall back for older torch
                autocast_ctx = None
                if hasattr(torch, 'autocast'):
                    autocast_ctx = torch.autocast('cuda')
                else:
                    autocast_ctx = torch.cuda.amp.autocast()

                with autocast_ctx:
                    predictions = model_wrapper.forward(batch)
                    
                    if hasattr(batch, 'n_id'):
                        device = predictions.device
                        if not hasattr(current_graph, '_y_device') or current_graph._y_device != device:
                            current_graph.y = current_graph.y.to(device)
                            current_graph._y_device = device
                        targets = current_graph.y[batch.n_id]
                    else:
                        targets = batch.y
                
                # Compute loss outside autocast to avoid fp16 underflow in physics terms
                import contextlib
                with (torch.autocast('cuda', enabled=False) if hasattr(torch, 'autocast') else contextlib.nullcontext()):
                    loss_result = physics_loss.compute_loss(predictions.float(), targets.float(), batch)
                    loss = loss_result['total_loss'] / accumulation_steps
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Accumulate loss and components on all ranks
                total_loss += loss_result['total_loss'].item()
                for key, val in loss_result.items():
                    if isinstance(val, torch.Tensor) and val.dim() == 0:
                        loss_components_sum[key] = loss_components_sum.get(key, 0.0) + val.item()
                
                if (batch_count + 1) % accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad()
                
                successful_batches += 1
                batch_count += 1
                
                if model_wrapper.rank == 0 and hasattr(progress_bar, 'set_postfix'):
                    # tqdm supports set_postfix; enumerate wrapper may not expose it to type checkers
                    getattr(progress_bar, 'set_postfix')({
                        'loss': loss_result['total_loss'].item(),
                        'nodes': batch.num_nodes,
                        'mem': f"{torch.cuda.memory_allocated(model_wrapper.device) / 1024**3:.1f}GB"
                    })
                
                del predictions, targets, loss_result, loss, batch
                
            except Exception as e:
                # Print local error
                if model_wrapper.rank == 0:
                    print(f"\nError in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                # Signal error to all ranks to prevent hangs
                if model_wrapper.use_ddp:
                    try:
                        err_tensor = torch.tensor(1, device=model_wrapper.device)
                        dist.all_reduce(err_tensor, op=dist.ReduceOp.SUM)
                    except Exception:
                        pass
                continue
        
        # Optimized memory cleanup - avoid blocking empty_cache
        del train_loader
        
        # Only clear cache if memory usage is high (>80%)
        try:
            allocated = torch.cuda.memory_allocated(model_wrapper.device)
            max_alloc = torch.cuda.max_memory_allocated(model_wrapper.device)
            if max_alloc and max_alloc > 0:
                memory_usage = allocated / max_alloc
                if memory_usage > 0.8:
                    torch.cuda.empty_cache()
            # Lightweight garbage collection periodically
            if graph_idx_enum % 3 == 0:
                import gc as _gc
                _gc.collect()
        except Exception:
            # If CUDA memory stats are unavailable, skip cleanup
            pass
    
    if successful_batches % accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()
    
    if model_wrapper.use_ddp:
        # Compute global average across all ranks using SUM reductions
        # 1) Reduce total loss and successful batch counts
        total_loss_tensor = torch.tensor(total_loss, device=model_wrapper.device)
        success_tensor = torch.tensor(successful_batches, device=model_wrapper.device, dtype=torch.float32)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(success_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (total_loss_tensor / success_tensor.clamp_min(1.0)).item()

        # 2) Reduce loss components (only known scalar components)
        component_keys = ['relative_l2', 'pressure_wss_consistency', 'wall_law',
                          'debug_wall_law_mode', 'debug_wall_law_wall_nodes',
                          'debug_wall_law_near_nodes', 'debug_wall_law_edges']
        avg_components = {}
        for k in component_keys:
            local_sum = torch.tensor(loss_components_sum.get(k, 0.0), device=model_wrapper.device, dtype=torch.float32)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            avg_components[k] = (local_sum / success_tensor.clamp_min(1.0)).item()
    else:
        avg_loss = total_loss / max(1, successful_batches)
        avg_components = {k: v / max(1, successful_batches) for k, v in loss_components_sum.items()}
    
    return avg_loss, avg_components


def validate_epoch_multigpu_subgraph(model_wrapper, val_graphs, sampler, physics_loss, 
                                   epoch, num_graphs=3, batches_per_graph=20):
    """Validation epoch with multi-GPU subgraph sampling"""
    
    model_wrapper.eval()
    total_loss = 0
    loss_components_sum = {}
    successful_batches = 0
    
    if model_wrapper.use_ddp:
        if model_wrapper.rank == 0:
            torch.manual_seed(1000 + epoch)
            sampled = torch.randperm(len(val_graphs))[:num_graphs].tolist()
        else:
            sampled = None
        obj_list = [sampled]
        dist.broadcast_object_list(obj_list, src=0)
        val_graph_indices = torch.tensor(obj_list[0], dtype=torch.long)
    else:
        val_graph_indices = torch.randperm(len(val_graphs))[:num_graphs]
    
    with torch.no_grad():
        for val_idx in val_graph_indices:
            val_graph = val_graphs[val_idx.item()]
            val_loader = sampler.create_loader(val_graph)
            
            batch_count = 0
            for batch_idx, batch in enumerate(val_loader):
                if batch_count >= batches_per_graph:
                    break
                
                try:
                    batch = batch.to(model_wrapper.device, non_blocking=True)
                    
                    # FIXED: 확실한 norm_params 전파 (current_graph -> val_graph로 수정)
                    if hasattr(val_graph, 'x_norm_params'):
                        batch.x_norm_params = val_graph.x_norm_params
                    else:
                        raise ValueError("Graph missing x_norm_params!")
                        
                    if hasattr(val_graph, 'y_norm_params'):
                        batch.y_norm_params = val_graph.y_norm_params  
                    else:
                        raise ValueError("Graph missing y_norm_params!")
                    
                    # Ensure Vel_direction is present
                    if not hasattr(batch, 'Vel_direction') or batch.Vel_direction is None:
                        batch.Vel_direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=model_wrapper.device)

                    # Ensure params are on the correct device
                    batch.x_norm_params = _ensure_norm_params_on_device(batch.x_norm_params, model_wrapper.device)
                    batch.y_norm_params = _ensure_norm_params_on_device(batch.y_norm_params, model_wrapper.device)
        
                    predictions = model_wrapper.forward(batch)
                    
                    if hasattr(batch, 'n_id'):
                        device = predictions.device
                        if not hasattr(val_graph, '_y_device') or val_graph._y_device != device:
                            val_graph.y = val_graph.y.to(device)
                            val_graph._y_device = device
                        targets = val_graph.y[batch.n_id]
                    else:
                        targets = batch.y
                    
                    loss_result = physics_loss.compute_loss(predictions, targets, batch)
                    
                    # Accumulate loss and components on all ranks
                    total_loss += loss_result['total_loss'].item()
                    for key, val in loss_result.items():
                        if isinstance(val, torch.Tensor) and val.dim() == 0:
                            loss_components_sum[key] = loss_components_sum.get(key, 0.0) + val.item()
                    
                    successful_batches += 1
                    batch_count += 1
                    
                    del predictions, targets, loss_result, batch
                    
                except Exception as e:
                    if model_wrapper.rank == 0:
                        print(f"\nValidation error in batch {batch_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                    continue
            
            del val_loader
            torch.cuda.empty_cache()
    
    # DDP synchronization for validation metrics
    if model_wrapper.use_ddp:
        # Global averaging across all ranks for validation
        total_loss_tensor = torch.tensor(total_loss, device=model_wrapper.device)
        success_tensor = torch.tensor(successful_batches, device=model_wrapper.device, dtype=torch.float32)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(success_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (total_loss_tensor / success_tensor.clamp_min(1.0)).item()

        component_keys = ['relative_l2', 'pressure_wss_consistency', 'wall_law']
        avg_components = {}
        for k in component_keys:
            local_sum = torch.tensor(loss_components_sum.get(k, 0.0), device=model_wrapper.device, dtype=torch.float32)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            avg_components[k] = (local_sum / success_tensor.clamp_min(1.0)).item()
    else:
        avg_loss = total_loss / max(1, successful_batches)
        avg_components = {k: v / max(1, successful_batches) for k, v in loss_components_sum.items()}
    
    return avg_loss, avg_components



# ===== Full-graph DDP helpers =====

def _choose_ddp_backend():
    """Pick a DDP backend that works on the current platform.

    Prefer 'nccl' when CUDA is available and OS is not Windows; otherwise 'gloo'.
    """
    import sys
    if torch.cuda.is_available() and not sys.platform.startswith('win'):
        return 'nccl'
    return 'gloo'


def _build_fullgraph_dataloader(graphs, batch_size_per_gpu, shuffle, use_ddp, num_gpus, rank):
    """Create a DataLoader over whole graphs with optional DistributedSampler.

    Notes:
    - PyTorch's default collate cannot handle PyG Data objects.
    - For full-graph training we typically use batch_size_per_gpu=1, so we provide
      a collate_fn that returns the single Data item directly.
    - If a larger batch size is used, we attempt to collate with PyG Batch; however,
      non-tensor attrs like normalization dicts may not collate as expected.
    """
    from torch.utils.data import DataLoader, DistributedSampler
    try:
        from torch_geometric.data import Batch as GeoBatch  # Optional for >1 batch
    except Exception:
        GeoBatch = None

    sampler = None
    if use_ddp and num_gpus > 1:
        sampler = DistributedSampler(graphs, num_replicas=num_gpus, rank=rank, shuffle=shuffle, drop_last=False)
        shuffle_flag = False
    else:
        shuffle_flag = shuffle

    # Custom collate to support PyG Data
    def _collate_pyg(batch):
        # Common case: one full graph per batch
        if len(batch) == 1:
            return batch[0]
        # If user increased batch size, try PyG Batch if available
        if GeoBatch is not None:
            try:
                return GeoBatch.from_data_list(batch)
            except Exception:
                # Fallback to first item; warn via print on rank 0 only
                if rank == 0:
                    print("Warning: Failed to collate multiple graphs; falling back to first item.")
                return batch[0]
        # No PyG Batch available; fallback
        if rank == 0:
            print("Warning: torch_geometric Batch not available; using first item of batch.")
        return batch[0]

    dl = DataLoader(
        graphs,
        batch_size=max(1, int(batch_size_per_gpu)),
        shuffle=shuffle_flag,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=_collate_pyg,
    )
    return dl, sampler


def train_epoch_fullgraph_ddp(model_wrapper, train_loader, train_sampler, optimizer, physics_loss,
                              epoch, accumulation_steps=1, strict_checks: bool = False,
                              amp_enabled: bool = True, debug_cuda_sync: bool = False):
    """Train with batches of whole graphs on each GPU, optionally sharded by DDP."""
    import contextlib
    model_wrapper.train()

    if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
        train_sampler.set_epoch(epoch)

    total_loss = 0.0
    loss_components_sum = {}
    successful_batches = 0

    # AMP scaler
    scaler = None
    try:
        AmpGradScaler = getattr(torch.amp, 'GradScaler', None)
        if AmpGradScaler is not None:
            scaler = AmpGradScaler('cuda', enabled=bool(amp_enabled))
    except Exception:
        pass
    if scaler is None:
        CudaAmpGradScaler = getattr(getattr(torch, 'cuda', object), 'amp', object).__dict__.get('GradScaler', None)
        scaler = CudaAmpGradScaler(enabled=bool(amp_enabled)) if CudaAmpGradScaler is not None else None

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        try:
            # Preflight validation to avoid device-side asserts early
            errs = _validate_pyg_graph(batch, require_y=True)
            if errs:
                if model_wrapper.rank == 0:
                    print(f"[TRAIN] Skipping invalid batch {step}: {errs[:3]}")
                if strict_checks:
                    raise RuntimeError(f"Invalid training batch: {errs}")
                continue

            batch = batch.to(model_wrapper.device, non_blocking=True)

            # Extra GPU-side sanity checks when debugging
            if (strict_checks or debug_cuda_sync) and hasattr(batch, 'edge_index') and batch.edge_index is not None:
                try:
                    N = int(batch.x.size(0)) if hasattr(batch, 'x') and batch.x is not None else None
                    ei = batch.edge_index
                    row, col = ei[0], ei[1]
                    if N is not None:
                        rmin = int(row.min().item()) if row.numel() > 0 else 0
                        rmax = int(row.max().item()) if row.numel() > 0 else -1
                        cmin = int(col.min().item()) if col.numel() > 0 else 0
                        cmax = int(col.max().item()) if col.numel() > 0 else -1
                        if rmin < 0 or cmin < 0 or rmax >= N or cmax >= N:
                            raise RuntimeError(f"edge_index OOB on GPU: N={N}, row[{rmin},{rmax}], col[{cmin},{cmax}]")
                except Exception as _e:
                    raise

            # Ensure normalization params and flow direction
            if hasattr(batch, 'x_norm_params'):
                batch.x_norm_params = _ensure_norm_params_on_device(batch.x_norm_params, model_wrapper.device)
            else:
                raise ValueError("Batch missing x_norm_params!")
            if hasattr(batch, 'y_norm_params'):
                batch.y_norm_params = _ensure_norm_params_on_device(batch.y_norm_params, model_wrapper.device)
            else:
                raise ValueError("Batch missing y_norm_params!")
            if not hasattr(batch, 'Vel_direction') or batch.Vel_direction is None:
                batch.Vel_direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=model_wrapper.device)

            # Targets required
            if not hasattr(batch, 'y') or batch.y is None:
                raise ValueError("Batch missing ground-truth 'y' for full-graph training")

            autocast_ctx = (torch.autocast('cuda', enabled=bool(amp_enabled))
                            if hasattr(torch, 'autocast') else torch.cuda.amp.autocast(enabled=bool(amp_enabled)))
            with autocast_ctx:
                predictions = model_wrapper.forward(batch)
                targets = batch.y
            if debug_cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize(model_wrapper.device)

            with (torch.autocast('cuda', enabled=False) if hasattr(torch, 'autocast') else contextlib.nullcontext()):
                loss_result = physics_loss.compute_loss(predictions.float(), targets.float(), batch)
                loss = loss_result['total_loss'] / accumulation_steps
            if debug_cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize(model_wrapper.device)

            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if debug_cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize(model_wrapper.device)

            total_loss += float(loss_result['total_loss'].item())
            for k, v in loss_result.items():
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    loss_components_sum[k] = loss_components_sum.get(k, 0.0) + float(v.item())

            if (step + 1) % accumulation_steps == 0:
                if scaler is not None and amp_enabled:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

            if debug_cuda_sync and torch.cuda.is_available():
                # Force CUDA sync to surface device-side errors at the exact line
                torch.cuda.synchronize(model_wrapper.device)

            successful_batches += 1
            del predictions, targets, loss_result, loss, batch

        except Exception as e:
            if model_wrapper.rank == 0:
                print(f"Error in full-graph batch {step}: {e}")
                import traceback as _tb
                _tb.print_exc()
            continue

    if successful_batches == 0:
        # No data processed on this rank for this epoch
        if model_wrapper.rank == 0:
            print("Warning: No training batches processed on this epoch (check dataset and sampler).")
        avg_loss = 0.0
        avg_components = {k: 0.0 for k in ['relative_l2', 'pressure_wss_consistency', 'wall_law']}
        return avg_loss, avg_components

    if successful_batches % accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()

    if model_wrapper.use_ddp:
        total_loss_tensor = torch.tensor(total_loss, device=model_wrapper.device)
        success_tensor = torch.tensor(successful_batches, device=model_wrapper.device, dtype=torch.float32)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(success_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (total_loss_tensor / success_tensor.clamp_min(1.0)).item()

        component_keys = ['relative_l2', 'pressure_wss_consistency', 'wall_law']
        avg_components = {}
        for k in component_keys:
            local_sum = torch.tensor(loss_components_sum.get(k, 0.0), device=model_wrapper.device, dtype=torch.float32)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            avg_components[k] = (local_sum / success_tensor.clamp_min(1.0)).item()
    else:
        avg_loss = total_loss / max(1, successful_batches)
        avg_components = {k: v / max(1, successful_batches) for k, v in loss_components_sum.items()}

    return avg_loss, avg_components


@torch.no_grad()
def validate_epoch_fullgraph_ddp(model_wrapper, val_loader, val_sampler, physics_loss, epoch,
                                 strict_checks: bool = False, amp_enabled: bool = True, debug_cuda_sync: bool = False):
    model_wrapper.eval()
    if val_sampler is not None and hasattr(val_sampler, 'set_epoch'):
        val_sampler.set_epoch(epoch)

    total_loss = 0.0
    loss_components_sum = {}
    successful_batches = 0

    for step, batch in enumerate(val_loader):
        try:
            # Preflight validation to avoid device-side asserts
            errs = _validate_pyg_graph(batch, require_y=True)
            if errs:
                if model_wrapper.rank == 0:
                    print(f"[VAL] Skipping invalid batch {step}: {errs[:3]}")
                if strict_checks:
                    raise RuntimeError(f"Invalid validation batch: {errs}")
                continue

            batch = batch.to(model_wrapper.device, non_blocking=True)

            if hasattr(batch, 'x_norm_params'):
                batch.x_norm_params = _ensure_norm_params_on_device(batch.x_norm_params, model_wrapper.device)
            if hasattr(batch, 'y_norm_params'):
                batch.y_norm_params = _ensure_norm_params_on_device(batch.y_norm_params, model_wrapper.device)
            if not hasattr(batch, 'Vel_direction') or batch.Vel_direction is None:
                batch.Vel_direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=model_wrapper.device)
            if not hasattr(batch, 'y') or batch.y is None:
                continue

            predictions = model_wrapper.forward(batch)
            targets = batch.y
            loss_result = physics_loss.compute_loss(predictions.float(), targets.float(), batch)

            total_loss += float(loss_result['total_loss'].item())
            for k, v in loss_result.items():
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    loss_components_sum[k] = loss_components_sum.get(k, 0.0) + float(v.item())

            successful_batches += 1
            del predictions, targets, loss_result, batch

            if debug_cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize(model_wrapper.device)

        except Exception as e:
            if model_wrapper.rank == 0:
                print(f"Validation error in full-graph batch {step}: {e}")
            continue

    if model_wrapper.use_ddp:
        total_loss_tensor = torch.tensor(total_loss, device=model_wrapper.device)
        success_tensor = torch.tensor(successful_batches, device=model_wrapper.device, dtype=torch.float32)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(success_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (total_loss_tensor / success_tensor.clamp_min(1.0)).item()
        component_keys = ['relative_l2', 'pressure_wss_consistency', 'wall_law']
        avg_components = {}
        for k in component_keys:
            local_sum = torch.tensor(loss_components_sum.get(k, 0.0), device=model_wrapper.device, dtype=torch.float32)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            avg_components[k] = (local_sum / success_tensor.clamp_min(1.0)).item()
    else:
        avg_loss = total_loss / max(1, successful_batches)
        avg_components = {k: v / max(1, successful_batches) for k, v in loss_components_sum.items()}

    return avg_loss, avg_components


def train_multigpu_subgraph_worker(rank, num_gpus, config, graphs, use_ddp, master_port):
    """Worker function for multi-GPU training"""
    # Make CUDA errors synchronous in debug to pinpoint failing op
    if bool(config.get('debug_cuda_sync', False)):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if use_ddp and num_gpus > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(num_gpus)
        backend = _choose_ddp_backend()
        dist.init_process_group(backend, rank=rank, world_size=num_gpus)
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
    
    try:
        split_idx = int(len(graphs) * 0.8)
        train_graphs = graphs[:split_idx]
        val_graphs = graphs[split_idx:]
        
        if rank == 0:
            print(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}")
        training_mode = config.get('training_mode', 'subgraph')  # 'subgraph' | 'full_graph'
        if training_mode == 'full_graph' and len(train_graphs) == 0:
            raise RuntimeError("No training graphs available after split. Check your dataset folder and split ratio.")
        
        model = EnhancedCFDSurrogateModel(
            node_feat_dim=config['input_dim'],
            edge_feat_dim=config['edge_feature_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            num_mp_layers=config['num_layers'],
            dropout_p=config['dropout'],
            num_global_tokens=2,
            use_global_tokens=config.get('use_global_tokens', False)  # Disable for DDP
        )
        
        model_wrapper = MultiGPUSubgraphModelWrapper(
            model, num_gpus=num_gpus, use_ddp=use_ddp, rank=rank
        )

        sampling_method = config.get('sampling_method', 'neighbor')
        sampler_kwargs = {}
        if training_mode == 'subgraph' and sampling_method == 'neighbor':
            sampler_kwargs = {
                'num_neighbors': config.get('num_neighbors', [25, 20, 15]),
                'batch_size': config.get('subgraph_batch_size', 64),
            }
        elif training_mode == 'subgraph' and sampling_method == 'graphsaint':
            sampler_kwargs = {
                'batch_size_nodes': config.get('saint_batch_size_nodes', 20000),
                'walk_length': config.get('saint_walk_length', 2),
                'num_steps': config.get('saint_num_steps', 1),
                'shuffle': True,
                'num_workers': config.get('saint_num_workers', 0),
                'partition_steps_across_ranks': config.get('saint_partition_steps_across_ranks', False),
            }
        elif training_mode == 'subgraph' and sampling_method == 'cluster':
            sampler_kwargs = {
                'cluster_size': config.get('cluster_size', 50000),
                'batch_size': config.get('cluster_batch_size', 1),
                'shuffle': True,
            }
        elif training_mode == 'subgraph' and sampling_method == 'manual_cluster':
            sampler_kwargs = {
                'cluster_size': config.get('manual_cluster_size', 50000),
                'batch_size': config.get('manual_cluster_batch_size', 1),
                'shuffle': True,
                'grid_axis_bins': config.get('manual_cluster_grid_axis_bins', None),
            }
        elif training_mode == 'full_graph':
            pass
        else:
            raise ValueError(f"Unsupported training_mode in config: {training_mode}")

        sampler = None
        if training_mode == 'subgraph':
            sampler = MultiGPUCFDGraphSampler(
                sampling_method=sampling_method,
                num_gpus=num_gpus,
                rank=rank,
                **sampler_kwargs,
            )
            sampler.use_ddp = use_ddp
        
        optimizer = torch.optim.Adam(
            model_wrapper.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        physics_loss = ComprehensivePhysicsLoss(
            loss_weights=config['physics_weights']
        )
        
        if rank == 0 and config.get('use_wandb', False):
            run_name = (
                f"subgraph-{sampling_method}-{num_gpus}gpu-bs{config.get('subgraph_batch_size', config.get('saint_batch_size_nodes', 'NA'))}"
                if training_mode == 'subgraph' else
                f"fullgraph-{num_gpus}gpu-gbs{config.get('graph_batch_size_per_gpu', 1)}"
            )
            wandb.init(project="gnawpinn-multigpu", config=config, name=run_name)
        
        best_val_loss = float('inf')
        
        for epoch in range(config['num_epochs']):
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1}/{config['num_epochs']}")
            
            # Optional synchronization barrier before training (can be disabled to avoid deadlocks)
            if use_ddp and num_gpus > 1 and config.get('sync_barrier', False):
                dist.barrier()  # Ensure all processes start epoch together
                if rank == 0:
                    print(f"DEBUG: All processes synchronized for epoch {epoch+1}")
            
            if training_mode == 'subgraph':
                train_loss, train_components = train_epoch_multigpu_subgraph(
                    model_wrapper, train_graphs, sampler, optimizer, physics_loss, epoch,
                    accumulation_steps=config.get('accumulation_steps', 1),
                    num_graphs_per_epoch=config.get('num_graphs_per_epoch', 5),
                    batches_per_graph=config.get('batches_per_graph', 25)
                )
            else:
                train_loader, train_dist_sampler = _build_fullgraph_dataloader(
                    train_graphs,
                    batch_size_per_gpu=config.get('graph_batch_size_per_gpu', 1),
                    shuffle=True,
                    use_ddp=use_ddp,
                    num_gpus=num_gpus,
                    rank=rank,
                )
                train_loss, train_components = train_epoch_fullgraph_ddp(
                    model_wrapper, train_loader, train_dist_sampler, optimizer, physics_loss, epoch,
                    accumulation_steps=config.get('accumulation_steps', 1),
                    strict_checks=bool(config.get('strict_graph_checks', False)),
                    amp_enabled=bool(config.get('amp_enabled', True)),
                    debug_cuda_sync=bool(config.get('debug_cuda_sync', False))
                )
            
            # Optional synchronization barrier after training epoch (disable if samplers may diverge)
            if use_ddp and num_gpus > 1 and config.get('sync_barrier', False):
                dist.barrier()
                if rank == 0:
                    print(f"DEBUG: Training epoch {epoch+1} completed on all processes")
            
            if rank == 0:
                print(f"Training Loss: {train_loss:.6f}")
                # Print detailed loss breakdown
                if train_components:
                    if 'relative_l2' in train_components:
                        print(f"  ├─ Relative L2 Loss: {train_components['relative_l2']:.6f}")
                    if 'pressure_wss_consistency' in train_components:
                        print(f"  ├─ Pressure-WSS Loss: {train_components['pressure_wss_consistency']:.6f}")
                    if 'wall_law' in train_components:
                        print(f"  └─ Wall Law Loss: {train_components['wall_law']:.6f}")
                    # Wall-law debug diagnostics (averaged over batches)
                    dbg_mode = train_components.get('debug_wall_law_mode', None)
                    if dbg_mode is not None:
                        print("  [Wall-Law Debug]")
                        print(f"    • Mode: {int(round(dbg_mode))}")
                        nodes = train_components.get('debug_wall_law_wall_nodes', 0.0)
                        near = train_components.get('debug_wall_law_near_nodes', 0.0)
                        edges = train_components.get('debug_wall_law_edges', 0.0)
                        print(f"    • Wall nodes: {nodes:.1f} | Near-wall nodes: {near:.1f} | Wall→Near edges: {edges:.1f}")
                    # Sanity check: expected weighted total from components
                    try:
                        w_rel = physics_loss.loss_weights.get('relative_l2', 1.0)
                        w_pws = physics_loss.loss_weights.get('pressure_wss_consistency', 0.0)
                        w_wall = physics_loss.loss_weights.get('wall_law', 0.0)
                        expected_weighted = (
                            w_rel * train_components.get('relative_l2', 0.0) +
                            w_pws * train_components.get('pressure_wss_consistency', 0.0) +
                            w_wall * train_components.get('wall_law', 0.0)
                        )
                        print(f"  ↳ Weighted Sum (from components): {expected_weighted:.6f}")
                    except Exception:
                        pass
            
            if (epoch + 1) % 5 == 0:
                if training_mode == 'subgraph':
                    val_loss, val_components = validate_epoch_multigpu_subgraph(
                        model_wrapper, val_graphs, sampler, physics_loss, epoch,
                        num_graphs=config.get('val_graphs_per_epoch', 3),
                        batches_per_graph=config.get('val_batches_per_graph', 20)
                    )
                else:
                    val_loader, val_dist_sampler = _build_fullgraph_dataloader(
                        val_graphs,
                        batch_size_per_gpu=config.get('graph_batch_size_per_gpu', 1),
                        shuffle=False,
                        use_ddp=use_ddp,
                        num_gpus=num_gpus,
                        rank=rank,
                    )
                    val_loss, val_components = validate_epoch_fullgraph_ddp(
                        model_wrapper, val_loader, val_dist_sampler, physics_loss, epoch,
                        strict_checks=bool(config.get('strict_graph_checks', False)),
                        amp_enabled=bool(config.get('amp_enabled', True)),
                        debug_cuda_sync=bool(config.get('debug_cuda_sync', False))
                    )
                
                if rank == 0:
                    print(f"Validation Loss: {val_loss:.6f}")
                    # Print detailed validation loss breakdown
                    if val_components:
                        if 'relative_l2' in val_components:
                            print(f"  ├─ Val Relative L2 Loss: {val_components['relative_l2']:.6f}")
                        if 'pressure_wss_consistency' in val_components:
                            print(f"  ├─ Val Pressure-WSS Loss: {val_components['pressure_wss_consistency']:.6f}")
                        if 'wall_law' in val_components:
                            print(f"  └─ Val Wall Law Loss: {val_components['wall_law']:.6f}")
                        dbg_mode = val_components.get('debug_wall_law_mode', None)
                        if dbg_mode is not None:
                            print("  [Val Wall-Law Debug]")
                            print(f"    • Mode: {int(round(dbg_mode))}")
                            nodes = val_components.get('debug_wall_law_wall_nodes', 0.0)
                            near = val_components.get('debug_wall_law_near_nodes', 0.0)
                            edges = val_components.get('debug_wall_law_edges', 0.0)
                            print(f"    • Wall nodes: {nodes:.1f} | Near-wall nodes: {near:.1f} | Wall→Near edges: {edges:.1f}")
                
                scheduler.step(val_loss)
                
                if rank == 0 and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_wrapper.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'config': config
                    }, 'best_model_multigpu_DDP.pt')
                    print(f"💾 Saved best model (val_loss: {best_val_loss:.6f})")
                
                if rank == 0 and config.get('use_wandb', False):
                    payload = {
                        "epoch": epoch,
                        "train/total_loss": train_loss,
                        "val/total_loss": val_loss,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "num_gpus": num_gpus,
                    }
                    if training_mode == 'subgraph':
                        payload["subgraph_batch_size"] = config.get('subgraph_batch_size', None)
                    else:
                        payload["graph_batch_size_per_gpu"] = config.get('graph_batch_size_per_gpu', 1)
                    wandb.log(payload)
            
            torch.cuda.empty_cache()
            gc.collect()
        
        if rank == 0 and config.get('use_wandb', False):
            wandb.finish()
        
        return model_wrapper
    
    finally:
        if use_ddp and num_gpus > 1:
            dist.destroy_process_group()


# ===== Simple dataset utilities and entrypoint =====

def load_graphs_from_folder(folder_path):
    """Load a list of PyG Data objects from a folder of .pt files.

    Expects each file to contain a dict or Data with keys: x, edge_index, y, and
    optionally x_norm_params, y_norm_params. If a dict is found, try to use 'data'.
    """
    import os as _os
    graphs = []
    if not _os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    for name in sorted(_os.listdir(folder_path)):
        if not name.lower().endswith('.pt'):
            continue
        path = _os.path.join(folder_path, name)
        try:
            obj = torch.load(path, map_location='cpu')
            if isinstance(obj, Data):
                graphs.append(obj)
            elif isinstance(obj, dict) and 'data' in obj and isinstance(obj['data'], Data):
                graphs.append(obj['data'])
            else:
                try:
                    d = Data(**obj)
                    graphs.append(d)
                except Exception:
                    continue
        except Exception:
            continue
    if not graphs:
        raise RuntimeError(f"No graphs (.pt) found in {folder_path}")
    return graphs


def run_full_graph_training(graphs, config, use_ddp=True, master_port=12355):
    """Run full-graph training with DDP when available.

    - graphs: list of PyG Data
    - config: dict, must include training_mode='full_graph'
    - use_ddp: enable ddp if multiple gpus are present
    """
    import multiprocessing as _mp
    if graphs is None or len(graphs) == 0:
        raise RuntimeError("No graphs provided to run_full_graph_training. Ensure your dataset folder contains .pt graph files.")

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_gpus = max(1, min(num_gpus, int(config.get('num_gpus', num_gpus))))

    if use_ddp and num_gpus > 1:
        # Ensure spawn start method on Windows
        try:
            _mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        procs = []
        for rank in range(num_gpus):
            p = _mp.Process(target=train_multigpu_subgraph_worker,
                            args=(rank, num_gpus, config, graphs, True, master_port))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        train_multigpu_subgraph_worker(0, 1, config, graphs, False, master_port)


def load_consolidated_graphs(data_dir: str = 'consolidated', file_pattern: str = '*graph_*.pt', limit: int | None = None):
    """Load graphs from consolidated directory.

    Parameters
    - data_dir: Directory containing consolidated graph .pt files
    - file_pattern: Glob pattern to match files (default '*graph_*.pt')
    - limit: Optional cap on number of files to load

    Returns
    - List of loaded graph objects (as stored in the .pt files)
    """
    # Use local imports to avoid requiring top-level changes
    import os
    import glob
    from tqdm import tqdm
    import torch

    graph_files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))

    if limit is not None:
        graph_files = graph_files[:limit]

    print(f"Found {len(graph_files)} graph files")

    graphs = []
    for graph_file in tqdm(graph_files, desc="Loading graphs"):
        try:
            graph_data = torch.load(graph_file, map_location='cpu', weights_only=False)
            graphs.append(graph_data)
        except Exception as e:
            print(f"Error loading {graph_file}: {e}")
            continue

    return graphs