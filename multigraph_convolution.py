import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import math

class MultiScaleGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__(aggr='mean')
        self.kernel_size = kernel_size
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_edge = None  # Will be created if edge_attr is provided
        
    def forward(self, x, edge_index, edge_attr=None):
        # Apply linear transformation first
        x = self.lin(x)
        
        # Handle edge attributes if provided
        if edge_attr is not None:
            if self.lin_edge is None:
                # Lazily create edge projection layer
                self.lin_edge = nn.Linear(edge_attr.size(-1), x.size(-1)).to(x.device)
            edge_attr = self.lin_edge(edge_attr)
        
        # Propagate with k-hop neighbors
        out = x
        for k in range(self.kernel_size):
            out = self.propagate(edge_index, x=out, edge_attr=edge_attr)
        
        return out
    
    def message(self, x_j, edge_attr=None):
        if edge_attr is not None:
            # Use addition instead of multiplication for better stability
            return x_j + edge_attr * 0.1  # Small weight for edge features
        return x_j


class DilatedGraphConv(MessagePassing):
    """Dilated Graph Convolution for capturing long-range dependencies"""
    
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(aggr='mean')
        self.dilation = dilation
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr=None):
        # Dilated propagation (k-hop neighbors)
        x_prop = x
        for _ in range(self.dilation):
            x_prop = self.propagate(edge_index, x=x_prop, edge_attr=edge_attr)
        
        # Combine with self-connection
        return self.lin(x_prop) + self.lin_self(x)
    
    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return x_j * edge_attr.sigmoid()
        return x_j


class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling for multi-resolution feature extraction"""
    
    def __init__(self, in_channels, pool_sizes=[1, 2, 4]):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        
        # Output projection
        total_channels = in_channels * len(pool_sizes) * 2  # mean + max pooling
        self.projection = nn.Linear(total_channels, in_channels)
        
    def forward(self, x, batch):
        pooled_features = []
        
        for pool_size in self.pool_sizes:
            # Adaptive pooling to fixed size
            n_nodes = x.size(0)
            if pool_size >= n_nodes:
                pool_size = n_nodes
            
            # Both mean and max pooling
            mean_pool = self._adaptive_pool(x, batch, pool_size, 'mean')
            max_pool = self._adaptive_pool(x, batch, pool_size, 'max')
            
            pooled_features.append(mean_pool)
            pooled_features.append(max_pool)
        
        # Concatenate all pooled features
        x_pooled = torch.cat(pooled_features, dim=-1)
        return self.projection(x_pooled)
    
    def _adaptive_pool(self, x, batch, pool_size, mode='mean'):
        """Adaptive pooling to specific size"""
        # This is simplified - in practice you'd use clustering or FPS
        if mode == 'mean':
            return global_mean_pool(x, batch)
        else:
            return global_max_pool(x, batch)