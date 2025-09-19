import torch
import torch.nn as nn
from torch_geometric.data import Data


class SmokeCFDModel(nn.Module):
    """
    Minimal GNN-ish model for smoke testing in DDP.
    - Simple node MLP + neighbor aggregation + residual, repeated L times.
    - Edge features are ignored for simplicity (robust to missing edge_attr).
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int = 64, output_dim: int = 4, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ) for _ in range(num_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        if x is None or edge_index is None:
            raise ValueError("data.x and data.edge_index are required")

        x = self.encoder(x)

        # simple mean aggregation on incoming edges to each node
        for layer in self.layers:
            row, col = edge_index  # messages from row -> col
            agg = torch.zeros_like(x)
            agg.index_add_(0, col, x[row])
            deg = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
            deg.index_add_(0, col, torch.ones_like(col, dtype=x.dtype).unsqueeze(1))
            deg = deg.clamp(min=1.0)
            agg = agg / deg
            x = x + layer(torch.cat([x, agg], dim=1))

        out = self.decoder(x)
        return out
