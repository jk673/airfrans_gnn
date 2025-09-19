import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class AdjustableSmokeLoss(nn.Module):
    """
    Simple configurable loss used for smoke testing.
    Combines MSE on outputs with optional L1 and a lightweight physics term
    on WSS magnitude vs. pressure gradient consistency (very simplified).
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.weights = {
            'mse': 1.0,
            'l1': 0.0,
            'pressure_wss': 0.0,
        }
        if weights:
            self.weights.update(weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, data) -> Dict[str, torch.Tensor]:
        comp: Dict[str, torch.Tensor] = {}

        mse = F.mse_loss(pred, target)
        comp['mse'] = mse

        l1 = (pred - target).abs().mean()
        comp['l1'] = l1

        # Very light graph-based consistency term if edge_index/pos exist
        pw = torch.tensor(0.0, device=pred.device)
        if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
            row, col = data.edge_index
            # pressure proxy from channel 0 of pred (or target fall back)
            p = pred[:, 0]
            if hasattr(data, 'pos') and data.pos is not None:
                pos = data.pos.to(pred.device)
            else:
                pos = data.x[:, :3]
            dp = (p[col] - p[row]).abs()
            dist = (pos[col] - pos[row]).pow(2).sum(dim=1).sqrt().clamp(min=1e-8)
            grad = (dp / dist).tanh()  # normalized gradient proxy
            # wss proxy from channels 1..3
            if pred.size(1) >= 4:
                tau = pred[:, 1:4]
                wss = tau.pow(2).sum(dim=1).sqrt().tanh()
                wss_mean = (wss[row] + wss[col]) * 0.5
                pw = (grad * wss_mean).mean()
        comp['pressure_wss'] = pw

        total = (
            self.weights['mse'] * mse +
            self.weights['l1'] * l1 +
            self.weights['pressure_wss'] * pw
        )
        comp['total'] = total
        return comp
