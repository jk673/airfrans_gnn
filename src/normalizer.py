import torch
from torch_geometric.data import Data


class StandardScaler:
    def __init__(self, min_std: float = 1e-4):
        """
        Simple per-channel standardization with a minimum std floor to avoid exploding scales
        when a channel is (near-)constant.

        Args:
            min_std: Minimum standard deviation per channel.
        """
        self.mean = None
        self.std = None
        self.min_std = float(min_std)

    def fit(self, t: torch.Tensor):
        self.mean = t.mean(dim=0)
        # Unbiased=False to match PyTorch default of torch.std; clamp to min_std
        std = t.std(dim=0)
        self.std = torch.clamp(std, min=self.min_std)
        return self

    def transform(self, t: torch.Tensor):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before calling transform().")
        # Ensure mean and std are on the same device as input tensor
        mean = self.mean.to(t.device)
        std = self.std.to(t.device)
        return (t - mean) / std

    def inverse(self, t: torch.Tensor):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before calling inverse().")
        # Ensure mean and std are on the same device as input tensor
        mean = self.mean.to(t.device)
        std = self.std.to(t.device)
        return t * std + mean


class NormalizedDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, x_scaler, y_scaler):
        self.graphs = graphs
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx: int):
        d = self.graphs[idx]

        if hasattr(d, 'clone'):
            dm = d.clone()
        else:
            dm = Data(**{k: v for k, v in d.items() if hasattr(d, 'items')})

        dm.x = self.x_scaler.transform(d.x)

        if hasattr(d, 'y') and d.y is not None:
            dm.y = self.y_scaler.transform(d.y)

        dm.x_norm_params = {
            'mean': self.x_scaler.mean.clone(),
            'scale': self.x_scaler.std.clone()
        }

        if hasattr(d, 'y') and d.y is not None:
            dm.y_norm_params = {
                'mean': self.y_scaler.mean.clone(),
                'scale': self.y_scaler.std.clone()
            }
        else:
            dm.y_norm_params = None

        # Physics loss용 edge 정보
        if hasattr(d, 'edge_attr'):
            dm.edge_attr = d.edge_attr
            # Only pass through edge_attr_dxdy when it already exists, or construct
            # it with the first 3 columns if they exist. Do NOT fabricate 2-column versions.
            if hasattr(d, 'edge_attr_dxdy'):
                dm.edge_attr_dxdy = d.edge_attr_dxdy
            elif d.edge_attr is not None and d.edge_attr.dim() == 2 and d.edge_attr.shape[1] >= 3:
                dm.edge_attr_dxdy = d.edge_attr[:, :3]
        # Pass-through physics 3D edges if present
        if hasattr(d, 'edge_attr_phys'):
            dm.edge_attr_phys = d.edge_attr_phys

        # Pass-through node_area if present (for area-weighted divergence)
        if hasattr(d, 'node_area'):
            dm.node_area = d.node_area

        # Boundary condition masks (physics loss용)
        # IMPORTANT: Use RAW (unnormalized) features to derive masks. If we use normalized x,
        #            the zero-coded normal components and wall distance thresholds break
        #            (standardization makes non-wall nodes non-zero in normal cols),
        #            causing every node to look like a wall.
        try:
            from src.airfrans_utils import build_bc_masks_airfrans
            # Build masks on a copy with raw features
            if hasattr(d, 'clone'):
                d_raw = d.clone()
            else:
                d_raw = Data(**{k: v for k, v in d.items() if hasattr(d, 'items')})
            # Ensure raw x is used
            d_raw.x = d.x
            # Carry required geometry/edges if present
            if hasattr(dm, 'edge_index'):
                d_raw.edge_index = dm.edge_index
            if hasattr(dm, 'pos'):
                d_raw.pos = dm.pos
            if hasattr(dm, 'edge_attr_dxdy'):
                d_raw.edge_attr_dxdy = dm.edge_attr_dxdy
            elif hasattr(dm, 'edge_attr'):
                d_raw.edge_attr = dm.edge_attr

            d_raw = build_bc_masks_airfrans(d_raw)

            # Transfer masks/normals to the normalized sample dm
            for name in ['is_wall', 'is_inlet', 'is_outlet', 'is_farfield', 'inlet_u', 'wall_normal']:
                if hasattr(d_raw, name):
                    setattr(dm, name, getattr(d_raw, name))

            # If bc_mask_dict is present, mirror into attributes as well
            if hasattr(d_raw, 'bc_mask_dict'):
                for bc_type, mask in d_raw.bc_mask_dict.items():
                    setattr(dm, f'is_{bc_type}', mask)
        except ImportError:
            # airfrans_utils가 없다면 BC mask 생성 생략
            pass

        return dm
