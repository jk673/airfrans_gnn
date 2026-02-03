# utils_prune.py
import torch
from torch_geometric.data import Data

def prune_isolated_nodes(data: Data, node_attr_keys=None) -> Data:
    """
    Remove nodes with degree==0, remap indices, and slice node-wise tensors.
    node_attr_keys: list of attributes to slice together (pos, x, y, masks, node_area, etc.)
    """
    if node_attr_keys is None:
        # 기본적으로 흔한 속성 추론
        node_attr_keys = []
        for k in ['pos','x','y','node_area','is_wall','is_inlet','is_outlet','is_farfield','train_mask','val_mask','test_mask']:
            if hasattr(data, k) and getattr(data, k) is not None:
                node_attr_keys.append(k)

    row, col = data.edge_index
    N = data.num_nodes
    deg = torch.bincount(row, minlength=N) + torch.bincount(col, minlength=N)
    keep = (deg > 0)
    if keep.all():
        return data  # nothing to prune

    # 새 인덱스 매핑
    old_idx = torch.arange(N, device=keep.device)
    new_idx = -torch.ones(N, dtype=torch.long, device=keep.device)
    new_idx[keep] = torch.arange(int(keep.sum()), device=keep.device)

    # edge 리맵 (고립노드로 가는 엣지는 없음)
    row_new = new_idx[row]
    col_new = new_idx[col]
    edge_index_new = torch.stack([row_new, col_new], dim=0)

    # node-wise 텐서 슬라이싱
    for key in node_attr_keys:
        val = getattr(data, key, None)
        if val is not None:
            setattr(data, key, val[keep])

    # 필수 필드 갱신
    data.edge_index = edge_index_new
    # edge_attr는 그대로 두되(엣지 수 변화 없음), 좌표계산 기반이었다면 그대로 유효
    # 필요하면 여기서 edge_attr_dxdy 등 재계산 가능

    return data
