# airfrans_utils.py
# -----------------
# AirfRANS/유사 CFD 그래프를 위한 전처리 유틸 모음
# - edge_attr_dxdy 생성: [dx, dy, dist]
# - node_area 추정: face 있으면 삼각형 면적/3, 없으면 perimeter^2/(4π) 근사
# - 경계 마스크: is_wall / is_inlet / is_outlet / is_farfield
# - 벽 노멀: 주어진 노멀 없으면 ∇(wall_distance)로 복구
# - 한 번에 실행하는 prepare_airfrans_graph_for_physics() 포함
#
# 의존성: torch, torch_scatter

from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import numpy as np
from matplotlib.path import Path

import torch
from torch_scatter import scatter_add
from torch_geometric.data import Data, Batch


# =========================
# Edge geometry primitives
# =========================

def _half_edges(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """무방향 그래프 가정에서 (row < col)인 한쪽 방향만 유지.
    edge_attr가 None인 경우는 이 유틸 호출 컨텍스트에서 지원하지 않음.
    """
    row, col = edge_index
    mask = row < col
    if mask.any():
        edge_index = edge_index[:, mask]
        if edge_attr is None:
            raise ValueError("_half_edges requires edge_attr when mask is applied")
        edge_attr = edge_attr[mask]
    if edge_attr is None:
        raise ValueError("_half_edges requires non-None edge_attr")
    return edge_index, edge_attr


def _valid_edges(edge_index: torch.Tensor, N: int) -> torch.Tensor:
    row, col = edge_index
    return (row >= 0) & (row < N) & (col >= 0) & (col < N)


def _extract_dxdy_length(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pos: Optional[torch.Tensor],
    prefer_dxdy: bool = True,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    dx, dy, length per edge 리턴.
    스키마:
      (A) default/dist-dir: [dist, dir_x, dir_y] (dist>0, |dir|≈1)
      (B) dxdy           : [dx, dy, dist]
    둘 다 아니면 pos로부터 계산.
    """
    device = edge_attr.device
    if prefer_dxdy and edge_attr.size(1) >= 3:
        col0 = edge_attr[:, 0].abs().median()
        col1 = edge_attr[:, 1].abs().median()
        is_default = (col1 <= 1.5) and torch.all(edge_attr[:, 0] > 0)
        if is_default:
            length = edge_attr[:, 0].clamp_min(eps)
            dx = edge_attr[:, 1] * length
            dy = edge_attr[:, 2] * length
            return dx, dy, length
        else:
            dx = edge_attr[:, 0]
            dy = edge_attr[:, 1]
            length = edge_attr[:, 2].abs().clamp_min(eps)
            return dx, dy, length

    if pos is None:
        raise ValueError("Need pos to compute dx,dy when edge_attr schema is unknown")
    row, col = edge_index
    dvec = pos[col, :2] - pos[row, :2]
    length = dvec.norm(dim=1).clamp_min(eps)
    dx = dvec[:, 0]
    dy = dvec[:, 1]
    return dx, dy, length


# =========================
# Public snippets
# =========================

@torch.no_grad()
def build_edge_attr_dxdy(data):
    """
    data.edge_attr_dxdy = [dx, dy, dist] 생성
    """
    # Gracefully skip if edges/positions are not available (e.g., raw samples without prebuilt edges)
    if not hasattr(data, 'pos') or data.pos is None:
        return data
    if not hasattr(data, 'edge_index') or data.edge_index is None or data.edge_index.numel() == 0:
        return data
    device = data.pos.device
    row, col = data.edge_index

    dvec = data.pos[col, :2] - data.pos[row, :2]      # [E,2]
    dist = dvec.norm(dim=1).clamp_min(1e-12)          # [E]
    dx   = dvec[:, 0]
    dy   = dvec[:, 1]

    data.edge_attr_dxdy = torch.stack([dx, dy, dist], dim=-1).to(device)
    return data


@torch.no_grad()
def estimate_node_area(data):
    """
    data.node_area 생성.
    우선순위:
      1) data.face (삼각형) 있으면: 각 삼각형 면적을 1/3씩 꼭짓점 노드에 분배
      2) 없으면: perimeter 근사  A_i ≈ (perimeter_i^2)/(4π)
    """
    if not hasattr(data, 'pos') or data.pos is None:
        return data
    device = data.pos.device
    N = getattr(data, 'num_nodes', data.pos.size(0))

    if getattr(data, 'face', None) is not None and data.face.numel() > 0:
        f = data.face.to(torch.long).to(device)  # [3,F]
        v0 = data.pos[f[0], :2]
        v1 = data.pos[f[1], :2]
        v2 = data.pos[f[2], :2]
        area_tri = 0.5 * torch.abs((v1 - v0)[:, 0] * (v2 - v0)[:, 1] - (v1 - v0)[:, 1] * (v2 - v0)[:, 0])  # [F]
        area_node = torch.zeros(N, device=device)
        one_third = (area_tri / 3.0)
        for k in range(3):
            area_node.index_add_(0, f[k], one_third)
        data.node_area = area_node.clamp_min(1e-12)
        return data

    # face가 없을 때: perimeter 근사 (edges 필요). 없으면 단순 기본값으로 대체.
    if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
        row, col = data.edge_index
        dvec = data.pos[col, :2] - data.pos[row, :2]
        length = dvec.norm(dim=1)
        per = scatter_add(length, row, dim=0, dim_size=N)
        per = per + scatter_add(length, col, dim=0, dim_size=N)
        area_proxy = (per * per) / (4.0 * math.pi)
        data.node_area = area_proxy.clamp_min(1e-12)
    else:
        # Fallback: uniform area proxy when edges are unavailable
        data.node_area = torch.ones(N, device=device)
    return data


# =========================
# Requested: Dual-mesh areas + preprocessing hook
# =========================

@torch.no_grad()
def compute_dual_mesh_areas(
    positions: Union[torch.Tensor, np.ndarray],
    cells: Union[torch.Tensor, np.ndarray]
) -> torch.Tensor:
    """Compute per-node dual areas by distributing each triangle area equally to its 3 vertices.

    Args:
        positions: [N,2 or 3] node coordinates (torch or numpy)
        cells: [3,F] or [F,3] triangle indices (torch or numpy)

    Returns:
        torch.Tensor [N] of non-negative areas
    """
    # Convert to torch tensors on CPU for robustness (move to GPU on return based on positions)
    pos_is_torch = isinstance(positions, torch.Tensor)
    if not pos_is_torch:
        pos_t = torch.as_tensor(positions, dtype=torch.float32)
    else:
        pos_t = positions.detach().to(dtype=torch.float32)
    dev = positions.device if isinstance(positions, torch.Tensor) else torch.device('cpu')

    if isinstance(cells, torch.Tensor):
        face = cells.detach().to(dtype=torch.long)
    else:
        face = torch.as_tensor(cells, dtype=torch.long)

    if face.ndim == 2 and face.size(0) == 3:  # [3,F]
        f0, f1, f2 = face[0], face[1], face[2]
    elif face.ndim == 2 and face.size(1) == 3:  # [F,3]
        f0, f1, f2 = face[:, 0], face[:, 1], face[:, 2]
    else:
        raise ValueError("cells must be shape [3,F] or [F,3] of triangle indices")

    # Use first two coords (x,y)
    P = pos_t[:, :2]
    v0 = P[f0]
    v1 = P[f1]
    v2 = P[f2]
    # Triangle area = 0.5*| (v1-v0) x (v2-v0) |
    a = 0.5 * torch.abs((v1 - v0)[:, 0] * (v2 - v0)[:, 1] - (v1 - v0)[:, 1] * (v2 - v0)[:, 0])
    N = P.size(0)
    area_node = torch.zeros(N, dtype=torch.float32)
    one_third = a / 3.0
    for idx in (f0, f1, f2):
        area_node.index_add_(0, idx, one_third)
    area_node = area_node.clamp_min(1e-12)
    return area_node.to(device=dev)


@torch.no_grad()
def preprocess_airfrans_data(data):
    """Preprocess a Data-like object to add normalized node_area and provide an option
    to extend edge attributes with node-area info.

    Steps:
    - Compute node areas from triangles if available (pos + face/cells)
    - Normalize by mean for stability
    - Store at data.node_area (torch.Tensor [N])
    Note: This function does not mutate data.edge_attr by default to avoid breaking
    existing consumers; use attach_node_area_to_edges to create an augmented edge_attr.
    """
    # Positions (may be missing in some pipelines)
    has_pos = hasattr(data, 'pos') and (data.pos is not None)
    pos = data.pos if has_pos else None
    # Cells: prefer `face` ([3,F] in PyG), else allow `cells`/`triangles`
    cells = None
    for key in ('face', 'cells', 'triangles'):
        if hasattr(data, key) and getattr(data, key) is not None:
            cells = getattr(data, key)
            break

    if cells is None or (isinstance(cells, torch.Tensor) and cells.numel() == 0):
        # No triangles. Try to compute areas even without pos using edge lengths.
        areas = None
        try:
            if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
                row, col = data.edge_index
                # Prefer physics 3D edges for dist
                dist = None
                if hasattr(data, 'edge_attr_phys') and data.edge_attr_phys is not None and data.edge_attr_phys.dim() == 2 and data.edge_attr_phys.size(1) >= 3:
                    dist = data.edge_attr_phys[:, 2].abs()
                elif hasattr(data, 'edge_attr_dxdy') and data.edge_attr_dxdy is not None and data.edge_attr_dxdy.dim() == 2 and data.edge_attr_dxdy.size(1) >= 3:
                    dist = data.edge_attr_dxdy[:, 2].abs()
                elif hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.dim() == 2 and data.edge_attr.size(1) >= 3:
                    dist = data.edge_attr[:, 2].abs()
                if dist is not None:
                    N = getattr(data, 'num_nodes', None)
                    if N is None and hasattr(data, 'x') and isinstance(data.x, torch.Tensor):
                        N = int(data.x.size(0))
                    if N is None and has_pos and isinstance(pos, torch.Tensor):
                        N = int(pos.size(0))
                    if N is None:
                        raise RuntimeError('Cannot determine num_nodes for area computation')
                    # Perimeter: sum of incident edge lengths; add both directions for symmetry
                    per = scatter_add(dist, row, dim=0, dim_size=N) + scatter_add(dist, col, dim=0, dim_size=N)
                    areas = (per * per) / (4.0 * math.pi)
        except Exception:
            areas = None
        if areas is None:
            # Fallback to estimator that may use pos/edges if available
            data = estimate_node_area(data)
            areas = getattr(data, 'node_area', None)
    else:
        if has_pos and isinstance(pos, torch.Tensor):
            areas = compute_dual_mesh_areas(pos, cells)
            data.node_area = areas
        else:
            # No positions available; fallback
            data = estimate_node_area(data)
            areas = getattr(data, 'node_area', None)

    if isinstance(areas, torch.Tensor):
        mean = areas.mean().clamp_min(1e-12)
        # Choose device/dtype sensibly even if pos is missing
        if has_pos and isinstance(pos, torch.Tensor):
            device = pos.device
            dtype = pos.dtype
        elif hasattr(data, 'x') and isinstance(data.x, torch.Tensor):
            device = data.x.device
            dtype = data.x.dtype
        else:
            device = areas.device
            dtype = areas.dtype
        data.node_area = (areas / mean).to(device=device, dtype=dtype)
    return data


@torch.no_grad()
def attach_node_area_to_edges(
    data,
    *,
    use_phys_edges: bool = True,
    mode: str = 'pair',  # 'pair' -> append [area_i, area_j], 'mean' -> append [area_mean]
    target_attr_name: str = 'edge_attr_phys_ext'
):
    """Create an augmented edge attribute tensor that includes node-area information.

    Does not overwrite existing edge_attr. Stores result in `target_attr_name`.
    - Base source: edge_attr_phys (preferred, shape [E,3]) or edge_attr (first 3 cols)
    - Appended columns:
        mode='pair' -> [dx, dy, dist, area_i, area_j]
        mode='mean' -> [dx, dy, dist, area_mean]
    """
    if not hasattr(data, 'edge_index') or data.edge_index is None:
        return data
    if not hasattr(data, 'node_area') or data.node_area is None:
        return data

    # Pick base 3D edge attr (dx,dy,dist)
    base = None
    if hasattr(data, 'edge_attr_phys') and data.edge_attr_phys is not None and data.edge_attr_phys.dim() == 2 and data.edge_attr_phys.size(1) >= 3:
        base = data.edge_attr_phys[:, :3]
    elif hasattr(data, 'edge_attr_dxdy') and data.edge_attr_dxdy is not None and data.edge_attr_dxdy.dim() == 2 and data.edge_attr_dxdy.size(1) >= 3:
        base = data.edge_attr_dxdy[:, :3]
    elif hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.dim() == 2 and data.edge_attr.size(1) >= 3:
        base = data.edge_attr[:, :3]
    else:
        # Compute from pos if nothing available
        data = build_edge_attr_dxdy(data)
        if not hasattr(data, 'edge_attr_dxdy'):
            return data
        base = data.edge_attr_dxdy[:, :3]

    row, col = data.edge_index
    ai = data.node_area[row].to(base.dtype)
    aj = data.node_area[col].to(base.dtype)
    if mode == 'mean':
        am = 0.5 * (ai + aj)
        ext = torch.cat([base, am.unsqueeze(1)], dim=1)
    else:  # pair
        ext = torch.cat([base, ai.unsqueeze(1), aj.unsqueeze(1)], dim=1)
    setattr(data, target_attr_name, ext)
    return data


# ---- Weighted differential operators (for wall-normal recovery) ----

def weighted_gradient(
    field: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int,
    *,
    pos: Optional[torch.Tensor] = None,
    prefer_dxdy: bool = True,
    weight_mode: str = "rbf",
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ∇f(i) ≈ sum_j w_ij (f_j-f_i) d_ij / r_ij^2  / sum_j w_ij
    """
    device = field.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    edge_attr  = edge_attr.to(device=device, dtype=field.dtype)
    if pos is not None:
        pos = pos.to(device=device, dtype=field.dtype)

    N = field.size(0)
    if edge_index.numel() == 0 or edge_attr.numel() == 0:
        z = torch.zeros(N, device=device, dtype=field.dtype)
        return z, z

    valid = _valid_edges(edge_index, N)
    if not torch.all(valid):
        edge_index = edge_index[:, valid]
        edge_attr  = edge_attr[valid]
        if edge_index.numel() == 0:
            z = torch.zeros(N, device=device, dtype=field.dtype)
            return z, z

    edge_index, edge_attr = _half_edges(edge_index, edge_attr)
    if edge_index is None or edge_attr is None or edge_index.numel() == 0:
        z = torch.zeros(N, device=device, dtype=field.dtype)
        return z, z
    row, col = edge_index

    dx, dy, length = _extract_dxdy_length(edge_index, edge_attr, pos, prefer_dxdy, eps)

    if weight_mode == "rbf":
        h2 = (length.mean() ** 2).clamp_min(eps)
        w = torch.exp(-(length * length) / (h2 + eps))
    else:
        w = 1.0 / (length * length + eps)

    df = (field[col] - field[row])  # [E]
    inv_r2 = 1.0 / (length * length + eps)

    gx_edge = w * df * (dx * inv_r2)
    gy_edge = w * df * (dy * inv_r2)

    num_x = scatter_add(gx_edge, row, dim=0, dim_size=N) + scatter_add(gx_edge, col, dim=0, dim_size=N)
    num_y = scatter_add(gy_edge, row, dim=0, dim_size=N) + scatter_add(gy_edge, col, dim=0, dim_size=N)
    den   = scatter_add(w,      row, dim=0, dim_size=N) + scatter_add(w,      col, dim=0, dim_size=N)
    den   = den.clamp_min(1.0)

    return num_x / den, num_y / den


@torch.no_grad()
def build_bc_masks_airfrans(
    data,
    *,
    wall_dist_col: int = 2,           # x[:,2] = wall distance
    normal_cols: tuple = (3, 4),      # x[:,3], x[:,4] = wall normal (0 if not surface)
    inlet_outlet_quantile: float = 0.02,
    farfield_quantile: float = 0.90,
    wall_dist_thresh: float = 1e-4,
    use_grad_for_normals_if_missing: bool = True,
    U_inf_from_x: bool = True,
):
    """
    data.is_wall, data.is_inlet, data.is_outlet, data.is_farfield, data.inlet_u, data.wall_normal 생성
    - is_wall: wall distance≈0 또는 normal 제공 노드
    - inlet/outlet: x 좌표 quantile 기준(좌=입구, 우=출구 가정)
    - farfield: 외곽 quantile 박스 기반
    - inlet_u: [N,2], 입구 노드에서만 freestream 속도 할당
    - wall_normal: [N,2], 없으면 ∇(wall_distance)로 복구
    """
    device = data.pos.device
    N = data.num_nodes

    X = getattr(data, 'x', None)
    if X is None:
        raise ValueError("data.x is required for AirfRANS features.")
    xfeat = X  # [N, ?]

    pos = data.pos[:, :2]
    xcoord = pos[:, 0]
    ycoord = pos[:, 1]

    # freestream velocity (물리 스케일)
    if U_inf_from_x and xfeat.size(1) >= 2:
        Ux_inf = xfeat[:, 0]
        Uy_inf = xfeat[:, 1]
        U_inf = torch.stack([Ux_inf, Uy_inf], dim=-1)
    else:
        U_inf = torch.zeros(N, 2, device=device)
        U_inf[:, 0] = 1.0

    # wall distance & normals 후보
    if xfeat.size(1) > wall_dist_col:
        wall_dist = xfeat[:, wall_dist_col].abs()
    else:
        wall_dist = torch.full((N,), 1e9, device=device)

    have_normals = False
    normals_unreliable = True
    if xfeat.size(1) > max(normal_cols):
        nx = xfeat[:, normal_cols[0]]
        ny = xfeat[:, normal_cols[1]]
        nx_ny = torch.stack([nx, ny], dim=-1)
        nz_mask = (nx_ny.abs().sum(dim=1) > 0)
        frac_nz = nz_mask.float().mean()
        # Treat normals as reliable only if they are present on a small subset (e.g., <20%)
        have_normals = frac_nz.item() > 0.0
        normals_unreliable = frac_nz.item() > 0.2
    else:
        nx_ny = torch.zeros(N, 2, device=device)

    # is_wall
    is_wall_from_dist = (wall_dist <= wall_dist_thresh)
    if have_normals and not normals_unreliable:
        # Only consider normals very close to the wall; be conservative
        near_thresh = torch.minimum(
            torch.as_tensor(wall_dist_thresh * 100.0, device=device, dtype=wall_dist.dtype),
            torch.quantile(wall_dist, 0.05)
        )
        is_wall_from_norm = (nx_ny.abs().sum(dim=1) > 0) & (wall_dist <= near_thresh)
        candidate = is_wall_from_dist | is_wall_from_norm
        # Sanity fallback: if normals cause >80% nodes to be wall, ignore normals
        frac = candidate.float().mean()
        if frac.item() > 0.8:
            is_wall = is_wall_from_dist
        else:
            is_wall = candidate
    else:
        # Ignore normals if they look unreliable (non-zero for a large fraction of nodes)
        is_wall = is_wall_from_dist
    data.is_wall = is_wall.to(torch.uint8).to(device)

    # wall normals: 제공 없으면 ∇(wall_dist)로 복구
    wall_normal = torch.zeros(N, 2, device=device)
    if have_normals:
        wall_normal = nx_ny.clone()
    elif use_grad_for_normals_if_missing:
        def _need_edge_attr(d):
            ea = getattr(d, 'edge_attr_dxdy', getattr(d, 'edge_attr', None))
            if ea is None:
                raise ValueError("Need edge_attr or edge_attr_dxdy for gradient.")
            return ea
        gx, gy = weighted_gradient(
            wall_dist,
            data.edge_index,
            _need_edge_attr(data).to(device),
            N,
            pos=pos,
            prefer_dxdy=True,
            weight_mode="rbf"
        )
        g = torch.stack([gx, gy], dim=-1)
        n = g / (g.norm(dim=1, keepdim=True).clamp_min(1e-12))
        wall_normal = n
    data.wall_normal = wall_normal

    # inlet/outlet (x-quantile)
    q = inlet_outlet_quantile
    x_min_q = torch.quantile(xcoord, q)
    x_max_q = torch.quantile(xcoord, 1.0 - q)
    is_inlet = (xcoord <= x_min_q) & (~is_wall)
    is_outlet = (xcoord >= x_max_q) & (~is_wall)

    # farfield (outer quantile box)
    x_lo = torch.quantile(xcoord, 1.0 - farfield_quantile)
    x_hi = torch.quantile(xcoord, farfield_quantile)
    y_lo = torch.quantile(ycoord, 1.0 - farfield_quantile)
    y_hi = torch.quantile(ycoord, farfield_quantile)
    is_outer_box = (xcoord <= x_lo) | (xcoord >= x_hi) | (ycoord <= y_lo) | (ycoord >= y_hi)
    is_farfield = is_outer_box & (~is_wall) & (~is_inlet) & (~is_outlet)

    data.is_inlet = is_inlet.to(torch.uint8).to(device)
    data.is_outlet = is_outlet.to(torch.uint8).to(device)
    data.is_farfield = is_farfield.to(torch.uint8).to(device)

    # inlet_u
    inlet_u = torch.zeros(N, 2, device=device, dtype=U_inf.dtype)
    inlet_u[is_inlet] = U_inf[is_inlet]
    data.inlet_u = inlet_u

    return data


@torch.no_grad()
def prepare_airfrans_graph_for_physics(data, *, verbose: bool = True):
    """
    edge_attr_dxdy, node_area, BC masks/노멀을 한 번에 준비.
    """
    data = build_edge_attr_dxdy(data)
    data = estimate_node_area(data)
    data = build_bc_masks_airfrans(data)

    if verbose:
        N = data.num_nodes
        ea = getattr(data, 'edge_attr_dxdy', None)
        print(f"[prep] edge_attr_dxdy: {None if ea is None else tuple(ea.shape)}")
        print(f"[prep] node_area: {tuple(data.node_area.shape)} | min={float(data.node_area.min()):.3e}")
        for name in ["is_wall", "is_inlet", "is_outlet", "is_farfield"]:
            m = getattr(data, name, None)
            if m is not None:
                print(f"[prep] {name}: {int(m.sum())} / {N}")
        if getattr(data, 'wall_normal', None) is not None:
            wn = data.wall_normal
            print(f"[prep] wall_normal: {tuple(wn.shape)} | avg_norm={float(wn.norm(dim=1).mean()):.3f}")
        iu = getattr(data, 'inlet_u', None)
        if iu is not None:
            mag = iu.norm(dim=1)
            mean_mag = float(mag[data.is_inlet.bool()].mean()) if int(data.is_inlet.sum()) > 0 else 0.0
            print(f"[prep] inlet_u: {tuple(iu.shape)} | |U_inf|(inlet,mean)={mean_mag:.3f}")
    return data


# =========================
# Graph preparation utilities
# =========================

def with_pos2(data):
    """Append 2D position (pos[:,:2]->pos2) into node features x to make 7D (orig 5 + pos2)."""
    x_orig = data.x
    pos = None
    if hasattr(data, 'pos') and data.pos is not None:
        pos = data.pos
    else:
        # Fallback: try denorm from x using x_norm_params if provided
        if hasattr(data, 'x_norm_params') and data.x_norm_params is not None:
            mean = torch.as_tensor(data.x_norm_params['mean'][:3], dtype=x_orig.dtype, device=x_orig.device)
            scale = torch.as_tensor(data.x_norm_params['scale'][:3], dtype=x_orig.dtype, device=x_orig.device)
            pos = x_orig[:, :3] * scale + mean
        else:
            pos = x_orig[:, :3]
    pos2 = pos[:, :2]
    x_cat = torch.cat([x_orig, pos2], dim=1)
    new = data.clone()
    new.x = x_cat
    new.pos2_appended = True
    return new


def get_surface_mask(d):
    """Return a boolean mask identifying surface (wall) nodes."""
    if hasattr(d, 'surf') and isinstance(d.surf, torch.Tensor) and d.surf.dtype == torch.bool:
        return d.surf.view(-1)
    # Use de-normalized features if normalization parameters are present
    x = d.x
    if hasattr(d, 'x_norm_params') and d.x_norm_params is not None:
        try:
            mean = torch.as_tensor(d.x_norm_params['mean'], dtype=x.dtype, device=x.device)
            scale = torch.as_tensor(d.x_norm_params['scale'], dtype=x.dtype, device=x.device)
            if mean.ndim == 1 and scale.ndim == 1 and mean.numel() == x.size(1) and scale.numel() == x.size(1):
                x = x * scale + mean
        except Exception:
            # Fallback to using normalized x if anything goes wrong
            pass
    if x is not None and x.size(1) >= 5:
        wall = x[:, 2]
        nxy = x[:, 3:5]
        return (wall < 1e-6) | (nxy.abs().sum(dim=1) > 0)
    elif x is not None and x.size(1) >= 3:
        wall = x[:, 2]
        return (wall < 1e-6)
    else:
        return torch.zeros(d.x.size(0), dtype=torch.bool, device=d.x.device)


def ensure_edge_features(d, want_dim: int = 4):
    """Ensure edge_attr is exactly [dx, dy, dist, is_surface_pair].
    - If edge_attr exists and already has 4 columns, keep it.
    - Else compute from pos/x and edge_index following the strict spec.
    """
    if hasattr(d, 'edge_attr') and d.edge_attr is not None and d.edge_attr.dim() == 2 and d.edge_attr.size(1) == want_dim:
        return d
    if not hasattr(d, 'edge_index') or d.edge_index is None or d.edge_index.numel() == 0:
        return d
    # positions
    if hasattr(d, 'pos') and d.pos is not None:
        pos = d.pos
    else:
        pos = d.x[:, :3]
    device = pos.device
    N = pos.size(0)
    row, col = d.edge_index
    diff = pos[col, :2] - pos[row, :2]
    dist = torch.norm(diff, dim=1, keepdim=True)
    # Surface detection (strict per spec)
    surface_mask = torch.zeros(N, dtype=torch.bool, device=device)
    if hasattr(d, 'surf') and d.surf is not None:
        surface_mask = d.surf.bool().view(-1)
    elif hasattr(d, 'x') and d.x is not None and d.x.size(1) >= 3:
        wall_dist = d.x[:, 2]
        surface_mask = (wall_dist < 1e-4)
    is_surface_pair = (surface_mask[row] & surface_mask[col]).float().unsqueeze(1)
    edge_attr = torch.cat([diff, dist, is_surface_pair], dim=1)
    d.edge_attr = edge_attr
    return d


def prep_graph(g):
    """Prepare a graph: validate features and ensure edge attributes exist."""
    gg = g
    if gg.x is None:
        raise ValueError('Graph missing node features x.')
    # Keep node features as original 5D here; do NOT append pos into x at this stage.
    # Only ensure edge features exist; they will be computed from d.pos (preferred) or x[:,:3].
    gg = ensure_edge_features(gg, want_dim=4)
    return gg


def validate_edges(ds, name='train'):
    """Validate edge_index and edge_attr consistency across a dataset."""
    n = len(ds)
    bad = 0
    for i in range(n):
        d = ds[i]
        ei = getattr(d, 'edge_index', None)
        ea = getattr(d, 'edge_attr', None)
        N = d.x.size(0)
        if ei is None or ea is None:
            bad += 1; print(f'[{name}] {i}: missing edges')
            continue
        if ei.dtype != torch.long or ei.size(0) != 2 or ei.size(1) == 0:
            bad += 1; print(f'[{name}] {i}: bad edge_index shape {tuple(ei.shape)} / dtype {ei.dtype}')
        if int(ei.min()) < 0 or int(ei.max()) >= N:
            bad += 1; print(f'[{name}] {i}: edge_index out of range [0,{N-1}] -> ({int(ei.min())},{int(ei.max())})')
        if ea.dim() != 2 or ea.size(0) != ei.size(1):
            bad += 1; print(f'[{name}] {i}: edge_attr shape mismatch {tuple(ea.shape)} vs E={ei.size(1)}')
    print(f'[validate] {name}: total={n} bad={bad}')


def _prep_graph_for_norm(g):
    """Optimized version that minimizes expensive operations"""

    # Check if we need to do anything at all
    needs_pos2 = (g.x is not None and g.x.size(1) == 5 and
                  not getattr(g, 'pos2_appended', False))

    has_correct_edges = (hasattr(g, 'edge_attr') and g.edge_attr is not None and
                        g.edge_attr.dim() == 2 and g.edge_attr.size(1) == 4)

    # If no changes needed, return original (avoid clone)
    if not needs_pos2 and has_correct_edges:
        return g

    # Only clone if we need to make changes
    d = g.clone()

    # Append pos2 if needed
    if needs_pos2:
        try:
            d = with_pos2(d)
        except Exception:
            pass

    # Ensure edge features only if needed
    if not has_correct_edges:
        d = ensure_edge_features(d, want_dim=4)

    return d


def _to_xy(t: torch.Tensor):
    return t[:, :2].detach().cpu().float().numpy()


def _poly_from_surface(data: Data):
    try:
        mask = get_surface_mask(data)
        src = None
        if hasattr(data, 'pos') and data.pos is not None:
            src = data.pos
        elif hasattr(data, 'x') and data.x is not None:
            src = data.x
        else:
            return None
        pts = src[:, :2]
        pts = pts[mask].detach().cpu().numpy()
        if pts.shape[0] < 3: return None
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        order = np.argsort(ang)
        return Path(pts[order], closed=True)
    except Exception:
        return None


# === Helpers to fix physics prerequisites per batch ===

@torch.no_grad()
def ensure_edge_attr_phys3(data) -> None:
    """Ensure data.edge_attr_dxdy has shape [E, 3] = [dx, dy, dist] for physics.
    - Never overwrite data.edge_attr (model might expect 5D).
    - Preferred sources: existing edge_attr_dxdy (2D) -> add dist; else edge_attr last 2 dims; else pos/edge_index.
    """
    try:
        # If already good, return
        if hasattr(data, 'edge_attr_dxdy') and data.edge_attr_dxdy is not None:
            ea = data.edge_attr_dxdy
            if ea.dim() == 2 and ea.shape[1] == 3:
                return
        # Try from existing edge_attr_dxdy (2D)
        if hasattr(data, 'edge_attr_dxdy') and data.edge_attr_dxdy is not None and data.edge_attr_dxdy.dim() == 2 and data.edge_attr_dxdy.shape[1] >= 2:
            dx = data.edge_attr_dxdy[:, 0]
            dy = data.edge_attr_dxdy[:, 1]
            dist = (dx.pow(2) + dy.pow(2)).sqrt() + 1e-12
            data.edge_attr_dxdy = torch.stack((dx, dy, dist), dim=1)
            return
        # Else try from model edge_attr (spec: [dx, dy, dist, is_surface_pair])
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.dim() == 2 and data.edge_attr.shape[1] >= 3:
            dx = data.edge_attr[:, 0]
            dy = data.edge_attr[:, 1]
            dist = data.edge_attr[:, 2].abs() + 1e-12
            data.edge_attr_dxdy = torch.stack((dx, dy, dist), dim=1)
            return
        # Else compute from pos and edge_index if available
        if hasattr(data, 'pos') and data.pos is not None and hasattr(data, 'edge_index') and data.edge_index is not None:
            src, dst = data.edge_index[0], data.edge_index[1]
            dxy = data.pos[dst, :2] - data.pos[src, :2]
            dx = dxy[:, 0]
            dy = dxy[:, 1]
            dist = torch.linalg.norm(dxy, dim=1) + 1e-12
            data.edge_attr_dxdy = torch.stack((dx, dy, dist), dim=1)
            return
        # Give up silently if none available
        return
    except Exception:
        return


@torch.no_grad()
def sanitize_norm_params(data, k: int, *, for_y: bool = True) -> None:
    """Ensure data.{y|x}_norm_params['mean'|'scale'] tensors have length k.
    - If longer, slice to k.
    - If shorter/missing, pad with last value (or 0 for mean, 1 for scale if empty).
    In-place update. No-op on errors.
    """
    try:
        attr = 'y_norm_params' if for_y else 'x_norm_params'
        if not hasattr(data, attr):
            return
        params = getattr(data, attr)
        if not isinstance(params, dict):
            return
        target_device: Optional[torch.device] = None
        if for_y and hasattr(data, 'y') and isinstance(data.y, torch.Tensor):
            target_device = data.y.device
        elif (not for_y) and hasattr(data, 'x') and isinstance(data.x, torch.Tensor):
            target_device = data.x.device
        for key in ('mean', 'scale'):
            if key not in params:
                continue
            t = params[key]
            if not isinstance(t, torch.Tensor):
                try:
                    t = torch.as_tensor(t)
                except Exception:
                    continue
            if t.ndim > 1:
                t = t.view(-1)
            if target_device is not None:
                t = t.to(device=target_device)
            n = int(t.numel())
            if n == k:
                params[key] = t
                continue
            if n > k:
                params[key] = t[:k]
                continue
            if key == 'mean':
                pad_val = t[-1].item() if n > 0 else 0.0
            else:
                pad_val = t[-1].item() if n > 0 else 1.0
            pad = torch.full((k - n,), float(pad_val), dtype=t.dtype, device=t.device)
            params[key] = torch.cat([t, pad], dim=0)
        setattr(data, attr, params)
    except Exception:
        return


def _extract_stats_from_scaler(scaler):
    """Return (mean, std) as 1D tensors from scaler which has .mean and .std tensors."""
    if scaler is None:
        return None, None
    mean = getattr(scaler, 'mean', None)
    std = getattr(scaler, 'std', None)
    if isinstance(mean, torch.Tensor):
        mean = mean.detach().cpu().flatten()
    if isinstance(std, torch.Tensor):
        std = std.detach().cpu().flatten()
    return mean, std


def collate_pyg_with_norm_params(batch):
    """PyG 배치 생성 시 정규화 파라미터를 올바르게 처리"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # 정규화 파라미터 저장
    norm_params = {}
    if hasattr(batch[0], 'y_norm_params'):
        norm_params['y_norm_params'] = batch[0].y_norm_params
    if hasattr(batch[0], 'x_norm_params'):
        norm_params['x_norm_params'] = batch[0].x_norm_params

    # 기본 PyG 배치 생성
    batched = Batch.from_data_list(batch)

    # 정규화 파라미터 복원
    for key, value in norm_params.items():
        setattr(batched, key, value)

    return batched
