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
from typing import Optional, Tuple

import torch
from torch_scatter import scatter_add


# =========================
# Edge geometry primitives
# =========================

def _half_edges(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
    """무방향 그래프 가정에서 (row < col)인 한쪽 방향만 유지."""
    row, col = edge_index
    mask = row < col
    if mask.any():
        edge_index = edge_index[:, mask]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]
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
    device = data.pos.device
    N = data.num_nodes

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

    # face가 없을 때: perimeter 근사
    row, col = data.edge_index
    dvec = data.pos[col, :2] - data.pos[row, :2]
    length = dvec.norm(dim=1)
    per = scatter_add(length, row, dim=0, dim_size=N)
    per = per + scatter_add(length, col, dim=0, dim_size=N)
    area_proxy = (per * per) / (4.0 * math.pi)
    data.node_area = area_proxy.clamp_min(1e-12)
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
    if xfeat.size(1) > max(normal_cols):
        nx = xfeat[:, normal_cols[0]]
        ny = xfeat[:, normal_cols[1]]
        nx_ny = torch.stack([nx, ny], dim=-1)
        have_normals = (nx_ny.abs().sum(dim=1) > 0).any().item()
    else:
        nx_ny = torch.zeros(N, 2, device=device)

    # is_wall
    is_wall_from_dist = (wall_dist <= wall_dist_thresh)
    if have_normals:
        is_wall_from_norm = (nx_ny.abs().sum(dim=1) > 0)
        is_wall = is_wall_from_dist | is_wall_from_norm
    else:
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
