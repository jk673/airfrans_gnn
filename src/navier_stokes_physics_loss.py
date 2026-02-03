# navier_stokes_physics_loss.py
# ------------------------------------------------------------
# Self-contained physics loss for steady incompressible (RANS-like)
# - Conservative divergence with dual-area/perimeter normalization
# - Weighted gradient/Laplacian (RBF or 1/r^2)
# - Skew-symmetric convection
# - Eddy viscosity with positivity (softplus)
# - Optional BC penalties via masks
# - Dynamic U_ref (from data) & 1/Re = nu/(U_ref*L_ref)
# - Legacy attribute name aliases (continuity_loss_weight, etc.)
# ------------------------------------------------------------

from __future__ import annotations
import math
from typing import Dict, Optional, Any, Tuple

from matplotlib.pyplot import step
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add





def _half_edges(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
    """Keep one direction per undirected pair using row<col mask."""
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
    Return dx, dy, length per edge (E,).
    Supports two schemas:
      (A) default/dist-dir: [dist, dir_x, dir_y]  where dist>0, |dir|≈1
      (B) dxdy schema      : [dx, dy, dist]
    Fallback to pos if schema unknown.
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


def _node_perimeter_or_area(
    edge_index: torch.Tensor,
    edge_length: torch.Tensor,
    num_nodes: int,
    mode: str = "area",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Approximate dual area from incident edge lengths if true node_area is absent.
    - perimeter: sum of incident edge lengths (row+col)
    - area: (perimeter^2)/(4*pi) as a crude proxy
    """
    row, col = edge_index
    per = scatter_add(edge_length, row, dim=0, dim_size=num_nodes)
    per = per + scatter_add(edge_length, col, dim=0, dim_size=num_nodes)
    per = per.clamp_min(eps)
    if mode == "perimeter":
        return per
    area = (per * per) / (4.0 * math.pi)
    return area.clamp_min(eps)


# =========================
# Discrete differential ops
# =========================

def conservative_divergence(
     velocity: torch.Tensor,              # [N,2]
     edge_index: torch.Tensor,            # [2,E]
     edge_attr: torch.Tensor,             # [E,edge_dim]
     num_nodes: int,
     *,
     pos: Optional[torch.Tensor] = None,
     prefer_dxdy: bool = True,
     node_area: Optional[torch.Tensor] = None,
     use_perimeter_norm: bool = False,
     eps: float = 1e-12,
     Lref: float = 1.0,
     area_floor_factor: float = 0.0,
     min_degree: int = 0,
) -> torch.Tensor:
    """
    Conservative divergence: sum of face-normal fluxes / dual area.
    Flux per edge uses face-centered velocity (avg of endpoints).
    """
    device = velocity.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    edge_attr = edge_attr.to(device=device, dtype=velocity.dtype)
    if pos is not None:
        pos = pos.to(device=device, dtype=velocity.dtype)

    N = velocity.size(0)
    if edge_index.numel() == 0 or edge_attr.numel() == 0:
        return torch.zeros(N, device=device, dtype=velocity.dtype)

    # Filter invalid
    valid = _valid_edges(edge_index, N)
    if not torch.all(valid):
        edge_index = edge_index[:, valid]
        edge_attr = edge_attr[valid]
        if edge_index.numel() == 0:
            return torch.zeros(N, device=device, dtype=velocity.dtype)

    # Keep originals for degree
    row_full, col_full = edge_index
    # Halve edges (undirected) for flux assembly
    edge_index, edge_attr = _half_edges(edge_index, edge_attr)
    row, col = edge_index

    dx, dy, length = _extract_dxdy_length(edge_index, edge_attr, pos, prefer_dxdy, eps)
    if Lref != 1.0:  # make geometry dimensionless to match u*
        _s = 1.0 / max(Lref, 1e-12)
        length = length * _s
        if node_area is not None:
            node_area = node_area.to(velocity.device, velocity.dtype) * (_s * _s)

    nx = dy / length
    ny = -dx / length

    ui = velocity[row]       # [E,2]
    uj = velocity[col]
    u_face = 0.5 * (ui + uj)
    flux = (u_face[:, 0] * nx + u_face[:, 1] * ny) * length  # [E]

    # Sum to nodes with sign
    div = scatter_add(flux, row, dim=0, dim_size=N) - scatter_add(flux, col, dim=0, dim_size=N)

    # Normalize by dual area (preferred) or perimeter
    if node_area is None:
        denom = _node_perimeter_or_area(edge_index, length, N, mode=("perimeter" if use_perimeter_norm else "area"))
    else:
        denom = node_area.to(device=device, dtype=velocity.dtype).clamp_min(eps)

    # ---- NEW: area floor based on median length ----
    if area_floor_factor > 0.0:
        med_len = torch.quantile(length.detach(), 0.5)
        area_floor = area_floor_factor * (med_len ** 2 + eps)
        denom = torch.maximum(denom, area_floor)

    out = div / denom

    # ---- NEW: mask out very low-degree nodes ----
    if min_degree > 0:
        # use full (unhalved) degree
        deg = torch.bincount(row_full, minlength=N) + torch.bincount(col_full, minlength=N)
        mask = deg >= min_degree
        out = out * mask.to(out.dtype)

    return out


def weighted_gradient(
    field: torch.Tensor,                 # [N]
    edge_index: torch.Tensor,            # [2,E]
    edge_attr: torch.Tensor,             # [E,edge_dim]
    num_nodes: int,
    *,
    pos: Optional[torch.Tensor] = None,
    prefer_dxdy: bool = True,
    weight_mode: str = "rbf",
    eps: float = 1e-12,
    Lref: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ∇f(i) ≈ sum_j w_ij (f_j-f_i) d_ij / r_ij^2  / sum_j w_ij
    Edge-halving & symmetric accumulation (row+col).
    """
    device = field.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    edge_attr = edge_attr.to(device=device, dtype=field.dtype)
    if pos is not None:
        pos = pos.to(device=device, dtype=field.dtype)

    N = field.size(0)
    if edge_index.numel() == 0 or edge_attr.numel() == 0:
        z = torch.zeros(N, device=device, dtype=field.dtype)
        return z, z

    valid = _valid_edges(edge_index, N)
    if not torch.all(valid):
        edge_index = edge_index[:, valid]
        edge_attr = edge_attr[valid]
        if edge_index.numel() == 0:
            z = torch.zeros(N, device=device, dtype=field.dtype)
            return z, z

    edge_index, edge_attr = _half_edges(edge_index, edge_attr)
    row, col = edge_index

    dx, dy, length = _extract_dxdy_length(edge_index, edge_attr, pos, prefer_dxdy, eps)
    if Lref != 1.0:
        s = 1.0 / max(Lref, 1e-12)
        dx, dy, length = dx * s, dy * s, length * s

    if weight_mode == "rbf":
        h2 = (length.mean() ** 2).clamp_min(eps)
        w = torch.exp(-(length * length) / (h2 + eps))
    else:
        w = 1.0 / (length * length + eps)

    df = (field[col] - field[row])  # [E]
    inv_r2 = 1.0 / (length * length + eps)

    gx_edge = w * df * (dx * inv_r2)  # [E]
    gy_edge = w * df * (dy * inv_r2)

    # Symmetric accumulation
    num_x = scatter_add(gx_edge, row, dim=0, dim_size=N) + scatter_add(gx_edge, col, dim=0, dim_size=N)
    num_y = scatter_add(gy_edge, row, dim=0, dim_size=N) + scatter_add(gy_edge, col, dim=0, dim_size=N)
    den = scatter_add(w, row, dim=0, dim_size=N) + scatter_add(w, col, dim=0, dim_size=N)
    den = den.clamp_min(1.0)

    return num_x / den, num_y / den


def weighted_laplacian(
    field: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int,
    *,
    pos: Optional[torch.Tensor] = None,
    prefer_dxdy: bool = True,
    weight_mode: str = "rbf",
    eps: float = 1e-12,
    Lref: float = 1.0,
) -> torch.Tensor:
    """
    Simple scalar Laplacian via divergence of gradient (2-pass weighted gradient).
    Δf ≈ ∂/∂x(gx) + ∂/∂y(gy)  using the same weighted scheme.
    """
    gx, gy = weighted_gradient(field, edge_index, edge_attr, num_nodes,
                               pos=pos, prefer_dxdy=prefer_dxdy, weight_mode=weight_mode, eps=eps, Lref=Lref)
    dgxdx, _ = weighted_gradient(gx, edge_index, edge_attr, num_nodes,
                                 pos=pos, prefer_dxdy=prefer_dxdy, weight_mode=weight_mode, eps=eps, Lref=Lref)
    _, dgydy = weighted_gradient(gy, edge_index, edge_attr, num_nodes,
                                 pos=pos, prefer_dxdy=prefer_dxdy, weight_mode=weight_mode, eps=eps, Lref=Lref)
    return dgxdx + dgydy


# =========================
# Physics Loss (complete)
# =========================

class NavierStokesPhysicsLoss(nn.Module):
    """
    Physics-informed loss for steady incompressible (RANS-like with eddy viscosity).
    Targets/Predictions convention: columns = [u, v, p_tilde, nu_t] (nu_t optional)

    Scaling (dimensionless):
      x,y        -> / L_ref
      u,v        -> / U_ref
      p_tilde    -> / U_ref^2      (i.e., p/(rho*U_ref^2))
      nu_t       -> / (U_ref*L_ref)
    """

    # === compatibility aliases (legacy attribute names) ===
    @property
    def data_loss_weight(self) -> float:
        return self.data_w
    @data_loss_weight.setter
    def data_loss_weight(self, v: float):
        self.data_w = float(v)

    @property
    def continuity_loss_weight(self) -> float:
        return self.cont_w
    @continuity_loss_weight.setter
    def continuity_loss_weight(self, v: float):
        self.cont_w = float(v)

    @property
    def momentum_loss_weight(self) -> float:
        return self.mom_w
    @momentum_loss_weight.setter
    def momentum_loss_weight(self, v: float):
        self.mom_w = float(v)

    @property
    def bc_loss_weight(self) -> float:
        return self.bc_w
    @bc_loss_weight.setter
    def bc_loss_weight(self, v: float):
        self.bc_w = float(v)

    def __init__(
        self,
        data_loss_weight: float = 1.0,
        bc_loss_weight: float = 0.0,

        # continuity
        continuity_loss_weight: float = 0.10,   # 연속항 시작값
        continuity_target_weight: float = 0.15, # 연속항 목표값

        # momentum
        momentum_loss_weight: float = 0.05,     # 모멘텀 시작값
        momentum_target_weight: float = 0.20,   # 모멘텀 목표값

        # ramp schedule
        curriculum_ramp_steps: int = 0,         # 공용 램프 스텝(0이면 비활성)
        cont_curriculum_ramp_steps: int = -1,   # 연속항 전용 스텝(-1이면 공용 사용)
        mom_curriculum_ramp_steps: int = -1,    # 모멘텀 전용 스텝(-1이면 공용 사용)
        ramp_mode: str = "linear",        

        # NEW: ramp start (in global steps)
        ramp_start_step: int = 0,          # 공용 시작 스텝(0=즉시 시작)
        cont_ramp_start_step: int = -1,    # 개별 시작 스텝(-1=공용 사용)
        mom_ramp_start_step: int = -1,     # 개별 시작 스텝(-1=공용 사용)

        # physics
        reynolds_number: float = 1e6,          # used if dynamic_re_from_data=False
        chord_length: float = 1.0,
        freestream_velocity: float = 1.0,      # default U_ref if dynamic_uref_from_data=False
        use_skew_symmetric_convection: bool = True,
        prefer_dxdy: bool = True,
        weight_mode: str = "rbf",
        use_perimeter_norm_for_div: bool = False,
        use_huber_for_physics: bool = False,
        huber_delta: float = 0.01,

        # --- dynamic U∞/Re options ---
        dynamic_uref_from_data: bool = True,   # U_ref를 data로부터 추정(U∞)
        dynamic_re_from_data: bool = True,     # 1/Re = nu/(U_ref * L_ref)
        nu_molecular: float = 1.5e-5,          # [m^2/s] 예시(공기). 데이터 단위에 맞게 조정
        uinf_from: str = "inlet",              # "inlet" | "farfield" | "robust"

        # --- debug options ---
        debug: bool = False,
        debug_level: int = 1,         # 0=off, 1=요약, 2=상세
        debug_every: int = 100,       # step 간격(학습시), 평가엔 매 배치 실행

        # --- stability knobs for divergence ---
        div_area_floor_factor: float = 0.25,  # area ≥ factor * (median_edge_len/Lref)^2
        div_min_degree: int = 2,              # deg<2 노드의 div는 무시(0)
    ):
        super().__init__()

        # Internal weights must always be floats
        self.data_w = float(data_loss_weight)
        self.bc_w   = float(bc_loss_weight)

        # continuity
        self.cont_w0 = float(continuity_loss_weight)
        self.cont_w_target = float(continuity_target_weight)

        # momentum
        self.mom_w0 = float(momentum_loss_weight)
        self.mom_w_target = float(momentum_target_weight)

        # schedules
        self.curr_steps = int(curriculum_ramp_steps)
        self.cont_curr_steps = int(cont_curriculum_ramp_steps)
        self.mom_curr_steps  = int(mom_curriculum_ramp_steps)
        self.ramp_start_step = int(ramp_start_step)
        self.cont_ramp_start_step = int(cont_ramp_start_step)
        self.mom_ramp_start_step  = int(mom_ramp_start_step)
        self.ramp_mode = str(ramp_mode)

        self.Re = reynolds_number
        self.Lref = chord_length
        self.Uref = freestream_velocity

        self.use_skew = use_skew_symmetric_convection
        self.prefer_dxdy = prefer_dxdy
        self.weight_mode = weight_mode
        self.use_perimeter_norm_for_div = use_perimeter_norm_for_div
        self.use_huber = use_huber_for_physics
        self.huber_delta = huber_delta

        # Dynamic U∞/Re options
        self.dynamic_uref_from_data = bool(dynamic_uref_from_data)
        self.dynamic_re_from_data   = bool(dynamic_re_from_data)
        self.nu_molecular = float(nu_molecular)
        self.uinf_from = str(uinf_from)
        self.div_area_floor_factor = float(div_area_floor_factor)
        self.div_min_degree = int(div_min_degree)

        self.mse = nn.MSELoss(reduction='mean')
        # debug
        self.debug = bool(debug)
        self.debug_level = int(debug_level)
        self.debug_every = max(1, int(debug_every))
        self.last_debug: Optional[dict] = None

        # ---------- debug helpers ----------
    @staticmethod
    def _q(x: torch.Tensor, q: float) -> float:
        x = x.detach()
        try:
            return float(torch.quantile(x, q))
        except Exception:
            # torch<1.7 호환 등
            k = max(1, int(q * x.numel()))
            return float(x.flatten().kthvalue(k).values)

    @staticmethod
    def _stat1d(x: torch.Tensor) -> dict:
        x = x.detach()
        if x.numel() == 0:
            return dict(n=0, mean=float('nan'), std=float('nan'), min=float('nan'),
                        max=float('nan'), p50=float('nan'), p90=float('nan'), p99=float('nan'),
                        nan=int(0), inf=int(0))
        nan = torch.isnan(x).sum().item()
        inf = torch.isinf(x).sum().item()
        xm = x[~torch.isnan(x) & ~torch.isinf(x)]
        if xm.numel() == 0:
            return dict(n=int(x.numel()), mean=float('nan'), std=float('nan'), min=float('nan'),
                        max=float('nan'), p50=float('nan'), p90=float('nan'), p99=float('nan'),
                        nan=int(nan), inf=int(inf))
        return dict(
            n=int(x.numel()),
            mean=float(xm.mean()),
            std=float(xm.std(unbiased=False)),
            min=float(xm.min()),
            max=float(xm.max()),
            p50=float(torch.median(xm)),
            p90=float(torch.quantile(xm, 0.90)),
            p99=float(torch.quantile(xm, 0.99)),
            nan=int(nan),
            inf=int(inf),
        )

    def _collect_geom_stats(self, data, pos_scaled, edge_attr, Lref) -> dict:
        device = pos_scaled.device
        N = data.num_nodes
        E = data.edge_index.size(1)
        # edge length (이미 conservative_divergence에서 1/Lref로 스케일하도록 패치했다면
        # 여기서는 "물리 길이" 대비를 보려고 raw length도 같이 수집)
        row, col = data.edge_index
        if edge_attr is not None and edge_attr.size(1) >= 3:
            # 해석: default [dist, dirx, diry] 또는 [dx, dy, dist]
            dist_col0_med = edge_attr[:, 0].abs().median()
            is_default = (edge_attr[:, 1].abs().median() <= 1.5) and torch.all(edge_attr[:, 0] > 0)
            if is_default:
                length_phys = edge_attr[:, 0]
            else:
                length_phys = edge_attr[:, 2].abs()
        else:
            dvec = data.pos[col, :2] - data.pos[row, :2]
            length_phys = dvec.norm(dim=1)
        stats = {
            'N': int(N),
            'E': int(E),
            'edge_len_phys': self._stat1d(length_phys),
        }
        na = getattr(data, 'node_area', None)
        if na is not None:
            stats['node_area_phys'] = self._stat1d(na)
        deg = torch.bincount(row, minlength=N) + torch.bincount(col, minlength=N)
        stats['degree'] = self._stat1d(deg.float())
        # masks
        for k in ['is_wall','is_inlet','is_outlet','is_farfield']:
            v = getattr(data, k, None)
            if v is not None:
                stats[f'cnt_{k}'] = int(v.sum().item())
        return stats

    def _ramp_factor(self, step: Optional[int], total_steps: int, start_step: int = 0) -> float:
        if step is None or total_steps <= 0:
            return 1.0
        
        # 시작 전에는 0, 시작 이후에는 0..1로 선형/코사인 증가
        if step < start_step:
           return 0.0
        t = max(0.0, min(1.0, float(step - start_step)/float(total_steps)))

        if self.ramp_mode == "cosine":
            import math
            return 0.5 - 0.5*math.cos(math.pi*t)
        
        return t



    def _get_x_phys(self, data, device):
        """
        Return denormalized features x_phys if x_norm_params exist; otherwise raw x.
        If data.x is missing, return None.
        """
        if getattr(data, "x", None) is None:
            return None
        x = data.x.to(device)
        xnp = getattr(data, "x_norm_params", None)
        if xnp is not None:
            return self._denorm(x, xnp)  # denormalize first
        return x  # raw features (assumed physical already)
    

    # ---------- scaling helpers ----------
    def _apply_dimensional_scaling_with_Uref(
        self,
        preds_phys: torch.Tensor,
        targs_phys: Optional[torch.Tensor],
        pos_phys: torch.Tensor,
        Uref_local: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        U = float(max(Uref_local, 1e-12))
        L = float(max(self.Lref, 1e-12))

        pos_scaled = pos_phys / L

        pred_scaled = preds_phys.clone()
        if pred_scaled.size(1) >= 2:
            pred_scaled[:, :2] = pred_scaled[:, :2] / U
        if pred_scaled.size(1) >= 3:
            pred_scaled[:, 2] = pred_scaled[:, 2] / (U ** 2)
        if pred_scaled.size(1) >= 4:
            pred_scaled[:, 3] = pred_scaled[:, 3] / (U * L)

        targ_scaled = None
        if targs_phys is not None:
            targ_scaled = targs_phys.clone()
            if targ_scaled.size(1) >= 2:
                targ_scaled[:, :2] = targ_scaled[:, :2] / U
            if targ_scaled.size(1) >= 3:
                targ_scaled[:, 2] = targ_scaled[:, 2] / (U ** 2)
            if targ_scaled.size(1) >= 4:
                targ_scaled[:, 3] = targ_scaled[:, 3] / (U * L)

        return pred_scaled, targ_scaled, pos_scaled

    def _denorm(self, x: torch.Tensor, norm_params: Dict) -> torch.Tensor:
        device = x.device
        mean = norm_params['mean']
        scale = norm_params.get('scale', norm_params.get('std', None))
        if isinstance(mean, (list, tuple)):
            mean = torch.tensor(mean, dtype=torch.float32, device=device)
        elif not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32, device=device)
        if isinstance(scale, (list, tuple)):
            scale = torch.tensor(scale, dtype=torch.float32, device=device)
        elif not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float32, device=device)
        if x.ndim == 2 and mean.ndim == 1:
            mean = mean.unsqueeze(0)
            scale = scale.unsqueeze(0)
        return x * scale + mean

    # ---------- robust penalty ----------
    def _quad_or_huber(self, r: torch.Tensor) -> torch.Tensor:
        if not self.use_huber:
            return (r * r).mean()
        d = self.huber_delta
        abs_r = r.abs()
        quad = 0.5 * (abs_r ** 2)
        lin = d * (abs_r - 0.5 * d)
        out = torch.where(abs_r <= d, quad, lin)
        return out.mean()

    # ---------- infer U∞ from data.x (denormalized) ----------
    def _infer_Uinf(self, data: Any, x_phys: torch.Tensor) -> float:
        """
        x_phys[:,0:2] = (denormalized) freestream velocity features assumed.
        U∞ = ||(u∞_x, u∞_y)||. Prefer inlet/farfield masks; fallback to 90th percentile non-wall.
        """
        device = x_phys.device
        if x_phys.size(1) < 2:
            return float(max(self.Uref, 1e-12))
        U = x_phys[:, :2]
        mag = U.norm(dim=1)

        if self.uinf_from == "inlet" and getattr(data, "is_inlet", None) is not None and int(data.is_inlet.sum()) > 0:
            m = data.is_inlet.bool().to(device)
            return float(mag[m].median().clamp_min(1e-12))

        if self.uinf_from == "farfield" and getattr(data, "is_farfield", None) is not None and int(data.is_farfield.sum()) > 0:
            m = data.is_farfield.bool().to(device)
            return float(mag[m].median().clamp_min(1e-12))

        # robust: exclude walls, take high quantile
        mask = torch.ones_like(mag, dtype=torch.bool, device=device)
        if getattr(data, "is_wall", None) is not None:
            mask &= ~data.is_wall.bool().to(device)
        vals = mag[mask] if mask.any() else mag
        if vals.numel() == 0:
            vals = mag
        q = torch.quantile(vals, 0.90)
        return float(q.clamp_min(1e-12))

    # ---------- continuity ----------
    def _continuity_loss(self, u_scaled: torch.Tensor, data: Any, pos_scaled: torch.Tensor) -> torch.Tensor:
        edge_index = data.edge_index.to(u_scaled.device)
        edge_attr = getattr(data, 'edge_attr_dxdy', getattr(data, 'edge_attr', None))
        if edge_attr is None:
            raise ValueError("Data must have edge_attr or edge_attr_dxdy.")

        node_area = getattr(data, 'node_area', None)
        div = conservative_divergence(
            velocity=u_scaled[:, :2],
            edge_index=edge_index,
            edge_attr=edge_attr.to(u_scaled.device),
            num_nodes=u_scaled.size(0),
            pos=pos_scaled,
            prefer_dxdy=self.prefer_dxdy,
            node_area=node_area,
            use_perimeter_norm=self.use_perimeter_norm_for_div,
            Lref=self.Lref,
            area_floor_factor=self.div_area_floor_factor,
            min_degree=self.div_min_degree,
        )
        return self._quad_or_huber(div)

    # ---------- momentum ----------
    def _momentum_loss(self, pred_scaled: torch.Tensor, data: Any, pos_scaled: torch.Tensor, mol_coeff: float) -> torch.Tensor:
        edge_index = data.edge_index.to(pred_scaled.device)
        edge_attr = getattr(data, 'edge_attr_dxdy', getattr(data, 'edge_attr', None))
        if edge_attr is None:
            raise ValueError("Data must have edge_attr or edge_attr_dxdy.")
        num_nodes = pred_scaled.size(0)
        device = pred_scaled.device

        # u,v,p_tilde, nu_t (nu_t optional)
        u = pred_scaled[:, 0]
        v = pred_scaled[:, 1]
        p = pred_scaled[:, 2] if pred_scaled.size(1) >= 3 else torch.zeros_like(u)
        nu_t = pred_scaled[:, 3] if pred_scaled.size(1) >= 4 else torch.zeros_like(u)
        # ensure non-negativity
        nu_t = F.softplus(nu_t)

        # gradients
        dudx, dudy = weighted_gradient(u, edge_index, edge_attr.to(device), num_nodes,
                                       pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode, Lref=self.Lref)
        dvdx, dvdy = weighted_gradient(v, edge_index, edge_attr.to(device), num_nodes,
                                       pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode, Lref=self.Lref)
        dpdx, dpdy = weighted_gradient(p, edge_index, edge_attr.to(device), num_nodes,
                                       pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode, Lref=self.Lref)

        # convective terms
        if self.use_skew:
            # standard u·∇u
            conv_u_std = u * dudx + v * dudy
            conv_v_std = u * dvdx + v * dvdy
            # divergence form ∇·(u⊗u)
            u2 = u * u
            v2 = v * v
            uv = u * v
            du2dx, _ = weighted_gradient(u2, edge_index, edge_attr.to(device), num_nodes,
                                         pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode)
            _, duvdy = weighted_gradient(uv, edge_index, edge_attr.to(device), num_nodes,
                                         pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode)
            duvdx, _ = weighted_gradient(uv, edge_index, edge_attr.to(device), num_nodes,
                                         pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode)
            _, dv2dy = weighted_gradient(v2, edge_index, edge_attr.to(device), num_nodes,
                                         pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode)
            conv_u = 0.5 * (conv_u_std + (du2dx + duvdy))
            conv_v = 0.5 * (conv_v_std + (duvdx + dv2dy))
        else:
            conv_u = u * dudx + v * dudy
            conv_v = u * dvdx + v * dvdy

        # viscous: ∇·[(1/Re + nu_t) ∇u] = (mol_coeff + nu_t) Δu + ∇nu_t · ∇u
        lap_u = weighted_laplacian(u, edge_index, edge_attr.to(device), num_nodes,
                                   pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode, Lref=self.Lref)
        lap_v = weighted_laplacian(v, edge_index, edge_attr.to(device), num_nodes,
                                   pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode, Lref=self.Lref)
        dnutdx, dnutdy = weighted_gradient(nu_t, edge_index, edge_attr.to(device), num_nodes,
                                           pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode)

        visc_u = (mol_coeff + nu_t) * lap_u + dnutdx * dudx + dnutdy * dudy
        visc_v = (mol_coeff + nu_t) * lap_v + dnutdx * dvdx + dnutdy * dvdy

        # residuals: conv + grad(p) - visc = 0
        res_u = conv_u + dpdx - visc_u
        res_v = conv_v + dpdy - visc_v

        return self._quad_or_huber(torch.stack([res_u, res_v], dim=-1))

    # ---------- boundary conditions (optional) ----------
    def _bc_loss(self, pred_scaled: torch.Tensor, data: Any) -> torch.Tensor:
        """
        Soft BC penalties if masks exist in data:
        - is_wall: no-slip u=0
        - is_inlet & inlet_u (opt): u=profile
        - is_farfield: u ~ U_inf(=1 in scaled), p~0
        - is_outlet: p~0
        All masks are expected as [N] bool or 0/1 tensors.
        """
        device = pred_scaled.device
        N = pred_scaled.size(0)
        u = pred_scaled[:, :2]
        p = pred_scaled[:, 2] if pred_scaled.size(1) >= 3 else torch.zeros(N, device=device)

        loss_terms = []

        # wall: no-slip
        is_wall = getattr(data, 'is_wall', None)
        if is_wall is not None:
            mask_w = is_wall.bool().to(device)
            if mask_w.any():
                # Check dimensions match (important for batched graphs)
                if mask_w.size(0) != N:
                    print(f"Warning: is_wall mask size {mask_w.size(0)} != pred size {N}, skipping BC loss")
                    return torch.zeros(1, device=device)
                u_wall = u[mask_w]
                loss_terms.append((u_wall ** 2).mean())

        # inlet: velocity match (if inlet_u provided); else weak penalty to freestream (1,0)
        is_inlet = getattr(data, 'is_inlet', None)
        if is_inlet is not None:
            mask_in = is_inlet.bool().to(device)
            if mask_in.any():
                if mask_in.size(0) != N:
                    print(f"Warning: is_inlet mask size {mask_in.size(0)} != pred size {N}, skipping BC loss")
                    return torch.zeros(1, device=device)
                u_inlet = u[mask_in]
                inlet_u = getattr(data, 'inlet_u', None)
                if inlet_u is not None:
                    inlet_u = inlet_u.to(device)
                    # Ensure inlet_u is properly sized for batched data
                    if inlet_u.size(0) != mask_in.sum():
                        # If inlet_u is provided per-graph, we need to expand it
                        inlet_u_target = inlet_u[mask_in]
                    else:
                        inlet_u_target = inlet_u
                    loss_terms.append(((u_inlet - inlet_u_target) ** 2).mean())
                else:
                    target = torch.tensor([[1.0, 0.0]], device=device).expand_as(u_inlet)
                    loss_terms.append(0.1 * ((u_inlet - target) ** 2).mean())

        # farfield: u≈U_inf and p≈0 (weak)
        is_far = getattr(data, 'is_farfield', None)
        if is_far is not None:
            mask_f = is_far.bool().to(device)
            if mask_f.any():
                if mask_f.size(0) != N:
                    print(f"Warning: is_farfield mask size {mask_f.size(0)} != pred size {N}, skipping BC loss")
                    return torch.zeros(1, device=device)
                u_far = u[mask_f]
                p_far = p[mask_f]
                target_u = torch.tensor([[1.0, 0.0]], device=device).expand_as(u_far)
                loss_terms.append(0.1 * ((u_far - target_u) ** 2).mean())
                loss_terms.append(0.1 * (p_far ** 2).mean())

        # outlet: p≈0
        is_out = getattr(data, 'is_outlet', None)
        if is_out is not None:
            mask_o = is_out.bool().to(device)
            if mask_o.any():
                if mask_o.size(0) != N:
                    print(f"Warning: is_outlet mask size {mask_o.size(0)} != pred size {N}, skipping BC loss")
                    return torch.zeros(1, device=device)
                p_out = p[mask_o]
                loss_terms.append((p_out ** 2).mean())

        if len(loss_terms) == 0:
            return torch.zeros(1, device=device)
        return torch.stack(loss_terms).mean()

    # ---------- curriculum ramp ----------
    def _ramp(self, step_or_epoch: Optional[int]) -> float:
        if not self.curr_steps or self.curr_steps <= 0:
            return 1.0
        if step_or_epoch is None:
            return 1.0
        t = max(0, min(step_or_epoch, self.curr_steps))
        return float(t) / float(self.curr_steps)

    # ---------- forward ----------
    def forward(
        self,
        predictions: torch.Tensor,        # normalized network outputs
        targets: torch.Tensor,            # normalized labels
        data: Optional[Any] = None,
        *,
        step: Optional[int] = None,       # or epoch
    ) -> Dict[str, torch.Tensor]:
        device = predictions.device
        losses: Dict[str, torch.Tensor] = {}

        # 1) Data loss (in normalized space)
        mse_loss = self.mse(predictions, targets)
        losses["mse_loss"] = mse_loss

        # 2) Prepare physical tensors (denormalized)
        preds_phys = predictions
        targs_phys = targets

        if getattr(data, 'y_norm_params', None) is not None:
            preds_phys = self._denorm(predictions, data.y_norm_params)
            try:
                targs_phys = self._denorm(targets, data.y_norm_params)
            except Exception:
                targs_phys = None

        # Positions (physical)
        if getattr(data, 'pos', None) is not None:
            pos_phys = data.pos.to(device)
        elif getattr(data, 'x_norm_params', None) is not None:
            x_phys_tmp = self._denorm(data.x.to(device), data.x_norm_params)
            pos_phys = x_phys_tmp[:, :2]
        else:
            pos_phys = data.x[:, :2].to(device)

        # x_phys for U∞ inference (denormalized)
        x_phys = self._get_x_phys(data, device)

        # 3) Dynamic U_ref (U∞) and scaling
        Uref_local = self.Uref
        if self.dynamic_uref_from_data and x_phys is not None and x_phys.size(1) >= 2:
            try:
                Uref_local = self._infer_Uinf(data, x_phys)
                Uref_local = max(min(Uref_local, 1e3), 0.05)  # clamp for robustness
            except Exception:
                Uref_local = self.Uref  # fallback

        pred_scaled, targ_scaled, pos_scaled = self._apply_dimensional_scaling_with_Uref(
            preds_phys, targs_phys, pos_phys, Uref_local
        )

        # 4) Dynamic 1/Re coefficient for molecular viscosity
        if self.dynamic_re_from_data:
            mol_coeff = self.nu_molecular / (max(Uref_local, 1e-12) * max(self.Lref, 1e-12))
        else:
            mol_coeff = 1.0 / max(self.Re, 1e-12)

        # 5) Physics losses
        cont_loss = torch.zeros((), device=device)
        mom_loss = torch.zeros((), device=device)
        bc_loss = torch.zeros((), device=device)

        if data is not None:
            try:
                cont_loss = self._continuity_loss(pred_scaled, data, pos_scaled)
            except Exception as e:
                print(f"[physics] continuity skipped: {e}")
            try:
                mom_loss = self._momentum_loss(pred_scaled, data, pos_scaled, mol_coeff)
            except Exception as e:
                print(f"[physics] momentum skipped: {e}")
            try:
                if self.bc_w > 0.0:
                    bc_loss = self._bc_loss(pred_scaled, data)
            except Exception as e:
                print(f"[physics] bc loss skipped: {e}")

        losses["continuity_loss"] = cont_loss
        losses["momentum_loss"] = mom_loss
        if self.bc_w > 0.0:
            losses["bc_loss"] = bc_loss

        # 6) Curriculum ramp
        ramp = self._ramp(step)
        bc_w   = self.bc_w   * ramp  # ← 이후 합산에 bc_w를 실제 사용해야 함

        cont_steps = self.cont_curr_steps if self.cont_curr_steps >= 0 else self.curr_steps
        mom_steps  = self.mom_curr_steps  if self.mom_curr_steps  >= 0 else self.curr_steps
        cont_start = self.cont_ramp_start_step if self.cont_ramp_start_step >= 0 else self.ramp_start_step
        mom_start  = self.mom_ramp_start_step  if self.mom_ramp_start_step  >= 0 else self.ramp_start_step

        r_cont = self._ramp_factor(step, cont_steps, cont_start)
        r_mom  = self._ramp_factor(step, mom_steps,  mom_start)

        cont_w = self.cont_w0 + (self.cont_w_target - self.cont_w0) * r_cont
        mom_w  = self.mom_w0  + (self.mom_w_target  - self.mom_w0)  * r_mom

        total = (
            self.data_w * mse_loss
            + cont_w * cont_loss
            + mom_w  * mom_loss
            + bc_w   * bc_loss          # ← 램프 적용된 bc_w 사용
        )
        losses["total_loss"] = total     # ← 반드시 넣어줘야 트레이너가 사용 가능


        losses["cont_weight_used"] = torch.as_tensor(cont_w, device=predictions.device)
        losses["mom_weight_used"]  = torch.as_tensor(mom_w,  device=predictions.device)
        losses["bc_weight_used"]   = torch.as_tensor(bc_w,   device=predictions.device)

        losses["cont_ramp"] = torch.as_tensor(r_cont, device=predictions.device)
        losses["mom_ramp"]  = torch.as_tensor(r_mom, device=predictions.device)

        # for logging convenience
        losses["uref_used"] = torch.tensor(float(Uref_local), device=device)
        losses["mol_coeff"] = torch.tensor(float(mol_coeff), device=device)

        # 7) DEBUG: 수치/스케일 이상 탐지 및 통계 수집 (학습엔 영향 없음)
        want_debug = self.debug and (step is None or (step % self.debug_every == 0))
        if want_debug and data is not None:
            dbg = {}
            try:
                edge_attr_used = getattr(data, 'edge_attr_dxdy', getattr(data, 'edge_attr', None))
                # u*, p*, nu_t* 통계
                uvec = pred_scaled[:, :2]
                umag = uvec.norm(dim=1)
                dbg['u_mag'] = self._stat1d(umag)
                if pred_scaled.size(1) >= 3:
                    dbg['p_tilde'] = self._stat1d(pred_scaled[:, 2])
                if pred_scaled.size(1) >= 4:
                    dbg['nu_t_star'] = self._stat1d(F.softplus(pred_scaled[:, 3]))
                # 기하/면적/차수
                dbg.update(self._collect_geom_stats(data, pos_scaled, edge_attr_used, self.Lref))
                # 연속 방정식 잔차 분포
                with torch.no_grad():
                    from copy import deepcopy
                    div_dbg = conservative_divergence(
                        velocity=uvec,
                        edge_index=data.edge_index.to(device),
                        edge_attr=edge_attr_used.to(device),
                        num_nodes=pred_scaled.size(0),
                        pos=pos_scaled,
                        prefer_dxdy=self.prefer_dxdy,
                        node_area=getattr(data, 'node_area', None),
                        use_perimeter_norm=self.use_perimeter_norm_for_div,
                        Lref=self.Lref,
                    )
                    dbg['div'] = self._stat1d(div_dbg)
                # 모멘텀 항 분해(대략)
                with torch.no_grad():
                    dudx, dudy = weighted_gradient(uvec[:,0], data.edge_index.to(device), edge_attr_used.to(device), pred_scaled.size(0),
                                                   pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode)
                    dvdx, dvdy = weighted_gradient(uvec[:,1], data.edge_index.to(device), edge_attr_used.to(device), pred_scaled.size(0),
                                                   pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode)
                    if pred_scaled.size(1) >= 3:
                        dpdx, dpdy = weighted_gradient(pred_scaled[:,2], data.edge_index.to(device), edge_attr_used.to(device), pred_scaled.size(0),
                                                       pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode)
                        dbg['|grad p*|'] = self._stat1d(torch.sqrt(dpdx**2 + dpdy**2))
                    lap_u = weighted_laplacian(uvec[:,0], data.edge_index.to(device), edge_attr_used.to(device), pred_scaled.size(0),
                                                pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode)
                    lap_v = weighted_laplacian(uvec[:,1], data.edge_index.to(device), edge_attr_used.to(device), pred_scaled.size(0),
                                                pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode)
                    dbg['|grad u*|'] = self._stat1d(torch.sqrt(dudx**2 + dudy**2))
                    dbg['|grad v*|'] = self._stat1d(torch.sqrt(dvdx**2 + dvdy**2))
                    dbg['lap_u*'] = self._stat1d(lap_u)
                    dbg['lap_v*'] = self._stat1d(lap_v)
                # 스케일/유닛 체크
                dbg['uref_used'] = float(Uref_local)
                dbg['mol_coeff'] = float(mol_coeff)
                # 마스크 개수
                for k in ['is_wall','is_inlet','is_outlet','is_farfield']:
                    v = getattr(data, k, None)
                    if v is not None: dbg[f'cnt_{k}'] = int(v.sum().item())
                # 간단한 경고 휴리스틱
                warn = []
                if dbg['uref_used'] < 0.05 or dbg['uref_used'] > 1e3:
                    warn.append(f"Uref_used={dbg['uref_used']:.3g} (check x_norm_params / masks)")
                if dbg['div']['p99'] > 1e3 or math.isnan(dbg['div']['mean']):
                    warn.append(f"divergence large: p99={dbg['div']['p99']:.3g}")
                if dbg['|grad u*|']['p99'] > 1e3 or dbg['|grad v*|']['p99'] > 1e3:
                    warn.append("velocity gradients unusually large")
                if dbg['edge_len_phys']['p99'] > 10*self.Lref:
                    warn.append("edge length >> Lref (check units)")
                dbg['warnings'] = warn
                self.last_debug = dbg
                # 로그에 간단 지표만 태그(숫자만)
                losses['dbg_div_p95'] = torch.tensor(dbg['div']['p90'], device=device)
                losses['dbg_u_mag_p95'] = torch.tensor(self._q(umag, 0.90), device=device)
                losses['dbg_edge_len_p95'] = torch.tensor(dbg['edge_len_phys']['p90'], device=device)
            except Exception as _:
                pass
        return losses


if __name__ == "__main__":
    print("Navier–Stokes physics loss (dynamic U∞/Re) ready.")
