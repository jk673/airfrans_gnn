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
from src.utils import _valid_edges





def _half_edges(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
    """Keep one direction per undirected pair using row<col mask."""
    row, col = edge_index
    mask = row < col
    if mask.any():
        edge_index = edge_index[:, mask]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]
    return edge_index, edge_attr


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
        bc_ramp_start_step: int = -1,      # BC 시작 스텝(-1=공용 사용)
        bc_curriculum_ramp_steps: int = -1, # BC 전용 램프 스텝(-1=공용 사용)

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
        self.bc_ramp_start_step   = int(bc_ramp_start_step)
        self.bc_curr_steps        = int(bc_curriculum_ramp_steps)
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

        mean = torch.as_tensor(mean, dtype=x.dtype, device=device)
        scale = torch.as_tensor(scale, dtype=x.dtype, device=device)

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
                                         pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode, Lref=self.Lref)
            _, duvdy = weighted_gradient(uv, edge_index, edge_attr.to(device), num_nodes,
                                         pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode, Lref=self.Lref)
            duvdx, _ = weighted_gradient(uv, edge_index, edge_attr.to(device), num_nodes,
                                         pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode, Lref=self.Lref)
            _, dv2dy = weighted_gradient(v2, edge_index, edge_attr.to(device), num_nodes,
                                         pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode, Lref=self.Lref)
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
                                           pos=pos_scaled, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode, Lref=self.Lref)

        visc_u = (mol_coeff + nu_t) * lap_u + dnutdx * dudx + dnutdy * dudy
        visc_v = (mol_coeff + nu_t) * lap_v + dnutdx * dvdx + dnutdy * dvdy

        # residuals: conv + grad(p) - visc = 0
        res_u = conv_u + dpdx - visc_u
        res_v = conv_v + dpdy - visc_v

        return self._quad_or_huber(torch.stack([res_u, res_v], dim=-1))

    # ---------- boundary conditions (optional) ----------
    def _bc_loss(
        self,
        pred_scaled: torch.Tensor,
        data: Any,
        Uref_local: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
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
        U_ref = float(self.Uref) if Uref_local is None else float(Uref_local)
        U_ref = max(U_ref, 1e-12)
        scale_u = torch.as_tensor(U_ref, device=device, dtype=pred_scaled.dtype)

        zero = torch.zeros((), device=device, dtype=pred_scaled.dtype)
        bc_terms: Dict[str, torch.Tensor] = {
            "bc_wall_loss": zero,
            "bc_inlet_loss": zero,
            "bc_farfield_loss": zero,
            "bc_outlet_loss": zero,
        }
        bc_loss_terms = []

        # wall: no-slip
        is_wall = getattr(data, 'is_wall', None)
        if is_wall is not None:
            mask_w = is_wall.bool().to(device)
            if mask_w.any():
                # Check dimensions match (important for batched graphs)
                if mask_w.size(0) != N:
                    print(f"Warning: is_wall mask size {mask_w.size(0)} != pred size {N}, skipping BC loss")
                    return {
                        "bc_loss": zero,
                        "bc_wall_loss": zero,
                        "bc_inlet_loss": zero,
                        "bc_farfield_loss": zero,
                        "bc_outlet_loss": zero,
                    }
                u_wall = u[mask_w]
                wall_loss = (u_wall ** 2).mean()
                bc_terms["bc_wall_loss"] = wall_loss
                bc_loss_terms.append(wall_loss)

        # inlet: velocity match (if inlet_u provided); else weak penalty to freestream (1,0)
        is_inlet = getattr(data, 'is_inlet', None)
        if is_inlet is not None:
            mask_in = is_inlet.bool().to(device)
            if mask_in.any():
                if mask_in.size(0) != N:
                    print(f"Warning: is_inlet mask size {mask_in.size(0)} != pred size {N}, skipping BC loss")
                    return {
                        "bc_loss": zero,
                        "bc_wall_loss": bc_terms["bc_wall_loss"],
                        "bc_inlet_loss": zero,
                        "bc_farfield_loss": zero,
                        "bc_outlet_loss": zero,
                    }
                u_inlet = u[mask_in]
                n_inlet = mask_in.sum().item()
                inlet_u = getattr(data, 'inlet_u', None)
                if inlet_u is not None:
                    inlet_u = inlet_u.to(device)
                    # Shape dispatch for various upstream formats
                    if inlet_u.shape[0] == n_inlet and inlet_u.dim() == 2:
                        # Already correctly sized [n_inlet, 2]
                        inlet_u_target = inlet_u / scale_u
                    elif inlet_u.shape[0] == N and inlet_u.dim() == 2:
                        # Full-graph size [N, 2] → filter by mask
                        inlet_u_target = inlet_u[mask_in] / scale_u
                    elif inlet_u.dim() == 1 and inlet_u.shape[0] == 2:
                        # Per-graph single vector [2] → broadcast
                        inlet_u_target = inlet_u.unsqueeze(0).expand(n_inlet, -1) / scale_u
                    else:
                        import warnings
                        warnings.warn(
                            f"inlet_u shape {tuple(inlet_u.shape)} unexpected for "
                            f"n_inlet={n_inlet}, N={N}; using (1,0) fallback"
                        )
                        inlet_u_target = torch.tensor([[1.0, 0.0]], device=device, dtype=u_inlet.dtype).expand(n_inlet, -1) / scale_u
                    inlet_loss = ((u_inlet - inlet_u_target) ** 2).mean()
                    bc_terms["bc_inlet_loss"] = inlet_loss
                    bc_loss_terms.append(inlet_loss)
                else:
                    target = torch.tensor([[1.0, 0.0]], device=device, dtype=u_inlet.dtype).expand_as(u_inlet) / scale_u
                    inlet_loss = 0.1 * ((u_inlet - target) ** 2).mean()
                    bc_terms["bc_inlet_loss"] = inlet_loss
                    bc_loss_terms.append(inlet_loss)

        # farfield: u≈U_inf and p≈0 (weak)
        is_far = getattr(data, 'is_farfield', None)
        if is_far is not None:
            mask_f = is_far.bool().to(device)
            if mask_f.any():
                if mask_f.size(0) != N:
                    print(f"Warning: is_farfield mask size {mask_f.size(0)} != pred size {N}, skipping BC loss")
                    return {
                        "bc_loss": zero,
                        "bc_wall_loss": bc_terms["bc_wall_loss"],
                        "bc_inlet_loss": bc_terms["bc_inlet_loss"],
                        "bc_farfield_loss": zero,
                        "bc_outlet_loss": zero,
                    }
                u_far = u[mask_f]
                p_far = p[mask_f]
                target_u = torch.tensor([[1.0, 0.0]], device=device, dtype=u_far.dtype).expand_as(u_far) / scale_u
                farfield_loss = 0.1 * ((u_far - target_u) ** 2).mean() + 0.1 * (p_far ** 2).mean()
                bc_terms["bc_farfield_loss"] = farfield_loss
                bc_loss_terms.append(farfield_loss)

        # outlet: pressure pinning p≈0 (Dirichlet, not zero-gradient)
        is_out = getattr(data, 'is_outlet', None)
        if is_out is not None:
            mask_o = is_out.bool().to(device)
            if mask_o.any():
                if mask_o.size(0) != N:
                    print(f"Warning: is_outlet mask size {mask_o.size(0)} != pred size {N}, skipping BC loss")
                    return {
                        "bc_loss": zero,
                        "bc_wall_loss": bc_terms["bc_wall_loss"],
                        "bc_inlet_loss": bc_terms["bc_inlet_loss"],
                        "bc_farfield_loss": bc_terms["bc_farfield_loss"],
                        "bc_outlet_loss": zero,
                    }
                p_out = p[mask_o]
                outlet_loss = (p_out ** 2).mean()
                bc_terms["bc_outlet_loss"] = outlet_loss
                bc_loss_terms.append(outlet_loss)

        bc_terms["bc_loss"] = (
            torch.stack(bc_loss_terms).mean() if bc_loss_terms else zero
        )
        return bc_terms

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
                    bc_info = self._bc_loss(pred_scaled, data, Uref_local)
                    if isinstance(bc_info, dict):
                        bc_loss = bc_info.get("bc_loss", torch.zeros((), device=device))
                        losses["bc_loss"] = bc_loss
                        losses.update(bc_info)
                    else:
                        bc_loss = bc_info
                        losses["bc_loss"] = bc_loss
                        losses["bc_wall_loss"] = torch.tensor(0.0, device=device)
                        losses["bc_inlet_loss"] = torch.tensor(0.0, device=device)
                        losses["bc_outlet_loss"] = torch.tensor(0.0, device=device)
                        losses["bc_farfield_loss"] = torch.tensor(0.0, device=device)
            except Exception as e:
                print(f"[physics] bc loss skipped: {e}")

        losses["continuity_loss"] = cont_loss
        losses["momentum_loss"] = mom_loss
        if self.bc_w > 0.0:
            losses.setdefault("bc_loss", bc_loss)
            losses.setdefault("bc_wall_loss", torch.tensor(0.0, device=device))
            losses.setdefault("bc_inlet_loss", torch.tensor(0.0, device=device))
            losses.setdefault("bc_outlet_loss", torch.tensor(0.0, device=device))
            losses.setdefault("bc_farfield_loss", torch.tensor(0.0, device=device))
        if self.bc_w <= 0.0:
            losses["bc_loss"] = torch.tensor(0.0, device=device)

        # 6) Curriculum ramp
        cont_steps = self.cont_curr_steps if self.cont_curr_steps >= 0 else self.curr_steps
        mom_steps  = self.mom_curr_steps  if self.mom_curr_steps  >= 0 else self.curr_steps
        bc_steps   = self.bc_curr_steps   if self.bc_curr_steps   >= 0 else self.curr_steps
        cont_start = self.cont_ramp_start_step if self.cont_ramp_start_step >= 0 else self.ramp_start_step
        mom_start  = self.mom_ramp_start_step  if self.mom_ramp_start_step  >= 0 else self.ramp_start_step
        bc_start   = self.bc_ramp_start_step   if self.bc_ramp_start_step   >= 0 else self.ramp_start_step

        r_cont = self._ramp_factor(step, cont_steps, cont_start)
        r_mom  = self._ramp_factor(step, mom_steps,  mom_start)
        r_bc   = self._ramp_factor(step, bc_steps,   bc_start)
        bc_w   = self.bc_w * r_bc

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


# ============================================================
# UnifiedNavierStokesPhysicsLoss
# Surface/Volume separation + FV area-weighted operators
# + variable-coefficient diffusion + dynamic scaling
# ============================================================

class UnifiedNavierStokesPhysicsLoss(nn.Module):
    """
    Unified Navier-Stokes physics loss with surface/volume separation,
    improved FV numerical operators, dynamic weight scaling, and
    denormalization support.

    Features:
    - Surface/Volume separated data loss
    - Finite-volume area-weighted operators with automatic sanity checks
    - Variable-coefficient diffusion (nu_eff per edge)
    - Upwind convection
    - Integrated pressure force (FV-style)
    - sklearn scaler support (register_buffer for mean/std)
    - Boundary loss (wall no-slip + inlet velocity)
    - Auto pressure mode selection (grad vs FV)
    """

    def __init__(
        self,
        data_weight: float = 1.0,
        continuity_weight: float = 0.1,
        momentum_weight: float = 0.01,
        boundary_weight: float = 1.0,
        nu: float = 1.56e-5,
        rho: float = 1.225,
        use_area_weighting: bool = True,
        use_dynamic_scaling: bool = False,
        x_scaler: Optional[Any] = None,
        y_scaler: Optional[Any] = None,

        debug_print: bool = True,
        debug_every: int = 1,
        debug_max_steps: int = 50,
        pressure_is_p_over_rho: bool = True,
        U_ref: float = 1.0,
        L_ref: float = 1.0,
    ):
        super().__init__()
        self.data_weight = data_weight
        self.continuity_weight = continuity_weight
        self.momentum_weight = momentum_weight
        self.boundary_weight = boundary_weight
        self.nu = nu
        self.rho = rho
        self.use_area_weighting = use_area_weighting
        self.use_dynamic_scaling = use_dynamic_scaling
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

        # Register scaler buffers (sklearn: mean_/scale_, custom: mean/std)
        def _maybe_register_scaler(prefix, scaler):
            if scaler is None:
                return
            mean = getattr(scaler, "mean_", None)
            std  = getattr(scaler, "scale_", None)
            if mean is None: mean = getattr(scaler, "mean", None)
            if std  is None: std  = getattr(scaler, "std",  None)
            if (mean is not None) and (std is not None):
                self.register_buffer(f"_{prefix}_mean", torch.as_tensor(mean, dtype=torch.float32))
                self.register_buffer(f"_{prefix}_std",  torch.as_tensor(std,  dtype=torch.float32))
        _maybe_register_scaler("x", x_scaler)
        _maybe_register_scaler("y", y_scaler)

        self.debug_print = debug_print
        self.debug_every = int(debug_every)
        self.debug_max_steps = int(debug_max_steps)
        self.pressure_is_p_over_rho = bool(pressure_is_p_over_rho)
        self.U_ref = float(U_ref)
        self.L_ref = float(L_ref)

        # Pressure discretization mode: 'auto'|'fv'|'grad'
        if not hasattr(self, "pressure_mode"):
            self.pressure_mode = "auto"

        # Dynamic scaling parameters
        self.register_buffer('data_scale', torch.tensor(1.0))
        self.register_buffer('physics_scale', torch.tensor(1.0))

        # Debug step counter
        if not hasattr(self, '_dbg_step'):
            self.register_buffer('_dbg_step', torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        pred: torch.Tensor,
        batch: Any,
        model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Main loss computation.

        Args:
            pred: predictions [N, output_dim]
            batch: batch data (PyG Data/Batch)
            model: model (optional)

        Returns:
            Loss dictionary
        """
        device = pred.device

        # 1. Denormalize if needed
        if self.y_scaler is not None:
            pred_denorm = self._denormalize_y(pred)
            target_denorm = self._denormalize_y(batch.y)
        else:
            pred_denorm = pred
            target_denorm = batch.y

        # 2. Surface/Volume separation
        surface_mask, volume_mask = self._get_surface_volume_masks(batch, device)

        # 3. Data Loss
        data_losses = self._compute_data_loss(
            pred, batch.y, surface_mask, volume_mask
        )

        # 4. Physics Loss
        physics_losses = self._compute_physics_loss(
            pred_denorm, batch, device, target_denorm
        )

        # 5. Boundary Loss
        boundary_loss = self._compute_boundary_loss(
            pred_denorm, batch, device
        )

        # 6. Total Loss
        total_loss = (
            self.data_weight * data_losses['total'] +
            self.continuity_weight * physics_losses['continuity'] +
            self.momentum_weight * physics_losses['momentum'] +
            self.boundary_weight * boundary_loss
        )

        return {
            'total': total_loss,
            'data_surface': data_losses['surface'],
            'data_volume': data_losses['volume'],
            'data_total': data_losses['total'],
            'continuity': physics_losses['continuity'],
            'momentum': physics_losses['momentum'],
            'boundary': boundary_loss
        }

    def _get_surface_volume_masks(
        self,
        batch: Any,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Surface/Volume separation using wall mask or normal vectors."""
        # 1) Prefer explicit wall mask
        if hasattr(batch, 'is_wall') and isinstance(batch.is_wall, torch.Tensor):
            surface_mask = batch.is_wall.to(device=device, dtype=torch.bool).view(-1)
            volume_mask = ~surface_mask
            return surface_mask, volume_mask

        # 2) Fallback to normals from x
        if self.x_scaler is not None:
            x_denorm = self._denormalize_x(batch.x)
            normal_x = x_denorm[:, 3]
            normal_y = x_denorm[:, 4]
        else:
            normal_x = batch.x[:, 3]
            normal_y = batch.x[:, 4]

        normal_magnitude = torch.sqrt(normal_x**2 + normal_y**2)
        surface_mask = normal_magnitude > 1e-10
        volume_mask = ~surface_mask
        return surface_mask, volume_mask

    def _compute_data_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        surface_mask: torch.Tensor,
        volume_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Surface and volume separated data loss."""
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        C_pred = pred.size(1)
        if target.size(1) != C_pred:
            if target.size(1) > C_pred:
                target = target[:, :C_pred]
            else:
                pred = pred[:, : target.size(1)]
                C_pred = pred.size(1)

        if surface_mask.any():
            surface_loss = F.mse_loss(pred[surface_mask], target[surface_mask])
        else:
            surface_loss = torch.tensor(0.0, device=pred.device)

        if volume_mask.any():
            volume_loss = F.mse_loss(pred[volume_mask], target[volume_mask])
        else:
            volume_loss = torch.tensor(0.0, device=pred.device)

        total_data_loss = surface_loss + volume_loss

        return {
            'surface': surface_loss,
            'volume': volume_loss,
            'total': total_data_loss
        }

    def _compute_physics_loss(
        self,
        pred: torch.Tensor,
        batch: Any,
        device: torch.device,
        target_denorm: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Physics-based loss computation."""
        velocity = pred[:, :2]
        pressure = pred[:, 2:3]

        node_area = getattr(batch, 'node_area', None)

        # Pick physics-compatible edge attributes
        if hasattr(batch, 'edge_attr_phys') and isinstance(batch.edge_attr_phys, torch.Tensor):
            edge_attr_phys = batch.edge_attr_phys
        elif hasattr(batch, 'edge_attr_dxdy') and isinstance(batch.edge_attr_dxdy, torch.Tensor):
            edge_attr_phys = batch.edge_attr_dxdy[:, :3]
        elif hasattr(batch, 'edge_attr') and isinstance(batch.edge_attr, torch.Tensor):
            edge_attr_phys = batch.edge_attr[:, :3] if batch.edge_attr.size(1) >= 3 else batch.edge_attr
        else:
            raise ValueError("Physics edge attributes not found: need edge_attr_phys or edge_attr with >=3 columns")

        surface_mask, volume_mask = self._get_surface_volume_masks(batch, device)
        has_volume = bool(volume_mask is not None and volume_mask.any())

        # Area ops sanity check
        area_ops, area_r = self._area_ops_enabled(batch.edge_index, edge_attr_phys, batch.num_nodes, node_area)

        # Continuity loss
        divergence = self._compute_divergence(
            velocity, batch.edge_index, edge_attr_phys, batch.num_nodes, node_area, area_ops=area_ops
        )

        U_ref = self._infer_U_ref(batch)
        self.U_ref = U_ref

        # Non-dimensionalize continuity
        if area_ops:
            ell_c = torch.sqrt(node_area.clamp_min(1e-12))
        else:
            ell_c = self._node_length_scale(batch.edge_index, edge_attr_phys, batch.num_nodes)

        cont_nd = divergence * (ell_c / U_ref)

        if has_volume:
            continuity_loss = torch.mean(cont_nd[volume_mask]**2)
        else:
            continuity_loss = torch.mean(cont_nd**2)

        # Effective viscosity
        nu_t = None
        if target_denorm is not None and target_denorm.shape[1] >= 4:
            nu_t = target_denorm[:, 3:4]
        elif hasattr(batch, 'y') and isinstance(batch.y, torch.Tensor) and batch.y.shape[1] >= 4 and self.y_scaler is None:
            nu_t = batch.y[:, 3:4]
        if nu_t is None:
            nu_t = torch.zeros_like(pressure)
        if nu_t.dim() == 1:
            nu_t = nu_t.view(-1, 1)
        nu_eff = (float(self.nu) if self.nu is not None else 0.0) + torch.relu(nu_t)

        # Momentum loss
        momentum_residual = self._compute_momentum_residual(
            velocity, pressure, batch.edge_index,
            edge_attr=edge_attr_phys,
            num_nodes=batch.num_nodes,
            node_area=node_area,
            nu_eff=nu_eff
        )

        # Non-dimensionalize momentum
        ell = self._node_length_scale(batch.edge_index, edge_attr_phys, batch.num_nodes)
        ref_accel = (U_ref * U_ref) / ell
        momentum_residual = momentum_residual / ref_accel.unsqueeze(-1)

        if has_volume:
            momentum_loss = torch.mean(torch.sum(momentum_residual[volume_mask]**2, dim=1))
        else:
            momentum_loss = torch.mean(torch.sum(momentum_residual**2, dim=1))

        # Debug dump
        if (self._dbg_step.item() < self.debug_max_steps) and (self._dbg_step.item() % self.debug_every == 0):
            def _area_ok():
                if (not self.use_area_weighting) or (node_area is None):
                    return False
                ell_loc = self._node_length_scale(batch.edge_index, edge_attr_phys, batch.num_nodes)
                r = torch.median(node_area.clamp_min(1e-12)) / torch.median((ell_loc * ell_loc).clamp_min(1e-12))
                return (r > 0.2) and (r < 5.0)
            use_fv_dbg = (self.pressure_mode == "fv") or (
                self.pressure_mode == "auto" and _area_ok()
            )
            self._debug_dump_momentum_terms(
                velocity, pressure,
                batch.edge_index, edge_attr_phys, batch.num_nodes,
                nu_eff=nu_eff, node_area=node_area,
                use_fv_pressure=use_fv_dbg,
                volume_mask=volume_mask,
                area_ops=area_ops,
                area_r=area_r
            )
        self._dbg_step += 1

        return {
            'continuity': continuity_loss,
            'momentum': momentum_loss
        }

    def _area_ops_enabled(self, edge_index, edge_attr, num_nodes, node_area) -> Tuple[bool, torch.Tensor]:
        """Check if FV area-weighted ops should be used. Returns (enabled, r=median(area)/median(ell^2))."""
        if (not self.use_area_weighting) or (node_area is None):
            return False, torch.tensor(float('nan'), device=edge_attr.device)
        ell = self._node_length_scale(edge_index, edge_attr, num_nodes)
        r = torch.median(node_area.clamp_min(1e-12)) / torch.median((ell * ell).clamp_min(1e-12))
        ok = (r > 0.2) and (r < 5.0)
        return bool(ok), r

    def _compute_divergence(
        self,
        velocity: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
        node_area: Optional[torch.Tensor] = None,
        area_ops: bool = False
    ) -> torch.Tensor:
        """Conservative divergence with optional area weighting."""
        device = velocity.device
        eps = 1e-10

        dx = edge_attr[:, 0]
        dy = edge_attr[:, 1]
        dist = self._safe_dist(edge_attr) + eps

        i, j = edge_index
        u_edge = 0.5 * (velocity[i, 0] + velocity[j, 0])
        v_edge = 0.5 * (velocity[i, 1] + velocity[j, 1])

        if area_ops:
            flux = (u_edge * dy - v_edge * dx)
        else:
            flux = (u_edge * dy - v_edge * dx) / dist

        div = scatter_add(flux, i, dim=0, dim_size=num_nodes)

        degree = scatter_add(
            torch.ones_like(flux), i, dim=0, dim_size=num_nodes
        ).clamp(min=1.0)

        if area_ops:
            area_safe = node_area.clamp(min=eps)
            div = div / area_safe
        else:
            div = div / degree

        return div

    def _compute_momentum_residual(
        self,
        velocity: torch.Tensor,
        pressure: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
        node_area: Optional[torch.Tensor] = None,
        area_ops: bool = False,
        nu_eff: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Momentum equation residual."""
        device = velocity.device

        # 1. Convection
        convection = self._compute_convection(
            velocity, edge_index, edge_attr, num_nodes, node_area, area_ops=area_ops
        )

        # 2. Pressure term selection (auto: area check + ND sanity)
        def _area_looks_ok() -> bool:
            if (not self.use_area_weighting) or (node_area is None):
                return False
            ell_loc = self._node_length_scale(edge_index, edge_attr, num_nodes)
            r = torch.median(node_area.clamp_min(1e-12)) / torch.median((ell_loc * ell_loc).clamp_min(1e-12))
            return (r > 0.2) and (r < 5.0)

        p_grad = self._compute_gradient(pressure, edge_index, edge_attr, num_nodes)
        if not self.pressure_is_p_over_rho:
            p_grad = p_grad / self.rho
        p_fv = None
        if self.use_area_weighting and (node_area is not None):
            p_fv = self._compute_pressure_force(pressure, edge_index, edge_attr, num_nodes, node_area)
            if not self.pressure_is_p_over_rho:
                p_fv = p_fv / self.rho

        if self.pressure_mode == "grad" or (self.pressure_mode == "auto" and not _area_looks_ok()):
            pressure_term = p_grad
        elif self.pressure_mode == "fv":
            pressure_term = p_fv
        else:
            ell_loc = self._node_length_scale(edge_index, edge_attr, num_nodes)
            ref_acc = (self.U_ref * self.U_ref) / ell_loc.clamp_min(1e-8)
            rms_grad = (p_grad / ref_acc.unsqueeze(-1)).norm(dim=1).mean()
            rms_fv   = (p_fv   / ref_acc.unsqueeze(-1)).norm(dim=1).mean()
            pressure_term = p_fv if (rms_fv >= 0.1 * rms_grad) else p_grad

        # 3. Viscous term: variable-coefficient diffusion
        if nu_eff is not None:
            viscous = self._compute_variable_diffusion(
                velocity, edge_index, edge_attr, num_nodes, nu_eff, node_area=node_area, area_ops=area_ops
            )
        else:
            viscous = self.nu * self._compute_laplacian(
                velocity, edge_index, edge_attr, num_nodes, node_area, area_ops=area_ops
            )

        residual = convection + pressure_term - viscous
        return residual

    def _compute_pressure_force(
        self,
        pressure: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
        node_area: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Integrated pressure force: (sum_e p_e * n_e * |e|) / area_i (FV-style)."""
        eps = 1e-10
        i, j = edge_index
        dx = edge_attr[:, 0]
        dy = edge_attr[:, 1]
        dist = self._safe_dist(edge_attr) + eps

        nx =  dy / dist
        ny = -dx / dist
        p_e = 0.5 * (pressure[i, 0] + pressure[j, 0])
        fx_e = p_e * nx * dist
        fy_e = p_e * ny * dist
        fx = scatter_add(fx_e, i, dim=0, dim_size=num_nodes)
        fy = scatter_add(fy_e, i, dim=0, dim_size=num_nodes)
        F_vec = torch.stack([fx, fy], dim=1)
        if self.use_area_weighting and (node_area is not None):
            area = node_area.clamp_min(1e-12).unsqueeze(-1)
            F_vec = F_vec / area
        else:
            deg = scatter_add(torch.ones_like(fx_e), i, dim=0, dim_size=num_nodes).clamp_min(1.0).unsqueeze(-1)
            F_vec = F_vec / deg
        return F_vec

    def _compute_variable_diffusion(
        self,
        field: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
        nu_eff: Optional[torch.Tensor],
        node_area: Optional[torch.Tensor] = None,
        area_ops: bool = False,
        eps: float = 1e-10
    ) -> torch.Tensor:
        """Compute div(nu_eff * grad(field)) using edge-averaged viscosity."""
        device = field.device
        i, j = edge_index
        dist = self._safe_dist(edge_attr) + eps
        inv_r  = 1.0 / dist
        inv_r2 = inv_r * inv_r

        if nu_eff is None:
            nu_e = torch.zeros(edge_index.shape[1], 1, device=device, dtype=field.dtype)
        else:
            if nu_eff.dim() == 1:
                nu_e = 0.5 * (nu_eff[i].unsqueeze(-1) + nu_eff[j].unsqueeze(-1))
            else:
                nu_e = 0.5 * (nu_eff[i] + nu_eff[j])
        df = (field[j] - field[i])
        w = inv_r if area_ops else inv_r2
        contrib = (nu_e * w.unsqueeze(-1)) * df

        acc = scatter_add(contrib, i, dim=0, dim_size=num_nodes)

        if area_ops:
            area = node_area.clamp_min(1e-12).unsqueeze(-1)
            acc = acc / area
        else:
            degree = scatter_add(
                torch.ones((edge_index.shape[1],), device=device),
                i, dim=0, dim_size=num_nodes
            ).clamp(min=1.0).unsqueeze(-1)
            acc = acc / degree

        return acc

    def _compute_boundary_loss(
        self,
        pred: torch.Tensor,
        batch: Any,
        device: torch.device,
        use_normalized: bool = True
    ) -> torch.Tensor:
        """Boundary condition loss (wall no-slip + inlet velocity)."""
        boundary_loss = torch.tensor(0.0, device=device)
        debug_boundary = bool(getattr(batch, 'debug_boundary_scale', False))
        if not hasattr(self, '_dbg_bc_count'):
            self._dbg_bc_count = 0

        # Wall boundary (no-slip)
        wall_mask = None
        if hasattr(batch, 'is_wall') and isinstance(batch.is_wall, torch.Tensor):
            wall_mask = batch.is_wall.to(device=device, dtype=torch.bool).view(-1)
        elif hasattr(batch, 'wall_mask') and isinstance(batch.wall_mask, torch.Tensor):
            wall_mask = batch.wall_mask.to(device=device, dtype=torch.bool).view(-1)
        if wall_mask is not None and wall_mask.any():
            wall_velocity = pred[wall_mask, :2]
            if debug_boundary or self._dbg_bc_count < 3:
                try:
                    wall_speed = torch.linalg.vector_norm(wall_velocity, dim=1)
                    print(f"[BoundaryScale] wall: N={int(wall_speed.numel())} | |u| mean={float(wall_speed.mean().item()):.3f} max={float(wall_speed.max().item()):.3f}")
                except Exception:
                    pass
            U = self.U_ref if hasattr(self, "U_ref") else torch.tensor(1.0, device=device)
            wall_nd2 = (torch.linalg.vector_norm(wall_velocity, dim=1) / U.clamp_min(1e-8))**2
            boundary_loss = boundary_loss + torch.mean(wall_nd2)

        # Inlet boundary
        inlet_mask = None
        inlet_target = None
        if hasattr(batch, 'is_inlet') and isinstance(batch.is_inlet, torch.Tensor):
            inlet_mask = batch.is_inlet.to(device=device, dtype=torch.bool).view(-1)
            if hasattr(batch, 'inlet_u') and isinstance(batch.inlet_u, torch.Tensor):
                inlet_target = batch.inlet_u.to(device=device, dtype=pred.dtype)
        elif hasattr(batch, 'inlet_mask') and hasattr(batch, 'inlet_velocity'):
            inlet_mask = batch.inlet_mask.to(device=device, dtype=torch.bool).view(-1)
            inlet_target = batch.inlet_velocity.to(device=device, dtype=pred.dtype)
        if inlet_mask is not None and inlet_target is not None and inlet_mask.any():
            inlet_velocity = pred[inlet_mask, :2]
            inlet_target_masked = inlet_target[inlet_mask]
            if debug_boundary or self._dbg_bc_count < 3:
                try:
                    sp = torch.linalg.vector_norm(inlet_velocity, dim=1)
                    st = torch.linalg.vector_norm(inlet_target_masked, dim=1)
                    mse = F.mse_loss(inlet_velocity, inlet_target_masked)
                    print(
                        f"[BoundaryScale] inlet: N={int(sp.numel())} | |u_pred| mean={float(sp.mean().item()):.3f} | |u_gt| mean={float(st.mean().item()):.3f} | MSE={float(mse.item()):.3f}"
                    )
                except Exception:
                    pass
            U = self.U_ref if hasattr(self, "U_ref") else torch.tensor(1.0, device=device)
            bc_diff_nd = (inlet_velocity - inlet_target_masked) / U.clamp_min(1e-8)
            boundary_loss = boundary_loss + torch.mean(torch.sum(bc_diff_nd**2, dim=1))

        if not debug_boundary:
            self._dbg_bc_count = min(self._dbg_bc_count + 1, 10)

        return boundary_loss

    def _compute_convection(
        self,
        velocity: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
        node_area: Optional[torch.Tensor] = None,
        area_ops: bool = False
    ) -> torch.Tensor:
        """Convection term with FV or lightweight approximation."""
        device = velocity.device
        eps = 1e-10

        i, j = edge_index
        dx = edge_attr[:, 0]
        dy = edge_attr[:, 1]
        dist = self._safe_dist(edge_attr) + eps

        if area_ops:
            nx =  dy / dist
            ny = -dx / dist
            vn_i = velocity[i, 0] * nx + velocity[i, 1] * ny
            take_i = (vn_i > 0).unsqueeze(-1)
            v_up = torch.where(take_i, velocity[i], velocity[j])
            vn_up = v_up[:, 0] * nx + v_up[:, 1] * ny
            Fx_e = vn_up * v_up[:, 0] * dist
            Fy_e = vn_up * v_up[:, 1] * dist
            cx = scatter_add(Fx_e, i, dim=0, dim_size=num_nodes)
            cy = scatter_add(Fy_e, i, dim=0, dim_size=num_nodes)
            area = node_area.clamp_min(1e-12)
            cx = cx / area
            cy = cy / area
            return torch.stack([cx, cy], dim=1)
        else:
            tnx = dx / dist
            tny = dy / dist
            vn_i = velocity[i, 0] * tnx + velocity[i, 1] * tny
            v_up = torch.where(vn_i.unsqueeze(-1) > 0, velocity[i], velocity[j])
            dv = velocity[j] - velocity[i]
            conv_x = v_up[:, 0] * dv[:, 0] / dist
            conv_y = v_up[:, 1] * dv[:, 1] / dist
            cx = scatter_add(conv_x, i, dim=0, dim_size=num_nodes)
            cy = scatter_add(conv_y, i, dim=0, dim_size=num_nodes)
            deg = scatter_add(torch.ones_like(conv_x), i, dim=0, dim_size=num_nodes).clamp_min(1.0)
            cx = cx / deg
            cy = cy / deg
            return torch.stack([cx, cy], dim=1)

    def _compute_gradient(
        self,
        scalar: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Gradient computation for scalar field."""
        eps = 1e-10

        i, j = edge_index
        dx = edge_attr[:, 0]
        dy = edge_attr[:, 1]
        dist = self._safe_dist(edge_attr) + eps

        ds = scalar[j] - scalar[i]

        grad_x = ds.squeeze() * dx / (dist**2)
        grad_y = ds.squeeze() * dy / (dist**2)

        gradient_x = scatter_add(grad_x, i, dim=0, dim_size=num_nodes)
        gradient_y = scatter_add(grad_y, i, dim=0, dim_size=num_nodes)

        degree = scatter_add(
            torch.ones_like(grad_x), i, dim=0, dim_size=num_nodes
        ).clamp(min=1.0)

        gradient_x = gradient_x / degree
        gradient_y = gradient_y / degree

        return torch.stack([gradient_x, gradient_y], dim=1)

    def _compute_laplacian(
        self,
        field: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
        node_area: Optional[torch.Tensor] = None,
        area_ops: bool = False
    ) -> torch.Tensor:
        """Laplacian computation for vector field."""
        device = field.device
        eps = 1e-10

        i, j = edge_index
        dist = self._safe_dist(edge_attr) + eps

        df = field[j] - field[i]

        if area_ops:
            laplacian_contrib = 2 * df / dist.unsqueeze(-1)
        else:
            laplacian_contrib = 2 * df / (dist**2).unsqueeze(-1)

        laplacian = scatter_add(
             laplacian_contrib, i, dim=0, dim_size=num_nodes
         )

        if area_ops:
            area = node_area.clamp_min(1e-12).unsqueeze(-1)
            laplacian = laplacian / area
        else:
            degree = scatter_add(
                torch.ones((edge_index.shape[1],), device=device),
                i, dim=0, dim_size=num_nodes
            ).clamp(min=1.0).unsqueeze(-1)
            laplacian = laplacian / degree

        return laplacian

    def _denormalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """Input data denormalization (autograd-friendly)."""
        if hasattr(self, "_x_mean") and hasattr(self, "_x_std"):
            mean = self._x_mean.to(x.device); std = self._x_std.to(x.device)
            return x * std + mean
        return x

    def _denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        """Output data denormalization (autograd-friendly)."""
        if hasattr(self, "_y_mean") and hasattr(self, "_y_std"):
            mean = self._y_mean.to(y.device); std = self._y_std.to(y.device)
            return y * std + mean
        return y

    def _infer_U_ref(self, batch) -> torch.Tensor:
        """Dynamically infer reference velocity U_ref from batch data."""
        if hasattr(self, "_x_mean") and hasattr(self, "_x_std"):
            x_denorm = self._denormalize_x(batch.x)
        elif hasattr(batch, "x_norm_params") and isinstance(batch.x_norm_params, dict):
            xm = batch.x_norm_params.get("mean", None)
            xs = batch.x_norm_params.get("scale", None)
            if xm is not None and xs is not None:
                mean = xm.to(batch.x.device)
                std  = xs.to(batch.x.device)
                x_denorm = batch.x * std + mean
            else:
                x_denorm = batch.x
        else:
            x_denorm = batch.x

        u_all = x_denorm[:, 0]
        v_all = x_denorm[:, 1]

        mask = None
        if hasattr(batch, 'is_farfield') and isinstance(batch.is_farfield, torch.Tensor) and batch.is_farfield.any():
            mask = batch.is_farfield
        elif hasattr(batch, 'is_inlet') and isinstance(batch.is_inlet, torch.Tensor) and batch.is_inlet.any():
            mask = batch.is_inlet

        if mask is not None:
            u = u_all[mask]
            v = v_all[mask]
        else:
            u = u_all
            v = v_all

        U_ref = torch.sqrt((u**2 + v**2).mean()).clamp_min(1e-6)
        return U_ref

    @torch.no_grad()
    def _debug_dump_momentum_terms(self, velocity, pressure, edge_index, edge_attr, num_nodes,
                                   nu_eff=None, node_area=None, use_fv_pressure=True, volume_mask=None,
                                   area_ops: bool = False, area_r: Optional[torch.Tensor] = None):
        """Debug helper to dump momentum term statistics."""
        dist = edge_attr[:, 2].clamp_min(1e-12)
        q = torch.quantile(dist, torch.tensor([0.0, 0.01, 0.10, 0.50], device=dist.device))
        dmin, d01, d10, d50 = [float(x) for x in q]

        conv = self._compute_convection(velocity, edge_index, edge_attr, num_nodes, node_area, area_ops=area_ops)
        if use_fv_pressure:
            pterm = self._compute_pressure_force(pressure, edge_index, edge_attr, num_nodes, node_area)
            if not self.pressure_is_p_over_rho:
                pterm = pterm / self.rho
        else:
            pterm = self._compute_gradient(pressure, edge_index, edge_attr, num_nodes)
            if not self.pressure_is_p_over_rho:
                pterm = pterm / self.rho
        diff = ( self._compute_variable_diffusion(velocity, edge_index, edge_attr, num_nodes, nu_eff, node_area=node_area, area_ops=area_ops)
                 if (nu_eff is not None) else
                 self.nu * self._compute_laplacian(velocity, edge_index, edge_attr, num_nodes, node_area, area_ops=area_ops) )

        ell = self._node_length_scale(edge_index, edge_attr, num_nodes)
        ref_accel = (self.U_ref * self.U_ref) / ell.clamp_min(1e-8)
        conv_nd  = conv / ref_accel.unsqueeze(-1)
        pterm_nd = pterm / ref_accel.unsqueeze(-1)
        diff_nd  = diff / ref_accel.unsqueeze(-1)
        resid_nd = conv_nd + pterm_nd - diff_nd
        mask = volume_mask if (volume_mask is not None and volume_mask.any()) else slice(None)
        conv_rms = float(conv_nd[mask].norm(dim=1).mean())
        p_rms    = float(pterm_nd[mask].norm(dim=1).mean())
        d_rms    = float(diff_nd[mask].norm(dim=1).mean())
        r_rms    = float(resid_nd[mask].norm(dim=1).mean())

        print(
            f"[phys][step {int(self._dbg_step.item())}] "
            f"dx min/1%/10%/50% = {dmin:.3e}/{d01:.3e}/{d10:.3e}/{d50:.3e} | "
            f"ND |conv|={conv_rms:.3e} |p-term|={p_rms:.3e} |diff|={d_rms:.3e} | resid={r_rms:.3e} | "
            f"area_ops={'on' if area_ops else 'off'}"
            + (f" r={float(area_r):.2f}" if area_r is not None and torch.isfinite(area_r) else "")
        )

    def _safe_dist(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Floor small distances to prevent 1/dx explosions."""
        dist = edge_attr[:, 2]
        med = torch.median(dist)
        floor = torch.clamp(med * 0.10, min=1e-8)
        return torch.clamp(dist, min=floor)

    def _node_length_scale(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Per-node local length scale: mean incident edge length."""
        i, _ = edge_index
        dist = self._safe_dist(edge_attr)
        sum_dist = scatter_add(dist, i, dim=0, dim_size=num_nodes)
        deg = scatter_add(torch.ones_like(dist), i, dim=0, dim_size=num_nodes).clamp_min(1.0)
        ell = (sum_dist / deg).clamp_min(1e-6)
        return ell


if __name__ == "__main__":
    print("Navier-Stokes physics loss (dynamic U_inf/Re) ready.")
    print("UnifiedNavierStokesPhysicsLoss also available.")
