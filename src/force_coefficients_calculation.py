import torch
from torch_geometric.data import Data




# 5. (선택) 그래프 수준 Lift Coefficient 계산 함수 정의
#   - 필요시 dataset iteration 후 data.y_graph 추가

def order_surface(pos_surf: torch.Tensor):
    c = pos_surf.mean(dim=0)
    rel = pos_surf - c
    angles = torch.atan2(rel[:,1], rel[:,0])
    return torch.argsort(angles)

def panel_lengths(pos_ordered: torch.Tensor):
    rolled = torch.roll(pos_ordered, -1, 0)
    seg = rolled - pos_ordered
    ds = seg.norm(dim=1)
    return ds, seg

def compute_lift_coefficient(data: Data, eps=1e-9):
    if not hasattr(data, 'surf'):
        return None
    xvars = data.x
    yvars = data.y
    pos = data.pos if hasattr(data,'pos') and data.pos is not None else None
    if pos is None:
        # fallback: construct pseudo-pos from first 2 normal dims (NOT ideal)
        N = xvars.size(0)
        pos = torch.zeros(N,2)
        pos[:,0] = torch.linspace(0,1,N)
        pos[:,1] = 0
    wall_dist = xvars[:,2]
    normals = xvars[:,3:5]
    surf_mask = (wall_dist < 1e-6) | (normals.abs().sum(dim=1)>0)
    surf_idx = torch.nonzero(surf_mask, as_tuple=False).squeeze(-1)
    if surf_idx.numel() < 5:
        return None
    pos_surf = pos[surf_idx]
    ord_idx = order_surface(pos_surf)
    ordered_pos = pos_surf[ord_idx]
    ds, seg_vec = panel_lengths(ordered_pos)
    n_nodes = normals[surf_idx][ord_idx]
    n_norm = n_nodes.norm(dim=1, keepdim=True).clamp_min(eps)
    n_hat = n_nodes / n_norm
    u_inf = xvars[0,0]; v_inf = xvars[0,1]
    U_inf = float(torch.sqrt(u_inf**2 + v_inf**2)) + eps
    p_over_rho = yvars[surf_idx,2][ord_idx]
    Fp = (-p_over_rho.unsqueeze(1)*n_hat) * ds.unsqueeze(1)
    F = Fp.sum(dim=0)
    chord = float(pos_surf[:,0].max() - pos_surf[:,0].min() + eps)
    q_ref = 0.5 * U_inf**2
    Cl = F[1]/(q_ref * chord + eps)
    return float(Cl)

def compute_force_coefficients(data: Data, eps=1e-9):
    """
    Compute drag coefficient (CD) and lift coefficient (CL) from surface pressure.
    
    Args:
        data: PyG Data object with x (features), y (targets), pos (positions)
        eps: small value to avoid division by zero
    
    Returns:
        dict with 'CD' and 'CL' if successful, None otherwise
    """
    xvars = torch.tensor(data.x) if not isinstance(data.x, torch.Tensor) else data.x.cpu()
    yvars = torch.tensor(data.y) if not isinstance(data.y, torch.Tensor) else data.y.cpu()
    pos = torch.tensor(data.pos) if data.pos is not None and not isinstance(data.pos, torch.Tensor) else (data.pos.cpu() if data.pos is not None else data.pos)
    if pos is None:
        # fallback: construct pseudo-pos from first 2 normal dims (NOT ideal)
        N = xvars.size(0)
        pos = torch.zeros(N,2)
        pos[:,0] = torch.linspace(0,1,N)
        pos[:,1] = 0
    
    # Surface mask: feature columns 4-5 (normals) are not zero
    normals = xvars[:,3:5]
    surf_mask = (normals.abs().sum(dim=1) > 0)
    surf_idx = torch.nonzero(surf_mask, as_tuple=False).squeeze(-1).long()
    if surf_idx.numel() < 5:
        return None
    
    pos_surf = pos[surf_idx]
    ord_idx = order_surface(pos_surf).long()
    ordered_pos = pos_surf[ord_idx]
    ds, seg_vec = panel_lengths(ordered_pos)
    n_nodes = normals[surf_idx[ord_idx]]
    n_norm = n_nodes.norm(dim=1, keepdim=True).clamp_min(eps)
    n_hat = n_nodes / n_norm
    
    # Freestream velocity: L2 norm of first two columns of features
    u_inf = xvars[0,0]; v_inf = xvars[0,1]
    U_inf = float(torch.sqrt(u_inf**2 + v_inf**2)) + eps
    
    temp = torch.index_select(yvars, 0, surf_idx)[:,2]
    p_over_rho = temp[ord_idx]
    Fp = (-p_over_rho.unsqueeze(1) * n_hat) * ds.unsqueeze(1)
    F = Fp.sum(dim=0)  # Total force [Fx, Fy]
    
    # Chord length
    chord = float(pos_surf[:,0].max() - pos_surf[:,0].min() + eps)
    
    # Dynamic pressure
    q_ref = 0.5 * U_inf**2
    
    # Coefficients
    CD = F[0] / (q_ref * chord + eps)
    CL = F[1] / (q_ref * chord + eps)
    
    return {'CD': float(CD), 'CL': float(CL)}