# 2a) Utilities: pos2 append, surface mask, and edge features
import torch
from torch_geometric.data import Data



def with_pos2(data):
    # Append 2D position (pos[:,:2]->pos2) into node features x to make 7D (orig 5 + pos2)
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
    if hasattr(d, 'surf') and isinstance(d.surf, torch.Tensor) and d.surf.dtype == torch.bool:
        return d.surf.view(-1)
    x = d.x
    if x is not None and x.size(1) >= 5:
        wall = x[:, 2]
        nxy = x[:, 3:5]
        return (wall < 1e-6) | (nxy.abs().sum(dim=1) > 0)
    elif x is not None and x.size(1) >= 3:
        wall = x[:, 2]
        return (wall < 1e-6)
    else:
        return torch.zeros(d.x.size(0), dtype=torch.bool, device=d.x.device)

def ensure_edge_features(d, want_dim: int = 5):
    if hasattr(d, 'edge_attr') and d.edge_attr is not None and d.edge_attr.size(1) == want_dim:
        return d
    if not hasattr(d, 'edge_index') or d.edge_index is None or d.edge_index.numel() == 0:
        return d
    row, col = d.edge_index
    # Dist, dir_xy, cos_n (using pos if available else first 3 dims of x), surf_pair
    if hasattr(d, 'pos') and d.pos is not None:
        pos = d.pos
    else:
        pos = d.x[:, :3]
    dvec = pos[col, :2] - pos[row, :2]
    dist = dvec.norm(dim=1, keepdim=True)
    dir_xy = dvec / (dist + 1e-8)
    nxy = torch.zeros(d.x.size(0), 2, device=d.x.device, dtype=d.x.dtype)
    if d.x.size(1) >= 5:
        nxy = d.x[:, 3:5]
    cos_n = ((dir_xy * (nxy[row] + nxy[col]) * 0.5).sum(dim=1, keepdim=True))
    surf = get_surface_mask(d).to(torch.float32)
    surf_pair = torch.stack([(surf[row] > 0.5) & (surf[col] > 0.5), (surf[row] > 0.5) ^ (surf[col] > 0.5)], dim=1).to(dist.dtype)
    edge_attr = torch.cat([dist, dir_xy, cos_n, surf_pair], dim=1)
    d.edge_attr = edge_attr
    return d

def prep_graph(g):
    gg = g
    if gg.x is None:
        raise ValueError('Graph missing node features x.')
    # Keep node features as original 5D here; do NOT append pos into x at this stage.
    # Only ensure edge features exist; they will be computed from d.pos (preferred) or x[:,:3].
    gg = ensure_edge_features(gg, want_dim=5)
    return gg

def validate_edges(ds, name='train'):
    n = len(ds)
    bad=0
    for i in range(n):
        d = ds[i]
        ei = getattr(d,'edge_index',None)
        ea = getattr(d,'edge_attr',None)
        N = d.x.size(0)
        if ei is None or ea is None:
            bad+=1; print(f'[{name}] {i}: missing edges')
            continue
        if ei.dtype!=torch.long or ei.size(0)!=2 or ei.size(1)==0:
            bad+=1; print(f'[{name}] {i}: bad edge_index shape {tuple(ei.shape)} / dtype {ei.dtype}')
        if int(ei.min())<0 or int(ei.max())>=N:
            bad+=1; print(f'[{name}] {i}: edge_index out of range [0,{N-1}] -> ({int(ei.min())},{int(ei.max())})')
        if ea.dim()!=2 or ea.size(0)!=ei.size(1):
            bad+=1; print(f'[{name}] {i}: edge_attr shape mismatch {tuple(ea.shape)} vs E={ei.size(1)}')
    print(f'[validate] {name}: total={n} bad={bad}')

def _prep_graph_for_norm(g):
    d = g.clone()
    # Ensure pos2 appended (to 7D) only if original x is 5D and we haven't already appended pos2
    try:
        if d.x is not None and d.x.size(1) == 5 and not getattr(d, 'pos2_appended', False):
            d = with_pos2(d)
    except Exception:
        # Fallback: keep as-is if with_pos2 not applicable
        pass
    # Ensure edge features exist
    d = ensure_edge_features(d, want_dim=5)
    return d


def _to_xy(t: torch.Tensor):
    return t[:, :2].detach().cpu().float().numpy()

def _poly_from_surface(data: Data):
    try:
        mask = get_surface_mask(data)
        pts = (data.pos if hasattr(data,'pos') and data.pos is not None else data.x)[:, :2]
        pts = pts[mask].detach().cpu().numpy()
        if pts.shape[0] < 3: return None
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
        order = np.argsort(ang)
        return Path(pts[order], closed=True)
    except Exception:
        return None