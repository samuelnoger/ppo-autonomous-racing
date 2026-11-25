import torch
import math
import numpy as np

def save_checkpoint(
    epoch,
    model,
    optimizer,
    checkpoint_path="checkpoints/last_ckpt.pth",
    total_epochs=None,
    scheduler=None,
    best_reward=-float("inf"),
    track_name=None,
):
    """Save model and optimizer state for checkpoint resuming.

    Saves model weights, optimizer state, learning rate scheduler state,
    and training metadata needed to resume training from this epoch.

    Args:
        epoch: Current epoch number.
        model: Model with state to save.
        optimizer: Optimizer with state to save.
        checkpoint_path: Where to save the checkpoint.
        total_epochs: Total number of epochs planned for training.
        scheduler: Optional LR scheduler with state to save.
        best_reward: Best reward achieved so far.
    """
    # Ensure destination directory exists
    import os
    dirpath = os.path.dirname(checkpoint_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "track_name": track_name,
            "total_epochs": total_epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "current_lr": optimizer.param_groups[0]["lr"],
            "best_reward": best_reward,
        },
        checkpoint_path,
    )
            
def passed_gate(car, gate_line, threshold=15):
    x1, y1, x2, y2 = gate_line
    # distance from point to line segment formula
    px, py = car.x, car.y
    line_mag = math.hypot(x2 - x1, y2 - y1)
    if line_mag < 1e-6:
        return False
    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag**2)
    u = max(0, min(1, u))
    ix = x1 + u * (x2 - x1)
    iy = y1 + u * (y2 - y1)
    dist = math.hypot(px - ix, py - iy)
    return dist < threshold

def generate_triangle_track(screen_width=1000, screen_height=600,width=600, height=400, corner_points=5, corner_radius=60):

    """
    Generates a triangular track with rounded corners.

    Returns:
        centerline: list of (x, y) points
    """
    cx = screen_width / 2
    cy = screen_height / 2

    hw = width / 2
    hh = height / 2

    centerline = []

    # corner radius
    r = corner_radius

    # triangle vertices
    vertices = [
        (cx, cy - hh + r,-3*np.pi/4, -np.pi/4),          # top vertex
        (cx + hw - r, cy + hh - r,-np.pi/4,np.pi/2),     # bottom-right vertex
        (cx - hw + r, cy + hh - r,np.pi/2,5*np.pi/4)     # bottom-left vertex
    ]

    for cx_c, cy_c, a0, a1 in vertices:
        angles = np.linspace(a0, a1, corner_points)
        for a in angles:
            x = cx_c + r * np.cos(a)
            y = cy_c + r * np.sin(a)
            centerline.append((x, y))

    centerline.append(centerline[0])

    return centerline

def generate_simple_track(screen_width=1000, screen_height=600,
                          width=600, height=400,
                          corner_points=5, corner_radius=140):
    """
    Generates a rectangular track with rounded corners.

    Returns:
        centerline: list of (x, y) points
    """

    cx = screen_width / 2
    cy = screen_height / 2

    hw = width / 2
    hh = height / 2

    centerline = []

    # corner radius
    r = corner_radius

    # corner centers
    corners = [
        (cx + hw - r, cy - hh + r, -np.pi/2, 0),        # top-right
        (cx + hw - r, cy + hh - r, 0, np.pi/2),         # bottom-right
        (cx - hw + r, cy + hh - r, np.pi/2, np.pi),     # bottom-left
        (cx - hw + r, cy - hh + r, np.pi, 3*np.pi/2)    # top-left
    ]

    for cx_c, cy_c, a0, a1 in corners:
        angles = np.linspace(a0, a1, corner_points)
        for a in angles:
            x = cx_c + r * np.cos(a)
            y = cy_c + r * np.sin(a)
            centerline.append((x, y))
            
    centerline.append(centerline[0])

    return centerline

def get_inputs(cars, track, gates_tensor, gate_indices, max_ray_dist):
    ray_angles = cars.ray_angles
    ray_dists_original = track.get_lines_along_rays(cars.pos, ray_angles, max_ray_dist)
    ray_dists = ray_dists_original / max_ray_dist

    speed_input = cars.speed.unsqueeze(1) / cars.max_speed

    car_heading_x = torch.cos(cars.angle)
    car_heading_y = torch.sin(cars.angle)

    curvatures = []
    distances = []
    n_gates = gates_tensor.shape[0]
    for lookahead in [2, 5, 9]:
        next_idx = (gate_indices + lookahead * cars.direction) % n_gates
        next_gate = gates_tensor[next_idx]

        next_center = (next_gate[:, 0:2] + next_gate[:, 2:4]) / 2.0
        next_dist = (next_center - cars.pos).norm(dim=1)
        distances.append(torch.clamp(next_dist / max_ray_dist, max=1.0).unsqueeze(1))

        next_gx = next_gate[:, 2] - next_gate[:, 0]
        next_gy = next_gate[:, 3] - next_gate[:, 1]
        next_norm = torch.hypot(next_gx, next_gy) + 1e-6
        next_dir_x = (-next_gy / next_norm) * cars.direction
        next_dir_y = (next_gx / next_norm) * cars.direction

        cos_curv = car_heading_x * next_dir_x + car_heading_y * next_dir_y
        sin_curv = car_heading_x * next_dir_y - car_heading_y * next_dir_x
        curvatures.append(cos_curv.unsqueeze(1))
        curvatures.append(sin_curv.unsqueeze(1))

    inputs = torch.cat([ray_dists, speed_input, *curvatures, *distances], dim=1)
    inputs[~cars.active] = 0.0
    return inputs, ray_dists_original



def line_intersection(line1, line2):
    """
    Compute intersection between two lines
    line1: (...,4)
    line2: (...,4)
    Returns (...,2) or NaN if no intersection
    """
    x1, y1, x2, y2 = line1.unbind(-1)
    x3, y3, x4, y4 = line2.unbind(-1)
    rx = x2 - x1
    ry = y2 - y1
    sx = x4 - x3
    sy = y4 - y3
    # 2D cross products for parametric segment intersection.
    den = rx * sy - ry * sx
    qpx = x3 - x1
    qpy = y3 - y1
    eps = 1e-6
    valid_den = den.abs() > eps
    safe_den = torch.where(valid_den, den, torch.ones_like(den))
    t = (qpx * sy - qpy * sx) / safe_den
    u = (qpx * ry - qpy * rx) / safe_den
    mask = valid_den & (t >= -eps) & (t <= 1.0 + eps) & (u >= -eps) & (u <= 1.0 + eps)
    px = x1 + t * rx
    py = y1 + t * ry
    px = torch.where(mask, px, torch.full_like(px, float("nan")))
    py = torch.where(mask, py, torch.full_like(py, float("nan")))
    return torch.stack([px, py], dim=-1)


def detect_pairwise_collisions(positions, radii=None):
    """Detect overlapping pairs among positions.

    positions: (N,2) tensor
    radii: None or (N,) tensor (if None, assume zeros)

    Returns: list of (i,j) pairs where i<j and overlap occurs
    """
    device = positions.device
    N = positions.shape[0]
    if N < 2:
        return []
    if radii is None:
        radii = torch.zeros(N, device=device)

    # vectorized: compute full pairwise distances and compare to sum of radii
    dists = torch.cdist(positions, positions)
    # Create matrix of pairwise radii sums
    rsum = radii.unsqueeze(0) + radii.unsqueeze(1)
    # consider only upper-triangle (i<j)
    mask = torch.triu(dists < rsum, diagonal=1)
    if not mask.any():
        return []
    idx_i, idx_j = mask.nonzero(as_tuple=True)
    # Return list of Python tuples for backward compatibility
    return list(zip(idx_i.tolist(), idx_j.tolist()))


def resolve_pairwise_collisions(positions, velocities, radii=None, masses=None, restitution=0.3, pos_correction=0.5):
    """Resolve pairwise collisions in-place and return updated positions, velocities, and impact magnitudes.

    - positions: (N,2) tensor
    - velocities: (N,2) tensor
    - radii: (N,) tensor or None
    - masses: (N,) tensor or None (defaults to ones)
    - restitution: coefficient of restitution (0..1)
    - pos_correction: fraction of penetration to correct by

    Returns: (positions, velocities, impact_magnitudes)
    impact_magnitudes: (N,) tensor with sum of absolute impulses applied to each body
    """
    device = positions.device
    N = positions.shape[0]
    if N < 2:
        return positions, velocities, torch.zeros(N, device=device)
    if radii is None:
        radii = torch.zeros(N, device=device)
    if masses is None:
        masses = torch.ones(N, device=device)

    impact = torch.zeros(N, device=device)

    pairs = detect_pairwise_collisions(positions, radii)
    for (i, j) in pairs:
        pi = positions[i]
        pj = positions[j]
        vi = velocities[i]
        vj = velocities[j]
        rij = pi - pj
        dist = torch.norm(rij)
        if dist.item() < 1e-6:
            # jitter to avoid zero division
            n = torch.tensor([1.0, 0.0], device=device)
            dist = torch.tensor(1e-3, device=device)
        else:
            n = rij / dist

        # penetration depth
        pen = (radii[i] + radii[j]) - dist
        if pen > 0:
            # positional correction (proportional to inverse masses)
            inv_m1 = 1.0 / (masses[i] + 1e-6)
            inv_m2 = 1.0 / (masses[j] + 1e-6)
            total_inv = inv_m1 + inv_m2
            if total_inv > 0:
                corr_i = (pen * pos_correction) * (inv_m1 / total_inv) * n
                corr_j = -(pen * pos_correction) * (inv_m2 / total_inv) * n
                positions[i] = positions[i] + corr_i
                positions[j] = positions[j] + corr_j

        # relative velocity along normal
        rel_v = (vi - vj)
        rel_norm = (rel_v * n).sum()
        # If moving apart skip impulse
        if rel_norm > 0:
            continue

        # impulse scalar
        inv_m1 = 1.0 / (masses[i] + 1e-6)
        inv_m2 = 1.0 / (masses[j] + 1e-6)
        j_impulse = -(1.0 + restitution) * rel_norm / (inv_m1 + inv_m2)

        # apply impulses
        vi = vi + (j_impulse * n) * inv_m1
        vj = vj - (j_impulse * n) * inv_m2

        velocities[i] = vi
        velocities[j] = vj

        impact[i] += j_impulse.abs()
        impact[j] += j_impulse.abs()

    return positions, velocities, impact



def rotate_track(track_coords):
    """Rotate the track 90 degrees.

    Accepts a list of (x,y), a numpy array shape (N,2), or a torch tensor and
    returns a list of (x,y) tuples suitable for downstream code that expects
    Python sequence coordinates.
    """
    # Convert to numpy array for robust indexing
    if isinstance(track_coords, torch.Tensor):
        arr = track_coords.cpu().numpy()
    else:
        arr = np.asarray(track_coords)

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("rotate_track expects an array-like of shape (N,2)")

    rotated = np.column_stack([-arr[:, 1], arr[:, 0]])

    # Return as list of (x,y) tuples to match callers like `load_track`
    return [tuple(p) for p in rotated]


def merge_checkpoints(path_a, path_b, out_path, alpha=0.5, map_location='cpu'):
    """Linearly merge two model checkpoints' `model_state` dictionaries.

    Args:
        path_a: path to first checkpoint (kept with weight alpha)
        path_b: path to second checkpoint (kept with weight 1-alpha)
        out_path: where to save the merged checkpoint
        alpha: blend weight for `path_a` in [0,1]
    """
    a = torch.load(path_a, map_location=map_location)
    b = torch.load(path_b, map_location=map_location)

    a_state = a.get('model_state', a)
    b_state = b.get('model_state', b)

    merged = {}
    for k, v in a_state.items():
        if k in b_state and v.shape == b_state[k].shape:
            merged[k] = v * alpha + b_state[k] * (1.0 - alpha)
        else:
            # fallback: prefer b if present, else a
            merged[k] = b_state.get(k, v)

    out = {}
    # include metadata from a where possible
    out.update({k: a.get(k, None) for k in ['epoch', 'optimizer_state', 'scheduler_state', 'total_epochs', 'best_reward', 'track_name']})
    out['model_state'] = merged

    # ensure directory exists
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(out, out_path)
    return out_path


def load_checkpoint_tolerant(checkpoint_path, model, optimizer=None, scheduler=None, device=None, verbose=True):
    """
    Load a checkpoint into `model` tolerantly: copy parameter tensors that match shapes
    and partially copy leading slices for parameters with expanded input dims.

    Attempts to restore optimizer and scheduler state when possible. Prints a
    short summary when `verbose` is True.

    Returns the loaded checkpoint dict (or whatever torch.load returned).
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

    ckpt = torch.load(checkpoint_path, map_location=device)
    # support checkpoint file that directly contains state_dict
    state_dict = ckpt.get('model_state', ckpt if isinstance(ckpt, dict) else {})

    model_state = model.state_dict()
    loaded = []
    partial = []
    skipped = []
    unexpected = []

    for k, v in state_dict.items():
        if k not in model_state:
            unexpected.append(k)
            continue
        tgt = model_state[k]
        try:
            if v.shape == tgt.shape:
                tgt.copy_(v)
                loaded.append(k)
            else:
                # copy overlapping leading slices where possible
                slices = tuple(slice(0, min(s, t)) for s, t in zip(v.shape, tgt.shape))
                tgt[slices].copy_(v[slices])
                partial.append((k, v.shape, tgt.shape))
        except Exception:
            skipped.append((k, getattr(v, 'shape', None), getattr(tgt, 'shape', None)))

    # try final load to ensure buffers/other state are set
    try:
        model.load_state_dict(model_state)
    except Exception:
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            if verbose:
                print('Warning: model.load_state_dict final reconcile failed')

    # If we performed any partial copies (shapes differed), avoid loading
    # optimizer/scheduler state because their internal buffers (exp avg
    # shapes) will likely mismatch the current model parameters.
    partial_occurred = len(partial) > 0 or len(skipped) > 0

    if optimizer is not None and isinstance(ckpt, dict) and ckpt.get('optimizer_state') is not None:
        if partial_occurred:
            if verbose:
                print('Note: partial model parameter copies detected; skipping optimizer state load to avoid shape mismatches')
        else:
            try:
                optimizer.load_state_dict(ckpt.get('optimizer_state'))
            except Exception:
                if verbose:
                    print('Warning: failed to load optimizer state')

    if scheduler is not None and isinstance(ckpt, dict) and ckpt.get('scheduler_state') is not None:
        if partial_occurred:
            if verbose:
                print('Note: partial model parameter copies detected; skipping scheduler state load')
        else:
            try:
                scheduler.load_state_dict(ckpt.get('scheduler_state'))
            except Exception:
                if verbose:
                    print('Warning: failed to load scheduler state')

    if verbose:
        print(f'Tolerant load: loaded={len(loaded)}, partial={len(partial)}, skipped={len(skipped)}, unexpected={len(unexpected)}')
        if partial:
            for k, s, t in partial:
                print(f'  partial: {k}: {s} -> {t}')

    return ckpt