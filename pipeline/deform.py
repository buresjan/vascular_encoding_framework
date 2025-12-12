"""
Wrap mesh rim deformation.
"""

import numpy as np
import pyvista as pv
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


def deform_rim_to_target(
    stl_path,
    orig_rim_vtp,
    target_rim_vtp,
    out_stl_path,
    r1=5.0,
    r2=20.0,
):
    """Deform STL so the original rim matches the target rim via harmonic extension."""
    surf = pv.read(stl_path)
    V = surf.points.copy()
    faces = surf.faces.reshape(-1, 4)
    F = faces[:, 1:4].astype(int)

    rim_orig = pv.read(orig_rim_vtp).points
    rim_tgt = pv.read(target_rim_vtp).points

    N = V.shape[0]

    tree_V = cKDTree(V)
    dist_rim_to_V, rim_vert_idx = tree_V.query(rim_orig)
    print("max dist(orig rim -> mesh vertex) =", dist_rim_to_V.max())

    tree_rim_tgt = cKDTree(rim_tgt)
    _, idx_tgt = tree_rim_tgt.query(rim_orig)
    rim_tgt_matched = rim_tgt[idx_tgt]

    D_samples = rim_tgt_matched - rim_orig

    vert2samples = {}
    for i, v_idx in enumerate(rim_vert_idx):
        vert2samples.setdefault(int(v_idx), []).append(i)

    rim_vertices = np.array(sorted(vert2samples.keys()), dtype=int)

    u = np.zeros_like(V)
    for v in rim_vertices:
        sample_ids = vert2samples[v]
        u[v] = D_samples[sample_ids].mean(axis=0)

    I, J, data = [], [], []
    deg = np.zeros(N, dtype=int)

    for (a, b, c) in F:
        for i, j in ((a, b), (b, c), (c, a)):
            if i == j:
                continue
            deg[i] += 1
            deg[j] += 1
            I.extend([i, j])
            J.extend([j, i])
            data.extend([-1.0, -1.0])

    for i in range(N):
        I.append(i)
        J.append(i)
        data.append(float(deg[i]))

    L = sparse.csr_matrix((data, (I, J)), shape=(N, N))

    all_idx = np.arange(N)
    B = rim_vertices
    F_idx = np.setdiff1d(all_idx, B)

    L_ff = L[F_idx, :][:, F_idx]
    L_fb = L[F_idx, :][:, B]

    uB = u[B]

    rhs_x = -L_fb @ uB[:, 0]
    rhs_y = -L_fb @ uB[:, 1]
    rhs_z = -L_fb @ uB[:, 2]

    uF_x = spsolve(L_ff, rhs_x)
    uF_y = spsolve(L_ff, rhs_y)
    uF_z = spsolve(L_ff, rhs_z)

    u[F_idx, 0] = uF_x
    u[F_idx, 1] = uF_y
    u[F_idx, 2] = uF_z

    tree_rim_orig = cKDTree(rim_orig)
    dist_all, _ = tree_rim_orig.query(V)

    w = np.zeros(N, dtype=float)
    inside1 = dist_all <= r1
    inside2 = dist_all >= r2
    mid = (~inside1) & (~inside2)

    w[inside1] = 1.0
    w[inside2] = 0.0
    t = (dist_all[mid] - r1) / (r2 - r1)
    s = 3.0 * t**2 - 2.0 * t**3
    w[mid] = 1.0 - s

    V_new = V + w[:, None] * u

    surf.points = V_new
    surf.save(out_stl_path)
    print("Saved:", out_stl_path)
    return out_stl_path
