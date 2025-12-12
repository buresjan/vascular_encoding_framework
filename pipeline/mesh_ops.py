"""
Mesh helpers: hashing, conversions, append, clip, and STL repair.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyvista as pv
import numpy as np

try:
    import trimesh
    import trimesh.repair as trepair
    import trimesh.grouping as tgroup
except ModuleNotFoundError:  # pragma: no cover
    trimesh = None
    trepair = None
    tgroup = None


def hash_params(params: dict, length: int = 8) -> str:
    payload = json.dumps(params, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:length]


def stl_from_vtp(vtp_path: Path, stl_path: Path) -> Path:
    mesh = pv.read(str(vtp_path))
    mesh.save(str(stl_path))
    return stl_path


def append_meshes(stl_a: Path, stl_b: Path, out_path: Path) -> Path:
    a = pv.read(str(stl_a))
    b = pv.read(str(stl_b))
    merged = a.merge(b, merge_points=False)
    merged.save(str(out_path))
    return out_path


def clip_bottom(
    stl_path: Path,
    out_path: Path,
    clip_offset: float = 0.5,
) -> Path:
    """Clip geometry from the bottom (keep above plane) and rebase to Z=0."""
    mesh = pv.read(str(stl_path))
    z_min = mesh.bounds[4]
    origin = (0.0, 0.0, z_min + clip_offset)

    clipped = mesh.clip(normal=(0, 0, -1), origin=origin, invert=True)
    clipped.points[:, 2] -= clipped.bounds[4]

    clipped = clipped.triangulate().clean()
    clipped.save(str(out_path))
    return out_path


def repair_stl(
    in_path: Path,
    out_path: Path,
    *,
    fill_holes: bool = True,
) -> dict:
    """
    Repair an STL using trimesh: remove duplicates/degenerates, split, fill holes, fix normals.
    Returns a log dict with simple diagnostics.
    """
    if trimesh is None or trepair is None:
        raise ImportError("trimesh is required for repair_stl; please pip install trimesh")
    m = trimesh.load_mesh(in_path, process=False)
    log = {}

    # Remove duplicate faces manually
    faces_sorted = np.sort(m.faces, axis=1)
    unique_idx = tgroup.unique_rows(faces_sorted)[1]
    m.update_faces(unique_idx)
    m.remove_unreferenced_vertices()
    # Remove degenerate/zero-area faces
    area = m.area_faces
    keep = area > 1e-12
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    log["faces"] = int(m.faces.shape[0])
    log["verts"] = int(m.vertices.shape[0])

    comps = m.split(only_watertight=False)
    repaired = []
    for c in comps:
        if fill_holes:
            try:
                c.fill_holes()
            except Exception:
                try:
                    trepair.fill_holes(c)
                except Exception:
                    pass
        try:
            c.fix_normals()
        except Exception:
            try:
                trepair.fix_normals(c)
            except Exception:
                pass
        faces_sorted = np.sort(c.faces, axis=1)
        unique_idx = tgroup.unique_rows(faces_sorted)[1]
        c.update_faces(unique_idx)
        area = c.area_faces
        keep = area > 1e-12
        c.update_faces(keep)
        c.remove_unreferenced_vertices()
        repaired.append(c)

    if len(repaired) == 1:
        m_fixed = repaired[0]
    else:
        m_fixed = trimesh.util.concatenate(repaired)

    try:
        m_fixed.fix_normals()
    except Exception:
        try:
            trepair.fix_normals(m_fixed)
        except Exception:
            pass
    faces_sorted = np.sort(m_fixed.faces, axis=1)
    unique_idx = tgroup.unique_rows(faces_sorted)[1]
    m_fixed.update_faces(unique_idx)
    m_fixed.remove_unreferenced_vertices()

    m_fixed.export(out_path)
    log["components"] = len(repaired)
    log["watertight"] = m_fixed.is_watertight
    log["euler"] = m_fixed.euler_number
    return log
