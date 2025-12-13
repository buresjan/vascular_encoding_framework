"""
Mesh helpers: hashing, conversions, append, and clip.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyvista as pv


def hash_params(params: dict, length: int = 8) -> str:
    payload = json.dumps(params, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:length]


def stl_from_vtp(vtp_path: Path, stl_path: Path) -> Path:
    """Convert a VTP surface to STL."""
    mesh = pv.read(str(vtp_path))
    mesh.save(str(stl_path))
    return stl_path


def append_meshes(stl_a: Path, stl_b: Path, out_path: Path) -> Path:
    """Merge two STL surfaces and clean overlapping points."""
    a = pv.read(str(stl_a))
    b = pv.read(str(stl_b))
    merged = a.merge(b, merge_points=True).clean()
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
