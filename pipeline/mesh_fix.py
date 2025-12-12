#!/usr/bin/env python3
"""
Mesh repair + outlet capping + watertight voxel rebuild for STL using trimesh.

Usage:
    # 1. Repair mesh (drop shards, patch only small cracks, keep 4 outlets open)
    python mesh_fix.py repair input.stl repaired.stl

    # 2. Cap the 4 axis-aligned outlets (still may not be watertight if interior cracks remain)
    python mesh_fix.py cap repaired.stl capped.stl

    # 3. (Recommended) Rebuild a watertight outer surface via voxelization
    python mesh_fix.py voxel_rebuild capped.stl watertight.stl --pitch 1.0
"""

import argparse
import collections
import sys

import numpy as np
import trimesh
from trimesh import grouping, repair, util

# ---------------------------
# Boundary + geometry helpers
# ---------------------------

def compute_boundary_loops(mesh: trimesh.Trimesh):
    """
    Find boundary loops (chains of edges with only one adjacent face).
    Returns a list of loops, each loop is a list of vertex indices.
    """
    edges = mesh.edges_sorted
    edge_counts = collections.Counter(map(tuple, edges))
    boundary_edges = [e for e, c in edge_counts.items() if c == 1]

    if not boundary_edges:
        return []

    # build adjacency graph on boundary edges
    adj = collections.defaultdict(list)
    for v0, v1 in boundary_edges:
        adj[v0].append(v1)
        adj[v1].append(v0)

    loops = []
    visited_edges = set()

    for start in list(adj.keys()):
        for nb in adj[start]:
            e = tuple(sorted((start, nb)))
            if e in visited_edges:
                continue

            loop = [start, nb]
            visited_edges.add(e)
            prev, cur = start, nb

            while True:
                neighbors = adj[cur]
                # all neighbors except where we came from
                next_candidates = [v for v in neighbors if v != prev]
                if not next_candidates:
                    break

                next_v = None
                for v in next_candidates:
                    e2 = tuple(sorted((cur, v)))
                    if e2 not in visited_edges:
                        next_v = v
                        break

                if next_v is None:
                    break

                loop.append(next_v)
                visited_edges.add(tuple(sorted((cur, next_v))))
                prev, cur = cur, next_v

                if cur == start:
                    # closed loop
                    break

            loops.append(loop)

    return loops


def loop_plane_stats(mesh: trimesh.Trimesh, loop):
    """
    Fit a best-fit plane to a boundary loop and return info:
    normal, axis alignment, loop length, centroid, etc.
    """
    verts = mesh.vertices[np.unique(loop)]
    centroid = verts.mean(axis=0)
    A = verts - centroid

    if len(verts) < 3:
        normal = np.zeros(3)
        axis_alignment = np.zeros(3)
    else:
        cov = A.T @ A
        w, v = np.linalg.eigh(cov)
        normal = v[:, np.argmin(w)]
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        axis_alignment = np.abs(normal)

    return {
        "centroid": centroid,
        "normal": normal,
        "axis_alignment": axis_alignment,
        "max_axis_alignment": float(axis_alignment.max()),
        "argmax_axis": int(axis_alignment.argmax()),
        "loop_len": len(loop),
        "spread": A.std(axis=0),
    }


def polygon_area_2d(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_point_in_triangle(p, a, b, c, eps=1e-9):
    """
    Barycentric point-in-triangle test in 2D.
    """
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < eps:
        return False

    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv

    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)


def triangulate_polygon_ear(points2d, eps=1e-9):
    """
    Simple ear-clipping polygon triangulation in 2D.
    points2d: (n, 2) vertices ordered along boundary.
    Returns list of (i, j, k) indices into points2d.
    """
    n = len(points2d)
    if n < 3:
        return []

    area = polygon_area_2d(points2d)
    if area < 0:
        idx = list(range(n - 1, -1, -1))
    else:
        idx = list(range(n))

    triangles = []

    def is_convex(i):
        a = points2d[idx[(i - 1) % len(idx)]]
        b = points2d[idx[i]]
        c = points2d[idx[(i + 1) % len(idx)]]
        ab = b - a
        bc = c - b
        cross = ab[0] * bc[1] - ab[1] * bc[0]
        return cross > eps

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        ear_found = False

        for i in range(len(idx)):
            if not is_convex(i):
                continue

            ia = idx[(i - 1) % len(idx)]
            ib = idx[i]
            ic = idx[(i + 1) % len(idx)]

            a = points2d[ia]
            b = points2d[ib]
            c = points2d[ic]

            ear = True
            for j in range(len(idx)):
                j_idx = idx[j]
                if j_idx in (ia, ib, ic):
                    continue
                p = points2d[j_idx]
                if is_point_in_triangle(p, a, b, c, eps=eps):
                    ear = False
                    break

            if ear:
                triangles.append((ia, ib, ic))
                del idx[i]
                ear_found = True
                break

        if not ear_found:
            # Polygon might be self-intersecting or degenerate
            break

    if len(idx) == 3:
        triangles.append(tuple(idx))

    return triangles


def project_loop_to_plane(mesh: trimesh.Trimesh, loop, normal=None):
    """
    Project a loop of vertices into a 2D coordinate system of its best-fit plane.
    Returns (points2d, centroid, u, v, normal).
    """
    verts = mesh.vertices[np.array(loop)]
    centroid = verts.mean(axis=0)

    if normal is None or np.linalg.norm(normal) < 1e-8:
        A = verts - centroid
        cov = A.T @ A
        w, v = np.linalg.eigh(cov)
        normal = v[:, np.argmin(w)]

    normal = np.asarray(normal, dtype=float)
    normal /= (np.linalg.norm(normal) + 1e-12)

    # choose a tangent not parallel to the normal
    if abs(normal[0]) < 0.9:
        tangent = np.array([1.0, 0.0, 0.0])
    else:
        tangent = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, tangent)
    u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(normal, u)
    v /= (np.linalg.norm(v) + 1e-12)

    rel = verts - centroid
    pts2d = np.column_stack((rel.dot(u), rel.dot(v)))
    return pts2d, centroid, u, v, normal


def cap_loop_on_mesh(mesh: trimesh.Trimesh, loop, normal=None):
    """
    Cap a single boundary loop by triangulating in its plane and
    adding faces to the mesh. Reuses existing loop vertices.
    """
    loop = list(loop)
    pts2d, _, _, _, nrm = project_loop_to_plane(mesh, loop, normal=normal)
    tris = triangulate_polygon_ear(pts2d)
    if len(tris) == 0:
        return mesh, 0

    faces_new = np.array(
        [[loop[a], loop[b], loop[c]] for (a, b, c) in tris],
        dtype=np.int64,
    )
    mesh.faces = np.vstack((mesh.faces, faces_new))
    return mesh, len(faces_new)


def loop_size_metrics(mesh: trimesh.Trimesh, loop):
    """
    Compute simple size metrics for a boundary loop:
    - diag: bounding-box diagonal in 3D
    - area: polygon area in best-fit plane (2D)
    """
    verts = mesh.vertices[np.array(loop)]
    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    diag = float(np.linalg.norm(bmax - bmin))

    if len(verts) < 3:
        area = 0.0
    else:
        pts2d, _, _, _, _ = project_loop_to_plane(mesh, loop)
        area = abs(polygon_area_2d(pts2d))

    return diag, area


# ---------------------------
# Repair helpers
# ---------------------------

def basic_cleanup(mesh: trimesh.Trimesh, digits_vertex=6):
    """
    Remove NaNs, degenerate faces, duplicate faces, unreferenced vertices,
    and merge nearly-identical vertices. Compatible with older trimesh.
    """
    mesh.remove_infinite_values()
    # Newer trimesh exposes remove_degenerate_faces; older has nondegenerate_faces()
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    else:
        mesh.update_faces(mesh.nondegenerate_faces())

    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    else:
        faces_sorted = np.sort(mesh.faces, axis=1)
        unique_idx = grouping.unique_rows(faces_sorted, keep_order=True)[0]
        if len(unique_idx) != len(mesh.faces):
            mesh.update_faces(unique_idx)

    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices(digits_vertex=digits_vertex)
    return mesh


def drop_tiny_components(mesh: trimesh.Trimesh,
                         min_rel_diag=0.02,
                         min_faces=10):
    """
    Drop connected components that are both small in extent and low in face count.
    """
    components = mesh.split(only_watertight=False)
    if len(components) == 1:
        return mesh

    diags = np.array([np.linalg.norm(c.extents) for c in components])
    faces = np.array([len(c.faces) for c in components])
    main_diag = diags.max()

    keep = []
    for c, d, f in zip(components, diags, faces):
        if d >= min_rel_diag * main_diag and f >= min_faces:
            keep.append(c)

    if not keep:
        # Fallback: keep the largest component
        keep.append(components[int(diags.argmax())])

    return util.concatenate(keep)


def patch_non_outlet_loops(mesh: trimesh.Trimesh,
                           axis_thresh=0.9999,
                           min_outlet_len=40,
                           num_outlets=4,
                           max_iters=4,
                           small_diag_thresh=3.0,
                           small_area_thresh=1.0,
                           verbose=False):
    """
    Iteratively cap SMALL boundary loops (cracks) while keeping the
    4 dominant axis-aligned 'outlet' loops open.

    A loop is auto-patched if:
        - it's NOT one of the selected outlet loops
        - len(loop) >= 3
        - diag(3D bbox) <= small_diag_thresh
        - area (in best-fit plane) <= small_area_thresh
    """
    for it in range(max_iters):
        loops = compute_boundary_loops(mesh)
        if not loops:
            break

        # plane stats for normals / outlet detection
        stats_list = [loop_plane_stats(mesh, loop) for loop in loops]

        # Loops that look like outlets (axis aligned & long)
        big_axis = [
            (i, s) for i, s in enumerate(stats_list)
            if s["max_axis_alignment"] >= axis_thresh
            and s["loop_len"] >= min_outlet_len
        ]
        big_axis_sorted = sorted(big_axis,
                                 key=lambda t: t[1]["loop_len"],
                                 reverse=True)
        outlet_indices = set(i for i, _ in big_axis_sorted[:num_outlets])

        added_iter = 0
        for i, (loop, s) in enumerate(zip(loops, stats_list)):
            if i in outlet_indices:
                continue
            if len(loop) < 3:
                continue

            diag, area = loop_size_metrics(mesh, loop)
            # Only patch truly small holes
            if diag > small_diag_thresh or area > small_area_thresh:
                continue

            mesh, added = cap_loop_on_mesh(mesh, loop, normal=s["normal"])
            added_iter += added

        if verbose:
            print(f"[repair] iter={it} loops={len(loops)} "
                  f"outlets={len(outlet_indices)} added_triangles={added_iter}")

        if added_iter == 0:
            break

    return mesh


def repair_pipeline(mesh: trimesh.Trimesh,
                    digits_vertex=6,
                    min_rel_diag=0.02,
                    min_faces=10,
                    axis_thresh=0.9999,
                    min_outlet_len=40,
                    num_outlets=4,
                    max_iters=4,
                    small_diag_thresh=3.0,
                    small_area_thresh=1.0,
                    verbose=False):
    """
    Full repair:
    - cleanup
    - drop tiny components
    - fix normals
    - patch SMALL non-outlet boundary loops (cracks)
    - fix normals again & drop fresh tiny components
    """
    mesh = basic_cleanup(mesh, digits_vertex=digits_vertex)
    if verbose:
        print("[repair] after basic_cleanup:", len(mesh.vertices),
              "verts,", len(mesh.faces), "faces")

    mesh = drop_tiny_components(mesh,
                                min_rel_diag=min_rel_diag,
                                min_faces=min_faces)
    if verbose:
        print("[repair] after drop_tiny_components:", len(mesh.vertices),
              "verts,", len(mesh.faces), "faces")

    repair.fix_normals(mesh)
    repair.fix_winding(mesh)

    mesh = patch_non_outlet_loops(mesh,
                                  axis_thresh=axis_thresh,
                                  min_outlet_len=min_outlet_len,
                                  num_outlets=num_outlets,
                                  max_iters=max_iters,
                                  small_diag_thresh=small_diag_thresh,
                                  small_area_thresh=small_area_thresh,
                                  verbose=verbose)

    repair.fix_normals(mesh)
    repair.fix_winding(mesh)

    mesh = drop_tiny_components(mesh,
                                min_rel_diag=min_rel_diag,
                                min_faces=min_faces)
    if verbose:
        print("[repair] final:", len(mesh.vertices), "verts,",
              len(mesh.faces), "faces")

    return mesh


# ---------------------------
# Outlet capping
# ---------------------------

def cap_outlets_pipeline(mesh: trimesh.Trimesh,
                         axis_thresh=0.9999,
                         min_outlet_len=40,
                         num_outlets=4,
                         verbose=False):
    """
    Cap only the 4 dominant axis-aligned outlet loops.
    """
    loops = compute_boundary_loops(mesh)
    stats_list = [loop_plane_stats(mesh, loop) for loop in loops]

    big_axis = [
        (i, s) for i, s in enumerate(stats_list)
        if s["max_axis_alignment"] >= axis_thresh and s["loop_len"] >= min_outlet_len
    ]
    big_axis_sorted = sorted(big_axis,
                             key=lambda t: t[1]["loop_len"],
                             reverse=True)
    outlet_indices = [i for i, _ in big_axis_sorted[:num_outlets]]

    if verbose:
        print("[cap] outlet loops (index, length, axis_alignment, centroid):")
        for i in outlet_indices:
            s = stats_list[i]
            print(f"    {i}: len={s['loop_len']}, axis={s['axis_alignment']}, "
                  f"centroid={s['centroid']}")

    added_total = 0
    for i in outlet_indices:
        loop = loops[i]
        s = stats_list[i]
        mesh, added = cap_loop_on_mesh(mesh, loop, normal=s["normal"])
        added_total += added
        if verbose:
            print(f"[cap] loop {i} len={len(loop)} -> added {added} tris")

    repair.fix_normals(mesh)
    repair.fix_winding(mesh)

    # Optional: clean up any stray tiny components produced in the process
    mesh = drop_tiny_components(mesh,
                                min_rel_diag=0.0,
                                min_faces=3)

    if verbose:
        print("[cap] total added triangles:", added_total)

    return mesh


# ---------------------------
# Voxel-based watertight rebuild
# ---------------------------

def voxel_rebuild_pipeline(mesh: trimesh.Trimesh,
                           pitch=None,
                           target_voxels=120,
                           verbose=False):
    """
    Rebuild a watertight outer surface by voxelizing and running marching cubes.

    This DOES NOT add any explicit planar caps on interior loops; instead it
    approximates the geometry as a binary voxel volume and extracts a closed
    surface from that volume.

    Recommended usage:
        - run on the output of the 'cap' step
        - choose pitch comparable to the voxel size you will use later
    """
    # Choose pitch if not specified: ~target_voxels on the longest axis
    if pitch is None:
        extent = float(np.max(mesh.extents))
        pitch = extent / float(target_voxels)

    if verbose:
        print(f"[voxel] bbox extents={mesh.extents}, pitch={pitch}")

    vg = mesh.voxelized(pitch=pitch)
    if verbose:
        print(f"[voxel] grid shape={vg.shape}, volume={vg.volume}")

    # Fill interior; this is robust to small cracks / non-watertight input
    vg.fill()

    m2 = vg.marching_cubes

    repair.fix_normals(m2)
    repair.fix_winding(m2)

    # Drop tiny components if any
    m2 = drop_tiny_components(m2, min_rel_diag=0.0, min_faces=10)

    if verbose:
        print("[voxel] rebuilt mesh:",
              len(m2.vertices), "verts,",
              len(m2.faces), "faces,",
              "watertight=", m2.is_watertight,
              "euler=", m2.euler_number)

    return m2, pitch


# ---------------------------
# Diagnostics
# ---------------------------

def print_boundary_summary(mesh: trimesh.Trimesh):
    """
    Quick summary: number of boundary loops and their sizes.
    """
    loops = compute_boundary_loops(mesh)
    print("boundary loops:", len(loops))
    for i, loop in enumerate(loops):
        stats = loop_plane_stats(mesh, loop)
        diag, area = loop_size_metrics(mesh, loop)
        print(f"  loop {i:3d}: len={stats['loop_len']:4d}, "
              f"diag={diag:6.2f}, area={area:7.2f}, "
              f"axis_align={stats['axis_alignment']}, "
              f"centroid={stats['centroid']}")


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Repair STL mesh, cap 4 axis-aligned outlets, and optionally rebuild a watertight outer surface via voxelization."
    )
    sub = parser.add_subparsers(dest="command")

    # repair
    p_repair = sub.add_parser(
        "repair",
        help="repair mesh but keep 4 large axis-aligned outlets open (patch only small cracks)",
    )
    p_repair.add_argument("input", help="input STL file")
    p_repair.add_argument("output", help="output STL file")
    p_repair.add_argument(
        "--small-diag", type=float, default=3.0,
        help="max 3D diagonal length for holes to auto-patch (default: 3.0)",
    )
    p_repair.add_argument(
        "--small-area", type=float, default=1.0,
        help="max projected area for holes to auto-patch (default: 1.0)",
    )
    p_repair.add_argument(
        "--verbose", action="store_true",
        help="print progress information",
    )

    # cap outlets
    p_cap = sub.add_parser(
        "cap",
        help="cap the 4 axis-aligned outlets on a repaired mesh",
    )
    p_cap.add_argument("input", help="input (repaired) STL file")
    p_cap.add_argument("output", help="output STL file")
    p_cap.add_argument(
        "--verbose", action="store_true",
        help="print progress information",
    )

    # voxel-based rebuild
    p_vox = sub.add_parser(
        "voxel_rebuild",
        help="rebuild a watertight outer surface via voxelization + marching cubes",
    )
    p_vox.add_argument("input", help="input STL file (ideally capped)")
    p_vox.add_argument("output", help="output STL file (watertight)")
    p_vox.add_argument(
        "--pitch", type=float, default=None,
        help="voxel size in model units; if omitted, chosen from model extents and --target-voxels",
    )
    p_vox.add_argument(
        "--target-voxels", type=int, default=120,
        help="approx. number of voxels along longest dimension if --pitch is not given (default: 120)",
    )
    p_vox.add_argument(
        "--verbose", action="store_true",
        help="print progress information",
    )

    # stats
    p_stats = sub.add_parser(
        "stats",
        help="print basic topological stats and boundary loop summary for a mesh",
    )
    p_stats.add_argument("input", help="input STL file")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Load mesh
    mesh = trimesh.load(args.input)
    if isinstance(mesh, trimesh.Scene):
        mesh = util.concatenate(mesh.dump())

    if args.command == "repair":
        mesh = repair_pipeline(
            mesh,
            small_diag_thresh=args.small_diag,
            small_area_thresh=args.small_area,
            verbose=args.verbose,
        )
        mesh.export(args.output)
        if args.verbose:
            print(f"[done] wrote {args.output}")

    elif args.command == "cap":
        mesh = cap_outlets_pipeline(mesh, verbose=args.verbose)
        mesh.export(args.output)
        if args.verbose:
            print(f"[done] wrote {args.output}")

    elif args.command == "voxel_rebuild":
        mesh, pitch = voxel_rebuild_pipeline(
            mesh,
            pitch=args.pitch,
            target_voxels=args.target_voxels,
            verbose=args.verbose,
        )
        mesh.export(args.output)
        if args.verbose:
            print(f"[done] wrote {args.output} (pitch={pitch})")

    elif args.command == "stats":
        print("vertices:", len(mesh.vertices), "faces:", len(mesh.faces))
        print("watertight:", mesh.is_watertight)
        print("euler:", mesh.euler_number)
        print("components:", len(mesh.split(only_watertight=False)))
        print_boundary_summary(mesh)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
