"""
Twistor Point Cloud I/O (twistor/io.py)
=======================================
Lightweight, zero-dependency loader for ASCII/binary .ply and .obj 3D scans,
plus ASCII .ply export for inspecting point clouds in MeshLab/CloudCompare.
"""

import os
import struct
import numpy as np
import torch


def load_point_cloud(filepath, max_points=None, normalize=True, device="cpu"):
    """
    Loads 3D vertex coordinates from a .ply (ASCII or binary_little_endian) or .obj file.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".ply":
        pts = _parse_ply(filepath)
    elif ext == ".obj":
        pts = _parse_obj(filepath)
    else:
        raise ValueError(f"Unsupported 3D file extension '{ext}'. Use .ply or .obj.")

    if max_points is not None and len(pts) > max_points:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]

    tensor_pts = torch.from_numpy(pts).to(device=device, dtype=torch.float32)

    if normalize:
        centroid = tensor_pts.mean(dim=0, keepdim=True)
        tensor_pts = tensor_pts - centroid
        max_radius = torch.linalg.norm(tensor_pts, dim=-1).max()
        if max_radius > 0:
            tensor_pts = tensor_pts / max_radius

    return tensor_pts


# Backward-compatible alias
load_ply_points = load_point_cloud


def save_ply_points(filepath, points):
    """
    Saves a [P, 3] tensor or numpy array as a standard ASCII .ply point cloud.
    """
    if isinstance(points, torch.Tensor):
        pts = points.detach().cpu().numpy()
    else:
        pts = np.asarray(points, dtype=np.float32)

    with open(filepath, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def _parse_obj(filepath):
    verts = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not verts:
        raise ValueError("OBJ file contains 0 vertices.")
    return np.array(verts, dtype=np.float32)


def _parse_ply(filepath):
    with open(filepath, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY file: unexpected EOF before end_header.")
            decoded = line.decode("ascii", errors="ignore").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        fmt = None
        vertex_count = 0
        in_vertex_element = False
        properties = []

        for line in header_lines:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                if tokens[1] == "vertex":
                    in_vertex_element = True
                    vertex_count = int(tokens[2])
                else:
                    in_vertex_element = False
            elif tokens[0] == "property" and in_vertex_element:
                if tokens[1] != "list":
                    properties.append((tokens[1], tokens[2]))

        if vertex_count == 0:
            raise ValueError("PLY file contains 0 vertices.")

        prop_names = [p[1] for p in properties]
        ix, iy, iz = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")

        if fmt == "ascii":
            pts = np.zeros((vertex_count, 3), dtype=np.float32)
            for i in range(vertex_count):
                parts = f.readline().decode("ascii").split()
                pts[i, 0] = float(parts[ix])
                pts[i, 1] = float(parts[iy])
                pts[i, 2] = float(parts[iz])
        elif fmt == "binary_little_endian":
            type_map = {
                "char": ("b", 1), "int8": ("b", 1),
                "uchar": ("B", 1), "uint8": ("B", 1),
                "short": ("h", 2), "int16": ("h", 2),
                "ushort": ("H", 2), "uint16": ("H", 2),
                "int": ("i", 4), "int32": ("i", 4),
                "uint": ("I", 4), "uint32": ("I", 4),
                "float": ("f", 4), "float32": ("f", 4),
                "double": ("d", 8), "float64": ("d", 8),
            }
            struct_fmt = "<" + "".join(type_map[p[0]][0] for p in properties)
            stride = struct.calcsize(struct_fmt)
            raw_bytes = f.read(vertex_count * stride)

            pts = np.zeros((vertex_count, 3), dtype=np.float32)
            for i in range(vertex_count):
                row = struct.unpack_from(struct_fmt, raw_bytes, i * stride)
                pts[i, 0] = row[ix]
                pts[i, 1] = row[iy]
                pts[i, 2] = row[iz]
        else:
            raise NotImplementedError(f"Unsupported PLY format: {fmt}")

    return pts