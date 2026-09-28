"""Geometry uses Angstrom coordinates and degrees; invalid geometry raises."""
import numpy as np


def points(*args):
    value = np.asarray(args, dtype=float)
    if value.shape != (len(args), 3) or not np.isfinite(value).all():
        raise ValueError("Expected finite 3D points")
    return value


def distance(a, b):
    a, b = points(a, b)
    return float(np.linalg.norm(a - b))


def angle(a, b, c):
    a, b, c = points(a, b, c)
    u, v = a - b, c - b
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("Angle contains a zero-length bond")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / norm, -1, 1))))


def dihedral(a, b, c, d):
    a, b, c, d = points(a, b, c, d)
    axis = c - b
    length = np.linalg.norm(axis)
    if length < 1e-12:
        raise ValueError("Dihedral central bond has zero length")
    axis = axis / length
    v = a - b - np.dot(a - b, axis) * axis
    w = d - c - np.dot(d - c, axis) * axis
    if min(np.linalg.norm(v), np.linalg.norm(w)) < 1e-12:
        raise ValueError("Dihedral contains collinear bonds")
    return float(np.degrees(np.arctan2(np.dot(np.cross(axis, v), w), np.dot(v, w))))


def angular_difference(a, b):
    """Signed shortest periodic difference in [-180, 180)."""
    if not np.isfinite([a, b]).all():
        raise ValueError("Angles must be finite")
    return float((a - b + 180) % 360 - 180)
