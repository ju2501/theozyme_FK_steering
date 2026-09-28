import numpy as np


def _pair(mobile, reference):
    x, y = np.asarray(mobile, float), np.asarray(reference, float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) == 0:
        raise ValueError("Matching nonempty (N, 3) coordinate arrays required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Coordinates must be finite")
    return x, y


def kabsch(mobile, reference):
    """Return aligned coordinates, RMSD, rotation and translation (row vectors).

    Reflections are forbidden. This aligns only the atoms passed by the caller;
    an active-site-only fit does NOT test the ligand-relative pose.
    """
    x, y = _pair(mobile, reference)
    if len(x) < 3:
        raise ValueError("At least three atoms required")
    xc, yc = x.mean(0), y.mean(0)
    u, _, vt = np.linalg.svd((x - xc).T @ (y - yc))
    fix = np.eye(3)
    fix[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ fix @ vt
    translation = yc - xc @ rotation
    aligned = x @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - y) ** 2, axis=1))))
    return aligned, rmsd, rotation, translation


def anchored_rmsd(mobile_anchors, reference_anchors, mobile_atoms, reference_atoms):
    """Fit anchors only, then measure query atoms without fitting them again."""
    a, b = _pair(mobile_anchors, reference_anchors)
    if len(a) < 3 or min(np.linalg.matrix_rank(a - a.mean(0)),
                         np.linalg.matrix_rank(b - b.mean(0))) < 2:
        raise ValueError("Anchors must include three non-collinear points")
    x, y = _pair(mobile_atoms, reference_atoms)
    _, _, rotation, translation = kabsch(a, b)
    return float(np.sqrt(np.mean(np.sum((x @ rotation + translation - y) ** 2, axis=1))))
