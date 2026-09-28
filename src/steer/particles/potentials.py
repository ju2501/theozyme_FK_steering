import numpy as np


def difference_update(log_weights, previous_phi, rewards, scale):
    """Accumulate log weights; never resurrect an invalid lineage.

    previous_phi starts at ZERO (not the first reward). At every scoring event,
    log G = scale * reward - previous_phi. The final event is always scored.
    """
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("Scale must be finite and nonnegative")
    w, old, r = [np.asarray(x, float) for x in (log_weights, previous_phi, rewards)]
    if w.ndim != 1 or w.shape != old.shape or w.shape != r.shape:
        raise ValueError("Particle vectors must have matching shapes")
    if any(np.isnan(x).any() or np.isposinf(x).any() for x in (w, old, r)):
        raise ValueError("NaN/+inf is not a valid potential")
    valid = np.isfinite(w) & np.isfinite(old) & np.isfinite(r)
    new_phi, updated = np.full_like(r, -np.inf), np.full_like(r, -np.inf)
    try:
        with np.errstate(over="raise", invalid="raise"):
            new_phi[valid] = scale * r[valid]
            updated[valid] = w[valid] + new_phi[valid] - old[valid]
    except FloatingPointError as exc:
        raise ValueError("Potential overflow; reduce reward scale") from exc
    if np.isnan(updated).any() or np.isposinf(updated).any():
        raise ValueError("Potential overflow; reduce reward scale")
    return updated, new_phi
