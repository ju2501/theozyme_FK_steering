import numpy as np


def normalize_log_weights(log_weights):
    logw = np.asarray(log_weights, dtype=float)
    if logw.ndim != 1 or not len(logw) or np.isnan(logw).any() or np.isposinf(logw).any():
        raise ValueError("Log weights must be a nonempty vector of finite values or -inf")
    if not np.isfinite(logw).any():
        raise ValueError("All particles are invalid / have zero weight")
    shifted = np.exp(logw - np.max(logw))
    return shifted / shifted.sum()


def _weights(weights):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or (w < 0).any() or w.max() <= 0:
        raise ValueError("Invalid weights")
    scaled = w / w.max()
    return scaled / scaled.sum()


def effective_sample_size(weights):
    w = _weights(weights)
    return float(1.0 / np.dot(w, w))


def systematic_resample(weights, rng):
    w = _weights(weights)
    positions = (rng.random() + np.arange(len(w))) / len(w)
    cdf = np.cumsum(w)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, positions, side="right")
