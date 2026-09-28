from dataclasses import dataclass
import numpy as np
from .resample import normalize_log_weights, effective_sample_size, systematic_resample
from .potentials import difference_update


@dataclass(frozen=True)
class FKConfig:
    num_particles: int = 16
    steps: int = 40
    scale: float = 2.0
    ess_fraction: float = 0.5
    every: int = 4
    start_fraction: float = 0.25
    end_fraction: float = 0.9

    def __post_init__(self):
        for name in ("num_particles", "steps", "every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite([self.scale, self.ess_fraction, self.start_fraction, self.end_fraction]).all():
            raise ValueError("FK parameters must be finite")
        if self.scale < 0 or not 0 < self.ess_fraction <= 1:
            raise ValueError("Invalid scale/ESS fraction")
        if not 0 <= self.start_fraction <= self.end_fraction <= 1:
            raise ValueError("Invalid steering interval")


@dataclass
class FKResult:
    states: list
    rewards: np.ndarray
    ancestry: np.ndarray
    history: list
    model_step_calls: int
    reward_calls: int


def run_particles(sampler, reward, config, seed=0, mode="fk"):
    """Sequential importance resampling, ascending steps 1..steps.

    Intermediate schedule controls scoring/selection. Terminal weighting and
    resampling are compulsory for nonzero FK so returned particles are unweighted
    samples, even when the intermediate schedule ends earlier. Finite-particle
    approximation, not an assertion of exact independent samples.

    Child particles get distinct per-slot RNG streams. Samplers must consume
    those streams for stochastic continuation; deterministic clones stay equal.
    """
    if mode not in ("none", "fk"):
        raise ValueError("mode must be none or fk")
    n = config.num_particles
    streams = np.random.SeedSequence(seed).spawn(n + 1)
    rngs = [np.random.default_rng(s) for s in streams[:n]]
    selection_rng = np.random.default_rng(streams[-1])
    states = [sampler.initialize(rng) for rng in rngs]
    ancestry = np.arange(n)
    log_weights, phi = np.zeros(n), np.zeros(n)
    history, reward_calls = [], 0
    enabled = mode == "fk" and config.scale > 0
    for step in range(1, config.steps + 1):
        states = [sampler.advance(s, step, config.steps, rng) for s, rng in zip(states, rngs)]
        terminal = step == config.steps
        scheduled = (config.start_fraction <= step / config.steps <= config.end_fraction
                     and step % config.every == 0)
        if not terminal and not (enabled and scheduled):
            continue
        rewards = np.asarray([reward(sampler.predict(s)) for s in states], float)
        reward_calls += n
        if np.isnan(rewards).any() or np.isposinf(rewards).any():
            raise ValueError("Reward must return finite scalar or -inf for invalid particles")
        finite = rewards[np.isfinite(rewards)]
        if not len(finite):
            raise ValueError("All particles are invalid at scoring step")
        if enabled:
            log_weights, phi = difference_update(log_weights, phi, rewards, config.scale)
            weights = normalize_log_weights(log_weights)
        else:
            weights = np.full(n, 1 / n)
        ess = effective_sample_size(weights)
        resampled = enabled and (terminal or ess < config.ess_fraction * n)
        parents = systematic_resample(weights, selection_rng) if resampled else np.arange(n)
        history.append({"step": step, "terminal": terminal, "ess_before": ess,
                        "reward_mean_valid": float(finite.mean()),
                        "reward_min_valid": float(finite.min()), "reward_max_valid": float(finite.max()),
                        "invalid_count": int(n - len(finite)),
                        "constant_reward": bool(np.ptp(finite) < 1e-12),
                        "resampled": bool(resampled), "parents": parents.tolist(),
                        "root_ancestors_after": ancestry[parents].tolist(),
                        "unique_roots_after": int(len(np.unique(ancestry[parents])))})
        if resampled:
            states = [sampler.clone(states[int(p)]) for p in parents]
            ancestry, phi, rewards = ancestry[parents].copy(), phi[parents].copy(), rewards[parents].copy()
            log_weights = np.zeros(n)
        else:
            # Shift by a common constant to prevent accumulation overflow.
            log_weights -= np.max(log_weights)
    return FKResult(states, rewards, ancestry, history, n * config.steps, reward_calls)
