"""Synthetic stochastic geometry process, NOT a protein/flow-matching model."""
import numpy as np


class ToySampler:
    def __init__(self):
        self.center = np.array([3.0, 2.5, 2.0])
        self.anchors = {"a": np.array([0., 0., 0.]), "b": np.array([2., 0., 0.]),
                        "c": np.array([0., 2., 0.])}

    def initialize(self, rng):
        return self.center + rng.normal(0, 1.0, 3)

    def advance(self, state, step, total_steps, rng):
        sigma = 0.3 * (1 - step / total_steps) + 0.06
        return state + 0.015 * (self.center - state) + rng.normal(0, sigma, 3)

    def predict(self, state):
        return {**{k: v.copy() for k, v in self.anchors.items()}, "d": state.copy()}

    def clone(self, state):
        return state.copy()
