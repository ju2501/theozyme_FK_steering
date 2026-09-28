import unittest
import numpy as np
from steer.particles.resample import normalize_log_weights, effective_sample_size, systematic_resample
from steer.particles.potentials import difference_update
from steer.particles.fk_steering import FKConfig, run_particles
from steer.hooks.toy import ToySampler


class FKTests(unittest.TestCase):
    def test_extreme_log_weights(self):
        np.testing.assert_allclose(normalize_log_weights([10000, 9999]), [0.73105858, 0.26894142])

    def test_uniform_ess(self):
        self.assertAlmostEqual(effective_sample_size([1, 1, 1, 1]), 4)

    def test_collapsed_ess(self):
        self.assertAlmostEqual(effective_sample_size([0, 0, 1, 0]), 1)

    def test_large_finite_weights(self):
        self.assertAlmostEqual(effective_sample_size([1e308, 1e308]), 2)

    def test_potential_overflow_explicit(self):
        with self.assertRaisesRegex(ValueError, 'overflow'):
            difference_update([0], [0], [-1e308], 100)

    def test_invalid_weights(self):
        for values in ([], [-np.inf, -np.inf], [np.nan, 0], [np.inf, 0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normalize_log_weights(values)

    def test_resampling_frequencies(self):
        rng = np.random.default_rng(321)
        weights = np.array([.1, .2, .3, .4])
        draws = np.concatenate([systematic_resample(weights, rng) for _ in range(3000)])
        np.testing.assert_allclose(np.bincount(draws, minlength=4) / len(draws), weights, atol=.01)

    def test_zero_weight_never_selected(self):
        np.testing.assert_array_equal(systematic_resample([0, 0, 1, 0], np.random.default_rng(1)), [2]*4)

    def test_skipped_resampling_accumulates_weights(self):
        w, phi = difference_update(np.zeros(2), np.zeros(2), [-2, -1], 2)
        w, phi = difference_update(w, phi, [-1, -3], 2)
        np.testing.assert_allclose(w, [-2, -6])
        np.testing.assert_allclose(normalize_log_weights(w), normalize_log_weights([-2, -6]))

    def test_invalid_lineage_not_resurrected(self):
        w, phi = difference_update(np.zeros(2), np.zeros(2), [-np.inf, -1], 1)
        w, phi = difference_update(w, phi, [0, -2], 1)
        self.assertTrue(np.isneginf(w[0]))
        self.assertFalse(np.isnan(w).any())

    def test_bad_config(self):
        for kwargs in ({'num_particles': 0}, {'steps': 1.5}, {'scale': -1},
                       {'ess_fraction': 0}, {'start_fraction': .9, 'end_fraction': .1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FKConfig(**kwargs)

    def test_zero_scale_matches_baseline_exactly(self):
        sampler, cfg = ToySampler(), FKConfig(num_particles=8, steps=12, scale=0)
        reward = lambda x: -float(np.sum(x['d']**2))
        a = run_particles(sampler, reward, cfg, 123, 'none')
        b = run_particles(sampler, reward, cfg, 123, 'fk')
        np.testing.assert_array_equal(a.states, b.states)
        np.testing.assert_array_equal(a.rewards, b.rewards)
        self.assertFalse(any(h['resampled'] for h in b.history))

    def test_single_particle_matches_baseline(self):
        sampler, cfg = ToySampler(), FKConfig(num_particles=1, steps=8)
        reward = lambda x: -float(np.sum(x['d']**2))
        a = run_particles(sampler, reward, cfg, 4, 'none')
        b = run_particles(sampler, reward, cfg, 4, 'fk')
        np.testing.assert_array_equal(a.states, b.states)

    def test_reproducible(self):
        sampler, cfg = ToySampler(), FKConfig(num_particles=6, steps=8, every=2)
        reward = lambda x: -float(np.sum(x['d']**2))
        a = run_particles(sampler, reward, cfg, 123)
        b = run_particles(sampler, reward, cfg, 123)
        np.testing.assert_array_equal(a.states, b.states)
        self.assertEqual(a.history, b.history)

    def test_clone_has_independent_memory(self):
        sampler = ToySampler()
        parent = sampler.initialize(np.random.default_rng(5))
        a, b = sampler.clone(parent), sampler.clone(parent)
        a[0] = 999
        self.assertNotEqual(b[0], a[0])
        self.assertNotEqual(parent[0], a[0])

    def test_stochastic_clones_diverge(self):
        sampler = ToySampler()
        parent = sampler.initialize(np.random.default_rng(1))
        a = sampler.advance(sampler.clone(parent), 1, 10, np.random.default_rng(2))
        b = sampler.advance(sampler.clone(parent), 1, 10, np.random.default_rng(3))
        self.assertFalse(np.array_equal(a, b))

    def test_terminal_correction_outside_schedule(self):
        sampler = ToySampler()
        cfg = FKConfig(num_particles=8, steps=5, start_fraction=0, end_fraction=0)
        result = run_particles(sampler, lambda x: -float(np.sum(x['d']**2)), cfg, 5)
        self.assertEqual(len(result.history), 1)
        self.assertTrue(result.history[-1]['terminal'])
        self.assertTrue(result.history[-1]['resampled'])
        self.assertEqual(result.model_step_calls, 40)

    def test_constant_reward_reported(self):
        result = run_particles(ToySampler(), lambda x: 1, FKConfig(steps=4, every=1), 9)
        self.assertTrue(all(h['constant_reward'] for h in result.history))

    def test_all_invalid_is_error(self):
        with self.assertRaisesRegex(ValueError, 'All particles'):
            run_particles(ToySampler(), lambda x: -np.inf, FKConfig(steps=1), 9)

    def test_known_terminal_tilt(self):
        class Binary:
            def initialize(self, rng): return int(rng.integers(2))
            def advance(self, state, step, total_steps, rng): return state
            def predict(self, state): return state
            def clone(self, state): return state
        # p(1)=1/2, exp(reward(1))=3 -> target p(1)=3/4.
        cfg = FKConfig(num_particles=4000, steps=2, scale=1, every=1, ess_fraction=.01,
                       start_fraction=0, end_fraction=1)
        result = run_particles(Binary(), lambda x: x * np.log(3), cfg, 718)
        self.assertAlmostEqual(np.mean(result.states), .75, delta=.03)
