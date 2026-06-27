"""Determinism tests for seeded sensor noise (additive, isolated).

Sensor noise is drawn from a per-instance ``numpy.random.Generator`` seeded
deterministically, instead of the process-global ``numpy.random``. This makes a
seed-reproducible run (e.g. a judged competition) yield the identical sensor
noise for a given seed, regardless of the global RNG state or parallel/forked
execution -- mirroring the seeding the Monte Carlo layer already uses
(``stochastic_model.py`` / ``monte_carlo.py``).

The tests construct sensors directly and do not touch the existing fixtures or
inherited tests. Sensor noise is sampled on a fixed time grid (not per adaptive
solver step), so a fixed number of sequential draws is a faithful stand-in for a
flight of a given duration.
"""

import numpy as np

from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.sensors.accelerometer import Accelerometer
from rocketpy.sensors.gnss_receiver import GnssReceiver


def _accelerometer(seed):
    # Non-zero white noise + random walk so the draws actually exercise the RNG.
    return Accelerometer(
        sampling_rate=10,
        noise_density=1.0,
        noise_variance=1.0,
        random_walk_density=0.5,
        random_walk_variance=1.0,
        seed=seed,
    )


def _noise_sequence(sensor, n=16):
    return [tuple(sensor.apply_noise(Vector([0.0, 0.0, 0.0]))) for _ in range(n)]


def test_same_seed_is_reproducible():
    assert _noise_sequence(_accelerometer(42)) == _noise_sequence(_accelerometer(42))


def test_different_seeds_decorrelate():
    assert _noise_sequence(_accelerometer(1)) != _noise_sequence(_accelerometer(2))


def test_noise_independent_of_global_numpy_rng():
    # Perturbing the process-global RNG must NOT change a seeded sensor's noise.
    # This is the regression guard for the original bug (noise drawn from the
    # global ``np.random``).
    np.random.seed(0)
    first = _noise_sequence(_accelerometer(7))
    np.random.seed(999)
    _ = [np.random.random() for _ in range(1000)]
    second = _noise_sequence(_accelerometer(7))
    assert first == second


def test_seeded_sensor_does_not_consume_global_rng():
    # A seeded sensor draws only from its own generator, leaving the global RNG
    # position untouched -- so it cannot contaminate other code or parallel envs.
    np.random.seed(0)
    position_before = np.random.get_state()[2]
    _noise_sequence(_accelerometer(7))
    position_after = np.random.get_state()[2]
    assert position_before == position_after


def test_recreating_generator_from_seed_reproduces_sequence():
    # Reproducibility comes from the seed: re-creating the generator from the
    # same seed (which is what a freshly-constructed sensor does) replays the
    # same sequence. The random walk is cumulative, so zeroing it matters too.
    sensor = _accelerometer(99)
    first = _noise_sequence(sensor)
    sensor._rng = np.random.default_rng(sensor._seed)
    sensor._random_walk_drift = Vector([0.0, 0.0, 0.0])
    assert _noise_sequence(sensor) == first


def _gnss_sequence(seed, n=8):
    gnss = GnssReceiver(
        sampling_rate=1,
        position_accuracy=5.0,
        altitude_accuracy=5.0,
        velocity_accuracy=1.0,
        seed=seed,
    )
    # Minimal launch-frame state: 100 m up, identity attitude quaternion.
    state = np.array([0, 0, 100, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    measurements = []
    for _ in range(n):
        gnss.measure(0.0, u=state, relative_position=Vector([0.0, 0.0, 0.0]))
        measurements.append(gnss.measurement)
    return measurements


def test_gnss_is_seeded_and_reproducible():
    assert _gnss_sequence(5) == _gnss_sequence(5)
    assert _gnss_sequence(5) != _gnss_sequence(6)
