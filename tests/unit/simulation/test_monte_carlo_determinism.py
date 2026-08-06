"""Unit tests for the Monte Carlo seeding helpers.

``MonteCarlo.simulate(random_seed=...)`` makes the sampled inputs reproducible by
turning the run's root seed into one independent child stream per simulation
index. Four small helpers do the work:

* ``__root_seed_sequence`` normalizes the ``random_seed`` argument (an int, a
  sequence of ints, a ``SeedSequence`` or None) into a fresh ``SeedSequence``;
* ``__child_seed`` derives the child seed for one simulation index in O(1) by
  extending the captured root ``spawn_key`` -- bit-identical to
  ``root.spawn(n)[index]`` but without materializing the whole spawned list, so
  a worker can rebuild any index from the small root state alone;
* ``_seed_sequence_to_int`` collapses a child into a 128-bit ``int`` (the seed
  type a documented ``CustomSampler.reset_seed`` accepts);
* ``__seed_simulation`` splits one per-index child seed three ways so the
  environment, rocket and flight draw from independent streams.

These tests exercise the helpers directly, with no fixtures and no simulation, so
they stay fast. The end-to-end reproducibility of ``simulate`` (serial and across
workers) is covered by ``tests/integration/simulation/test_monte_carlo_determinism``.

Reaching a name-mangled member is an established pattern in this suite (see
``tests/unit/test_sensitivity.py`` and ``tests/unit/environment/test_environment.py``);
it lets the seeding invariants be asserted without running a Monte Carlo.
"""

import random as stdlib_random
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy.simulation import MonteCarlo
from rocketpy.simulation.monte_carlo import (
    _SimMonitor,
    _claim_next_index,
    _seed_sequence_to_int,
)

_root_seed_sequence = MonteCarlo._MonteCarlo__root_seed_sequence
_child_seed = MonteCarlo._MonteCarlo__child_seed
_seed_simulation = MonteCarlo._MonteCarlo__seed_simulation


def _entropy(seed_sequence, n=4):
    """A stable, comparable fingerprint of a ``SeedSequence``'s stream."""
    return tuple(int(x) for x in seed_sequence.generate_state(n))


def _plan(root):
    """A stand-in ``self`` carrying only the root state ``__child_seed`` reads."""
    return SimpleNamespace(
        _MonteCarlo__root_state=(
            root.entropy,
            root.spawn_key,
            root.pool_size,
            root.n_children_spawned,
        )
    )


def _advanced_root(seed, already_spawned):
    """A root whose own child counter has advanced (n_children_spawned != 0),
    the state a user's already-spawned SeedSequence would arrive in."""
    root = np.random.SeedSequence(seed)
    root.spawn(already_spawned)
    return root


# --------------------------------------------------------------------------- #
# __root_seed_sequence: normalizing the flexible seed argument                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "make_seed",
    [
        pytest.param(lambda: 12345, id="int"),
        pytest.param(lambda: np.int64(12345), id="numpy-int"),
        pytest.param(lambda: [1, 2, 3], id="sequence"),
        pytest.param(lambda: np.random.SeedSequence(12345), id="seedsequence"),
    ],
)
def test_root_seed_sequence_accepts_seed_like_values(make_seed):
    """An int, a numpy integer, a sequence of ints and a SeedSequence are all
    accepted, normalize to a SeedSequence, and are reproducible."""
    root = _root_seed_sequence(make_seed())
    assert isinstance(root, np.random.SeedSequence)
    assert _entropy(root) == _entropy(_root_seed_sequence(make_seed()))


def test_root_seed_sequence_none_draws_fresh_entropy():
    """None yields a SeedSequence seeded from fresh OS entropy (not reproducible)."""
    root = _root_seed_sequence(None)
    assert isinstance(root, np.random.SeedSequence)
    assert root.entropy is not None


@pytest.mark.parametrize(
    "make_generator",
    [
        pytest.param(lambda: np.random.default_rng(999), id="generator"),
        pytest.param(lambda: np.random.PCG64(999), id="bitgenerator"),
    ],
)
def test_root_seed_sequence_rejects_stateful_generators(make_generator):
    """A Generator/BitGenerator is a stateful RNG, not a seed value, so it is
    rejected instead of being reduced to its underlying SeedSequence."""
    with pytest.raises(TypeError, match="SeedSequence"):
        _root_seed_sequence(make_generator())


def test_root_seed_sequence_copies_full_state_without_mutating_caller():
    """A supplied SeedSequence is copied from its FULL state -- entropy, spawn_key,
    pool_size and n_children_spawned -- not just its entropy, and the caller object
    is not mutated. Asserting on ``.state`` is what gives this teeth: an
    entropy-only copy would silently drop spawn_key/n_children_spawned (making a
    spawned-child seed collide with its parent) and fail the state comparison."""
    source = np.random.SeedSequence(2024).spawn(3)[2]  # non-empty spawn_key
    source.spawn(5)  # advance its own child counter, so it is not 0
    assert source.spawn_key == (2,)
    assert source.n_children_spawned == 5

    state_before = dict(source.state)
    clone = _root_seed_sequence(source)

    assert clone is not source, "must return a copy, not the caller"
    assert clone.state == state_before, "copy must preserve the full seed state"
    assert source.state == state_before, "caller must not be mutated"
    # The copy reproduces exactly what an independent full-state rebuild produces.
    rebuilt = np.random.SeedSequence(**state_before)
    assert [_entropy(c) for c in clone.spawn(3)] == [
        _entropy(c) for c in rebuilt.spawn(3)
    ]


# --------------------------------------------------------------------------- #
# __child_seed: O(1) per-index derivation, bit-identical to spawn(n)[index]    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "make_root",
    [
        pytest.param(lambda: np.random.SeedSequence(2024), id="int-root"),
        pytest.param(lambda: np.random.SeedSequence([7, 8, 9]), id="sequence-root"),
        pytest.param(
            lambda: np.random.SeedSequence(2024).spawn(3)[2], id="spawned-root"
        ),
        pytest.param(lambda: _advanced_root(99, 4), id="advanced-counter-root"),
    ],
)
def test_child_seed_matches_spawn_bit_for_bit(make_root):
    """Deriving index i by extending the root spawn_key equals ``root.spawn(n)[i]``
    exactly, so the O(1) derivation changes no sampled inputs versus a full spawn.
    A fresh, identical root is built on each side so neither run mutates the other.
    The ``advanced-counter`` root (n_children_spawned != 0) covers a user passing a
    SeedSequence they have already spawned from: the base offset must equal
    n_children_spawned or the derived index would collide with those children.
    """
    n = 6
    derived = [_entropy(_child_seed(_plan(make_root()), i)) for i in range(n)]
    spawned = [_entropy(child) for child in make_root().spawn(n)]
    assert derived == spawned


def test_child_seed_is_worker_order_independent():
    """Any index maps to the same child regardless of the order indices are asked
    for -- the property that makes a run invariant to worker scheduling."""
    plan = _plan(np.random.SeedSequence(2024))
    forward = {i: _entropy(_child_seed(plan, i)) for i in range(5)}
    backward = {i: _entropy(_child_seed(plan, i)) for i in reversed(range(5))}
    assert forward == backward


def test_child_seed_supports_indices_beyond_32_bits():
    """A simulation index past 2**32 is not truncated: it derives a distinct child
    from its neighbour and matches the direct spawn_key construction for it."""
    root = np.random.SeedSequence(11)
    plan = _plan(root)
    big = 2**32 + 5
    assert _entropy(_child_seed(plan, big)) != _entropy(_child_seed(plan, big + 1))
    expected = np.random.SeedSequence(
        entropy=root.entropy, spawn_key=(big,), pool_size=root.pool_size
    )
    assert _entropy(_child_seed(plan, big)) == _entropy(expected)


# --------------------------------------------------------------------------- #
# _seed_sequence_to_int: 128-bit int seed for the samplers                     #
# --------------------------------------------------------------------------- #


def test_seed_sequence_to_int_is_deterministic_128_bit_int():
    def child():
        return np.random.SeedSequence(42).spawn(1)[0]

    seed = _seed_sequence_to_int(child())
    assert isinstance(seed, int)
    assert 0 <= seed < 2**128
    assert seed == _seed_sequence_to_int(child()), "must be deterministic"


def test_seed_sequence_to_int_uses_all_128_bits():
    """The int combines all four uint32 words, not a single 32-bit word, so it
    keeps the full entropy pool rather than collapsing collision risk to n**2 /
    2**32. A single-word reduction would compare unequal here."""
    ss = np.random.SeedSequence(42).spawn(1)[0]
    one_word = int(np.random.SeedSequence(42).spawn(1)[0].generate_state(1)[0])
    assert _seed_sequence_to_int(ss) != one_word
    assert _seed_sequence_to_int(ss).bit_length() > 32


def test_seed_int_is_accepted_by_the_modern_rng_apis():
    """The 128-bit int a sampler receives works with random.Random and
    numpy.random.default_rng -- the paths a CustomSampler uses. Passing a
    SeedSequence there instead is unsafe: from Python 3.11 random.Random rejects
    it with a TypeError, and before 3.11 it is silently hashed rather than used as
    entropy. Either way an int is the right thing to hand a sampler."""
    seed = _seed_sequence_to_int(np.random.SeedSequence(1).spawn(1)[0])
    assert isinstance(stdlib_random.Random(seed).random(), float)
    assert np.random.default_rng(seed).random() is not None
    if sys.version_info >= (3, 11):
        with pytest.raises(TypeError):
            stdlib_random.Random(np.random.SeedSequence(1))


# --------------------------------------------------------------------------- #
# __seed_simulation: splitting one child seed across the three models          #
# --------------------------------------------------------------------------- #


class _RecordingModel:
    """Stand-in stochastic model that records the seeds it is handed."""

    def __init__(self):
        self.seeds = []

    def _set_stochastic(self, seed=None):
        self.seeds.append(seed)


def _split_seeds(child_seed):
    """Run ``__seed_simulation`` against recording models; return the three seeds."""
    models = SimpleNamespace(
        environment=_RecordingModel(),
        rocket=_RecordingModel(),
        flight=_RecordingModel(),
    )
    _seed_simulation(models, child_seed)
    return models.environment.seeds, models.rocket.seeds, models.flight.seeds


def test_seed_simulation_hands_each_model_a_distinct_128_bit_int():
    """The per-index child seed is split three ways, and each model receives a
    plain 128-bit int (not a SeedSequence) from an independent stream."""
    env_seeds, rocket_seeds, flight_seeds = _split_seeds(np.random.SeedSequence(2024))
    assert [len(env_seeds), len(rocket_seeds), len(flight_seeds)] == [1, 1, 1]
    seeds = [env_seeds[0], rocket_seeds[0], flight_seeds[0]]
    assert all(isinstance(s, int) and 0 <= s < 2**128 for s in seeds)
    assert len(set(seeds)) == 3, "env/rocket/flight must be decorrelated"


def test_seed_simulation_is_deterministic_per_child():
    """A given child seed reseeds the three models identically every time."""

    def split(child):
        env, rocket, flight = _split_seeds(child)
        return [env[0], rocket[0], flight[0]]

    assert split(np.random.SeedSequence(2024)) == split(np.random.SeedSequence(2024))


# --------------------------------------------------------------------------- #
# _claim_next_index: atomic hand-out of the next simulation index             #
# --------------------------------------------------------------------------- #


def test_claim_next_index_hands_out_each_index_once_under_contention():
    """Holding the mutex across keep_simulating() and increment() must hand out
    each index exactly once, even when every worker reaches the claim together.

    A barrier releases all workers at once and a widened check-to-increment
    window would let an unlocked claim run several workers past the count < n
    check before any increments; the lock is what keeps the result to exactly
    n_simulations indices (0..n-1, none repeated) and the counter from
    overshooting.
    """
    n_simulations = 5
    n_workers = 8
    monitor = _SimMonitor(initial_count=0, n_simulations=n_simulations, start_time=0.0)

    # Widen the window between the check and the increment so that, without the
    # lock, several workers could pass count < n before any of them increments.
    real_keep_simulating = monitor.keep_simulating

    def slow_keep_simulating():
        result = real_keep_simulating()
        time.sleep(0.02)
        return result

    monitor.keep_simulating = slow_keep_simulating

    mutex = threading.Lock()
    barrier = threading.Barrier(n_workers)
    claimed = []
    claimed_lock = threading.Lock()

    def worker():
        barrier.wait()
        while True:
            index = _claim_next_index(monitor, mutex)
            if index is None:
                break
            with claimed_lock:
                claimed.append(index)

    workers = [threading.Thread(target=worker) for _ in range(n_workers)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()

    assert sorted(claimed) == list(range(n_simulations))
    assert monitor.count == n_simulations
