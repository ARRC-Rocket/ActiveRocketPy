"""Nested StochasticRocket components are reseeded from distinct SeedSequence
children, so components that sample the same distribution (a main and a drogue
parachute, for example) do not draw identical values. Reproducible under a fixed
seed. See the seeding design in ``StochasticRocket._set_stochastic``.
"""

import ast
import inspect

import pytest

from rocketpy.stochastic import StochasticAirBrakes
from rocketpy.stochastic.stochastic_model import StochasticModel

# Captured once, before any patching, so wrapping it repeatedly in one test does
# not stack (each recorder wraps the real method, not a previous recorder).
_REAL_SET_STOCHASTIC = StochasticModel._set_stochastic


def _record_component_seeds(monkeypatch, rocket, seed):
    """Return the seeds handed to every nested component for one reseed."""
    recorded = []

    def recording(self, seed=None):
        recorded.append(seed)
        return _REAL_SET_STOCHASTIC(self, seed)

    monkeypatch.setattr(StochasticModel, "_set_stochastic", recording)
    rocket._set_stochastic(seed)
    return recorded


def test_rocket_components_receive_distinct_seeds(monkeypatch, stochastic_calisto):
    """Every nested component (body, aerodynamic surfaces, motor, rail buttons and
    the two parachutes) is reseeded from its own child, so none collide."""
    seeds = _record_component_seeds(monkeypatch, stochastic_calisto, 42)

    assert len(seeds) > 3, "expected the rocket body plus several components"
    assert len(seeds) == len(set(seeds)), (
        "components share a seed -- they would draw perfectly correlated samples"
    )


def test_rocket_component_seeds_are_reproducible(monkeypatch, stochastic_calisto):
    """The same root seed reseeds every component identically; a different root
    seed changes them."""
    first = _record_component_seeds(monkeypatch, stochastic_calisto, 42)
    again = _record_component_seeds(monkeypatch, stochastic_calisto, 42)
    different = _record_component_seeds(monkeypatch, stochastic_calisto, 43)

    assert again == first, "same seed must reproduce every component seed"
    assert different != first, "a different seed must change the component seeds"


def test_the_reseed_covers_every_collection_create_object_uses(stochastic_calisto):
    """Whatever ``create_object`` iterates has to be reseeded too.

    Checked against the source rather than against a fixture, because a
    collection that no fixture populates is exactly the one that gets missed:
    air brakes were built and sampled and never reseeded, and every seeding
    test passed because no fixture had one.
    """
    rocket = stochastic_calisto
    tree = ast.parse(inspect.getsource(type(rocket).create_object).lstrip())
    iterated = {
        node.iter.attr
        for node in ast.walk(tree)
        # Comprehensions too: this scan exists to catch a collection added
        # later, and a loop rewritten as one would slip past a For-only walk.
        if isinstance(node, (ast.For, ast.comprehension))
        and isinstance(node.iter, ast.Attribute)
        and isinstance(node.iter.value, ast.Name)
        and node.iter.value.id == "self"
        and not node.iter.attr.startswith("_")
    }
    declared = set(type(rocket)._stochastic_collections())

    assert iterated, "found no collections in create_object; the scan is broken"
    assert iterated <= declared, (
        f"create_object samples these but the reseed never reaches them: "
        f"{sorted(iterated - declared)}"
    )


def test_air_brakes_are_reseeded_like_every_other_component(
    monkeypatch, stochastic_calisto, calisto_air_brakes_clamp_on
):
    """Air brakes were in ``create_object`` and not in the reseed, so their
    samples came from wherever the Generator had been left rather than from the
    simulation index. Measured before the fix: 3 surfaces, 1 motor, 1 rail
    button and 2 parachutes reseeded, air brakes 0 of 1.
    """
    stochastic_calisto.add_air_brakes(
        calisto_air_brakes_clamp_on.air_brakes[0],
        calisto_air_brakes_clamp_on._controllers[0],
    )
    air_brake = stochastic_calisto.air_brakes[0]
    seen = []
    original = air_brake._set_stochastic
    monkeypatch.setattr(
        air_brake,
        "_set_stochastic",
        lambda seed=None: (seen.append(seed), original(seed))[1],
    )

    stochastic_calisto._set_stochastic(42)

    assert seen, "air brakes were not reseeded"
    assert seen[0] is not None


@pytest.mark.parametrize(
    "spec",
    [0.001, (0.001, "normal"), (0.0, 0.001, "normal"), [0.0005, 0.001, 0.002]],
    ids=["scalar", "tuple2", "tuple3", "list"],
)
def test_eccentricity_is_resampled_from_the_new_generator(stochastic_calisto, spec):
    """``add_cp_eccentricity`` and ``add_thrust_eccentricity`` run after
    ``__init__``, so their values never reached the dict the base class
    re-validates. Validation binds a distribution to the Generator that is live
    at the time, so the tuple kept sampling from the one the rocket was built
    with: same seed, different eccentricity, while every constructor field
    reproduced exactly.
    """
    rocket = stochastic_calisto
    rocket.add_cp_eccentricity(x=spec, y=spec)
    rocket.add_thrust_eccentricity(x=spec, y=spec)

    def sample():
        rocket._set_stochastic(777)
        drawn = next(rocket.dict_generator())
        return {k: v for k, v in drawn.items() if "eccentricity" in k}

    first = sample()

    assert len(first) == 4, f"expected four eccentricities, got {sorted(first)}"
    assert sample() == first, "the same seed drew a different eccentricity"


def test_the_air_brake_sample_follows_the_seed_not_the_call_order(
    stochastic_calisto, calisto_air_brakes_clamp_on
):
    """That the reseed reaches the air brake is only half of it.

    What matters is the value it draws: the same seed has to give the same
    sample, and a different seed a different one. Asserting only that
    ``_set_stochastic`` was called would pass over an air brake reseeded with a
    constant.
    """
    # Built here rather than taken from the fixture: wrapping an AirBrakes with
    # no arguments gives every parameter a zero standard deviation, so it draws
    # the same values under any seed and the assertions below would hold over an
    # air brake that was never reseeded at all.
    stochastic_calisto.add_air_brakes(
        StochasticAirBrakes(
            air_brakes=calisto_air_brakes_clamp_on.air_brakes[0],
            drag_coefficient_curve_factor=(1.0, 0.1),
        ),
        calisto_air_brakes_clamp_on._controllers[0],
    )
    air_brake = stochastic_calisto.air_brakes[0]

    def drawn(seed):
        stochastic_calisto._set_stochastic(seed)
        return next(air_brake.dict_generator())

    first = drawn(31337)

    assert first, "the air brake sampled nothing, so this proves nothing"
    assert drawn(31337) == first, "the same seed drew a different air brake"
    assert drawn(31338) != first, "a different seed drew the same air brake"
