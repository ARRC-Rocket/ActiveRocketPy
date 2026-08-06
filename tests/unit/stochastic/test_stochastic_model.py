from types import SimpleNamespace

import pytest

from rocketpy import Environment
from rocketpy.stochastic import StochasticEnvironment
from rocketpy.stochastic.stochastic_model import StochasticModel


def _sampled_option(model):
    """Return the value ``dict_generator`` picks for the ``options`` attribute."""
    return next(model.dict_generator())["options"]


def test_list_attribute_sampling_is_reproducible_under_seed():
    """A list-valued stochastic attribute is drawn through the model's own seeded
    numpy generator, so a fixed seed reproduces the choice. It used to be drawn
    with the stdlib ``random.choice`` (an unseeded global instance), which
    ``random_seed`` could not govern. Heterogeneous entries (paths, callables,
    lists) are returned unchanged rather than coerced to a numpy dtype the way
    ``numpy.random.choice`` would.
    """
    options = ["/motor/a.eng", "/motor/b.eng", (lambda t: t), [1, 2, 3]]
    model = StochasticModel(obj=SimpleNamespace(), options=options)

    model._set_stochastic(42)
    first = _sampled_option(model)
    model._set_stochastic(42)
    assert _sampled_option(model) == first, "same seed must reproduce the choice"
    assert any(first is option for option in options), "object returned unchanged"

    chosen_ids = set()
    for seed in range(16):
        model._set_stochastic(seed)
        chosen_ids.add(id(_sampled_option(model)))
    assert len(chosen_ids) > 1, "different seeds must be able to pick differently"


@pytest.mark.parametrize(
    "fixture_name",
    [
        "stochastic_rail_buttons",
        "stochastic_main_parachute",
        "stochastic_environment",
        "stochastic_environment_custom_sampler",
        "stochastic_tail",
        "stochastic_calisto",
    ],
)
def test_visualize_attributes(request, fixture_name):
    """Tests the visualize_attributes method of the StochasticModel class. It
    must run without breaking and return the formatted report string (which is
    also printed), so the report is never silently lost.
    """
    fixture = request.getfixturevalue(fixture_name)
    report = fixture.visualize_attributes()
    assert isinstance(report, str)
    assert report


def _effective_wind_x(environment):
    """The wind the Environment would actually fly with."""
    wind = environment.wind_velocity_x
    return float(wind(0)) if callable(wind) else float(wind)


def test_reseeding_does_not_take_the_last_run_as_the_next_nominal():
    """Reseeding with the same seed has to give the same inputs.

    ``StochasticEnvironment.create_object`` writes the randomised value back
    onto the Environment rather than building a copy, so re-reading the nominal
    from it on the next reseed compounded: 10 -> 8.576 -> 7.355 -> 6.308, each
    one the last multiplied by the same factor again.
    """
    environment = Environment()
    environment.set_atmospheric_model(type="standard_atmosphere")
    environment.wind_velocity_x = 10.0
    stochastic = StochasticEnvironment(
        environment=environment, wind_velocity_x_factor=(1.0, 0.1)
    )

    winds = []
    for _ in range(4):
        stochastic._set_stochastic(12345)
        winds.append(_effective_wind_x(stochastic.create_object()))

    assert len(set(winds)) == 1, f"the same seed drifted across reseeds: {winds}"


def test_a_simulation_index_does_not_depend_on_the_indices_before_it():
    """What the per-index seeding claims: index i gets the same inputs however
    it is reached. Running 0, 1, 2 in order has to match running 2 on its own,
    which is what a worker that happens to pick up index 2 first would do.
    """

    def wind_for(seeds):
        environment = Environment()
        environment.set_atmospheric_model(type="standard_atmosphere")
        environment.wind_velocity_x = 10.0
        stochastic = StochasticEnvironment(
            environment=environment, wind_velocity_x_factor=(1.0, 0.1)
        )
        wind = None
        for seed in seeds:
            stochastic._set_stochastic(seed)
            wind = _effective_wind_x(stochastic.create_object())
        return wind

    assert wind_for([101, 102, 103]) == wind_for([103])


def test_a_scalar_nominal_does_not_drift_across_reseeds():
    """Not only the factors. ``_validate_scalar`` and the ``(std, "distribution")``
    tuple both take their nominal from the object, and ``create_object`` writes
    the drawn value back onto that same object, so a plain scalar spec drifts
    the same way a factor compounds.
    """
    environment = Environment()
    environment.set_atmospheric_model(type="standard_atmosphere")
    stochastic = StochasticEnvironment(environment=environment, elevation=100.0)

    elevations = []
    for _ in range(4):
        stochastic._set_stochastic(2024)
        elevations.append(float(stochastic.create_object().elevation))

    assert len(set(elevations)) == 1, f"the nominal elevation drifted: {elevations}"
