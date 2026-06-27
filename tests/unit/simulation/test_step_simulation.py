"""Characterization tests for ``Flight.step_simulation()``.

The fork's stepped-simulation API (``run_simulation=False`` plus repeated
``step_simulation()`` calls) must reproduce a one-shot ``simulate()`` so that
callers driving the flight one node at a time -- e.g. the Balloon Popping
Challenge environment, which steps it every timestep -- obtain the same
trajectory. A parachute-free rocket (``flight_calisto``) is used because
parachute triggers are not yet migrated into the stepping path.

Scope: this guards that *uncontrolled* stepping matches ``simulate()``. Stepping
with live controller/actuator updates between nodes is a separate concern and is
not covered here.
"""

import numpy as np

from rocketpy import Flight


def _stepped_twin(reference_flight):
    """A non-simulated ``Flight`` twin of ``reference_flight``, to step by hand.

    Launch parameters are read back from the reference so the twin cannot drift
    from it.
    """
    return Flight(
        environment=reference_flight.env,
        rocket=reference_flight.rocket,
        rail_length=reference_flight.rail_length,
        inclination=reference_flight.inclination,
        heading=reference_flight.heading,
        terminate_on_apogee=reference_flight.terminate_on_apogee,
        run_simulation=False,
    )


def _run_stepped(flight, max_steps=100000):
    """Drive ``step_simulation`` to completion.

    Returns the number of calls made and the set of phase indices visited.
    """
    steps = 0
    phases_seen = {flight._step_state["phase_index"]}
    while not flight._step_state["finished"]:
        flight.step_simulation()
        phases_seen.add(flight._step_state["phase_index"])
        steps += 1
        assert steps < max_steps, "stepped simulation did not terminate"
    return steps, phases_seen


class TestStepSimulation:
    """Stepping must match a one-shot ``simulate()`` and finalise correctly."""

    def test_initial_state_is_unfinished_at_first_phase(self, flight_calisto):
        stepped = _stepped_twin(flight_calisto)
        assert stepped._step_state["finished"] is False
        assert stepped._step_state["phase_index"] == 0
        assert stepped._step_state["node_index"] == 0

    def test_stepping_visits_multiple_phases_then_finishes(self, flight_calisto):
        stepped = _stepped_twin(flight_calisto)
        _, phases_seen = _run_stepped(stepped)
        assert stepped._step_state["finished"] is True
        assert len(phases_seen) > 1  # at least a rail phase and a flight phase

    def test_stepped_trajectory_matches_simulate(self, flight_calisto):
        stepped = _stepped_twin(flight_calisto)
        _run_stepped(stepped)
        # Stepping replays the same solver nodes as simulate(). A tight tolerance
        # (rather than exact equality) keeps the guard robust to last-bit noise in
        # the LSODA Fortran solver across platforms, while still catching any real
        # trajectory divergence.
        np.testing.assert_allclose(stepped.t, flight_calisto.t, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(
            np.array(stepped.solution),
            np.array(flight_calisto.solution),
            rtol=1e-8,
            atol=1e-10,
        )

    def test_post_process_artifacts_exist_after_stepping(self, flight_calisto):
        stepped = _stepped_twin(flight_calisto)
        _run_stepped(stepped)
        # post_process_simulation() and initialize_prints_plots() fire on finish.
        assert stepped.t_final == stepped.t
        assert stepped.prints is not None
        assert stepped.plots is not None

    def test_stepping_after_finished_is_a_noop(self, flight_calisto):
        stepped = _stepped_twin(flight_calisto)
        _run_stepped(stepped)
        t_final, y_final = stepped.t, np.array(stepped.y_sol)
        stepped.step_simulation()  # already finished -> must return immediately
        assert stepped.t == t_final
        np.testing.assert_array_equal(stepped.y_sol, y_final)
