"""Characterization tests for ``Flight.step_simulation()``.

The fork's stepped-simulation API (``run_simulation=False`` plus repeated
``step_simulation()`` calls) must reproduce a one-shot ``simulate()`` so that
callers driving the flight one node at a time -- e.g. the Balloon Popping
Challenge environment, which steps it every timestep -- obtain the same
trajectory. A parachute-free rocket (``flight_calisto``) is used because
parachute triggers are not yet migrated into the stepping path.

Scope: ``TestStepSimulation`` guards that *uncontrolled* stepping matches
``simulate()``. ``TestControlledStepSimulation`` guards the controlled use case
the Balloon Popping Challenge actually relies on: injecting an actuator command
between ``step_simulation()`` calls must change the trajectory.
"""

import copy

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


def _no_op_roll_logger(
    time,
    sampling_rate,
    state,
    state_history,
    observed_variables,
    roll_control,
    sensors,
    environment,
):  # pylint: disable=unused-argument
    """Controller that only *logs* the roll torque, never sets it.

    This mirrors the Balloon Popping Challenge wiring, where the controller is a
    passive logger and the command is injected externally between steps. It must
    not override the externally set ``roll_torque``.
    """
    return (time, roll_control.roll_torque)


def _controlled_twin(calisto, max_roll_torque=100.0):
    """A deep copy of ``calisto`` fitted with a no-op-logger roll actuator.

    A copy is used so the original fixture rocket stays actuator-free and can
    serve as the uncontrolled baseline within the same test. Roll control is
    chosen because its torque is added straight onto the body-axis moment
    (``M3 += roll_control.roll_torque``), giving a thrust-independent effect that
    shows up cleanly in the roll rate ``w3``. ``max_roll_torque`` is left well
    above the commanded value so the command passes through unclamped, and no
    rate limit / time constant is set so the injected value is applied verbatim.
    """
    twin = copy.deepcopy(calisto)
    twin.add_roll_control(
        controller_function=_no_op_roll_logger,
        sampling_rate=10.0,
        max_roll_torque=max_roll_torque,
    )
    return twin


def _step_with_roll(env, rocket, command, max_steps=100000):
    """Drive a stepped flight, injecting a roll command before every step.

    Mirrors the BPC ``step()`` loop, which sets
    ``rocket.roll_control.roll_torque`` ahead of each ``step_simulation()``.
    ``command`` is either a constant torque (N.m) or a callable ``command(t)``
    returning the torque to inject at the current flight time.

    ``time_overshoot=False`` is essential and is exactly what BPC uses: it turns
    the controller's sampling nodes into real solver time nodes, so each
    ``step_simulation()`` advances one small node and the command is injected
    every timestep. With the default ``time_overshoot=True`` each call would
    instead integrate a whole flight phase, injecting the command only a handful
    of times -- not the per-timestep control loop under test. The on-rail phase
    is roll-constrained, so commanding through it is harmless; the body only
    starts spinning up once the rocket leaves the rail.

    Returns the finished flight and the number of injected steps.
    """
    command_fn = command if callable(command) else (lambda t: command)
    flight = Flight(
        environment=env,
        rocket=rocket,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
        run_simulation=False,
        time_overshoot=False,
    )
    steps = 0
    while not flight._step_state["finished"]:
        rocket.roll_control.roll_torque = command_fn(flight.t)
        flight.step_simulation()
        steps += 1
        assert steps < max_steps, "stepped simulation did not terminate"
    return flight, steps


class TestControlledStepSimulation:
    """Injecting an actuator command between steps must move the trajectory.

    This is the fork's actual use case -- the one the Balloon Popping Challenge
    relies on: a passive logger controller plus a roll command set externally
    between ``step_simulation()`` calls.
    """

    # Roll torque commanded between steps, in N.m. Well inside the actuator
    # range so it is applied verbatim; large relative to the tiny roll inertia
    # so the angular response is unmistakable.
    COMMAND = 5.0
    # Flight time (s) at which the reversal test flips the command sign. Comfor-
    # tably between rail departure and the (~25 s) apogee of this rocket, and
    # early enough that the post-reversal torque decisively reverses the spin.
    REVERSAL_TIME = 10.0

    def test_injected_roll_command_changes_angular_state(
        self, calisto, example_plain_env
    ):
        rocket = _controlled_twin(calisto)
        commanded, commanded_steps = _step_with_roll(
            example_plain_env, rocket, self.COMMAND
        )
        neutral, _ = _step_with_roll(example_plain_env, rocket, 0.0)

        # The command really was injected per timestep, not once per phase: the
        # fine-grained loop runs for many steps (hundreds for this rocket).
        assert commanded_steps > 50

        # Sample the roll rate off the rail, up to just before apogee, on a grid
        # common to both flights (their nodes need not coincide).
        grid = np.linspace(2.0, min(commanded.t, neutral.t) - 0.5, 12)
        commanded_w3 = np.array([commanded.w3(t) for t in grid])
        neutral_w3 = np.array([neutral.w3(t) for t in grid])

        # Neutral command -> the actuator contributes no moment -> no roll.
        assert np.max(np.abs(neutral_w3)) < 1e-6

        # Commanded -> the body spins up, the roll rate growing while the torque
        # is sustained. Orders of magnitude above any solver noise.
        assert np.all(np.diff(commanded_w3) > 0)
        assert np.abs(commanded_w3[-1]) > 100.0

        # The two angular trajectories diverge: the injected command had a real
        # dynamical effect.
        divergence = np.max(np.abs(commanded_w3 - neutral_w3))
        assert divergence > 100.0

    def test_mid_flight_command_reversal_is_tracked(self, calisto, example_plain_env):
        # Spin one way, then -- partway up -- command the opposite torque. This
        # can only register if the command is re-read on every step: a command
        # applied once (or latched) could never make the roll rate turn around.
        rocket = _controlled_twin(calisto)

        def command(t):
            return self.COMMAND if t < self.REVERSAL_TIME else -self.COMMAND

        reversed_flight, _ = _step_with_roll(example_plain_env, rocket, command)

        grid = np.linspace(2.0, reversed_flight.t - 0.5, 20)
        w3 = np.array([reversed_flight.w3(t) for t in grid])

        # Roll rate climbs, peaks in the interior (the sign flip), then comes
        # back down through zero -- the body is now spinning the other way.
        assert w3.max() > 100.0
        assert 0 < int(np.argmax(w3)) < len(w3) - 1
        assert w3[-1] < 0

    def test_zero_command_matches_uncontrolled(self, calisto, example_plain_env):
        # Build the controlled twin before flying the baseline so both share an
        # identical, simulation-untouched starting rocket.
        rocket = _controlled_twin(calisto)

        # Uncontrolled baseline: the untouched fixture rocket, run one-shot.
        baseline = Flight(
            environment=example_plain_env,
            rocket=calisto,
            rail_length=5.2,
            inclination=85,
            heading=0,
            terminate_on_apogee=True,
        )

        # Same rocket plus a no-op actuator, stepped with a zero command.
        neutral, _ = _step_with_roll(example_plain_env, rocket, 0.0)

        grid = np.linspace(0.5, min(baseline.t, neutral.t) - 0.2, 60)

        # The actuator acts only on the roll axis, so commanding zero must leave
        # the angular state bit-for-bit identical to the uncontrolled flight:
        # the divergence in the tests above is the *command*, never the mere
        # presence of the actuator.
        for state in ("w1", "w2", "w3"):
            baseline_state = getattr(baseline, state)
            neutral_state = getattr(neutral, state)
            np.testing.assert_allclose(
                [neutral_state(t) for t in grid],
                [baseline_state(t) for t in grid],
                rtol=0,
                atol=1e-9,
            )

        # The translational trajectory also tracks the uncontrolled run. It is
        # not bit-identical: ``time_overshoot=False`` forces an LSODA restart at
        # every controller node, and those restarts accumulate sub-metre path
        # noise over the multi-kilometre flight (well within the integrator's own
        # tolerance). This is pure numerical path noise from the inert actuator,
        # not a dynamical effect -- the roll axis it acts on is left exactly
        # unchanged (asserted above) -- so it cannot be confused with a command.
        for state in ("x", "y", "z", "vx", "vy", "vz"):
            baseline_state = getattr(baseline, state)
            neutral_state = getattr(neutral, state)
            np.testing.assert_allclose(
                [neutral_state(t) for t in grid],
                [baseline_state(t) for t in grid],
                rtol=2e-3,
                atol=1e-1,
            )
