"""What the run is allowed to call a success, and how it comes down when it is not.

The completeness check reads the two logs back and decides whether the run can
be reported as complete. Everything it accepts is a claim about the results, so
a row it cannot read, an index it cannot trust, or a file that disagrees with
its pair has to stop the run rather than be skipped past.

The shutdown tests cover the other half: a fleet where one worker is not coming
back has to be brought down in bounded time, and the failure that started it has
to survive that.
"""

import threading
import types

import pytest

import rocketpy.simulation.monte_carlo as mc


def _runner(tmp_path, rows, outputs=None, count=2, initial=0, interrupted=False):
    """A stand-in carrying only what the completeness check reads."""
    inputs_file = tmp_path / "inputs.txt"
    outputs_file = tmp_path / "outputs.txt"
    inputs_file.write_text(rows, encoding="utf-8")
    outputs_file.write_text(rows if outputs is None else outputs, encoding="utf-8")
    return types.SimpleNamespace(
        input_file=inputs_file,
        output_file=outputs_file,
        number_of_simulations=count,
        _initial_sim_idx=initial,
        _interrupted=interrupted,
    )


def _check(runner):
    mc.MonteCarlo._MonteCarlo__check_each_index_was_recorded_once(runner)


COMPLETE = '{"index": 0}\n{"index": 1}\n'


CORRUPT = {
    "a row cut off mid-write": (COMPLETE + "{not json\n", "not readable JSON"),
    "a row that is not an object": (COMPLETE + "[]\n", "does not carry"),
    "a row with no index": (COMPLETE + '{"foo": 1}\n', "does not carry"),
    "a boolean index": ('{"index": 0}\n{"index": true}\n', "does not carry"),
    "a float index": ('{"index": 0}\n{"index": 1.0}\n', "does not carry"),
    "a negative index": (COMPLETE + '{"index": -1}\n', "does not carry"),
    "an index past the run": (COMPLETE + '{"index": 99}\n', "outside the range"),
    "the same index twice": (COMPLETE + '{"index": 1}\n', "more than once"),
    "an index never written": ('{"index": 0}\n', "never written"),
}


@pytest.mark.parametrize(
    ("rows", "expected"), list(CORRUPT.values()), ids=list(CORRUPT)
)
def test_a_corrupt_log_is_not_a_successful_run(tmp_path, rows, expected):
    """Every one of these was accepted as a complete run.

    ``True`` and ``1.0`` are the two that look harmless: both compare equal to
    ``1``, so an ``isinstance`` check or a bare dict lookup counts them as the
    index they are not. Hence ``type(index) is int``.
    """
    with pytest.raises(RuntimeError, match=expected):
        _check(_runner(tmp_path, rows))


def test_a_complete_run_is_still_accepted(tmp_path):
    """The control. Without this the table above passes on a check that
    rejects everything."""
    _check(_runner(tmp_path, COMPLETE))


def test_an_earlier_run_left_in_the_files_is_not_an_error(tmp_path):
    """``append=True`` keeps the earlier run's rows, and they are below
    ``_initial_sim_idx``. Rejecting anything outside the new range would make
    every appended run fail."""
    rows = '{"index": 0}\n{"index": 1}\n{"index": 2}\n{"index": 3}\n'
    _check(_runner(tmp_path, rows, count=4, initial=2))


def test_an_interrupted_run_still_has_to_be_readable(tmp_path):
    """Being short is allowed after Ctrl-C. Being corrupt is not: skipping the
    check entirely meant a duplicate or an unreadable row went unreported."""
    _check(_runner(tmp_path, '{"index": 0}\n', interrupted=True))

    with pytest.raises(RuntimeError, match="more than once"):
        _check(_runner(tmp_path, '{"index": 0}\n{"index": 0}\n', interrupted=True))


def test_the_two_files_have_to_agree(tmp_path):
    """A worker that wrote its inputs and stopped before its outputs. Each file
    on its own can look complete, because the row one is missing is in the
    other."""
    with pytest.raises(RuntimeError, match="disagree about which simulations"):
        _check(_runner(tmp_path, COMPLETE, outputs='{"index": 0}\n'))


class _Worker:
    """A process stub that can be told to ignore termination.

    Every call is appended to a shared ``trace`` so the order across the whole
    fleet can be asserted, not just the per-worker counts. A stub join costs no
    time, so a test that only counts calls cannot tell "signal everyone, then
    wait" from "signal one and wait for it before reaching the next".
    """

    def __init__(self, name="worker", alive=False, deaf=False, exitcode=0, trace=None):
        self.name = name
        self._alive = alive
        self._deaf = deaf
        self.exitcode = exitcode
        self.terminated = 0
        self.killed = 0
        self.joins = []
        self.trace = [] if trace is None else trace

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated += 1
        self.trace.append(("terminate", self.name))
        if not self._deaf:
            self._alive = False

    def kill(self):
        self.killed += 1
        self.trace.append(("kill", self.name))
        self._alive = False

    def join(self, timeout=None):
        self.joins.append(timeout)
        self.trace.append(("join", self.name))


class _Event:
    def __init__(self, flag=False):
        self.flag = flag

    def set(self):
        self.flag = True

    def is_set(self):
        return self.flag


def test_the_wait_gives_up_as_soon_as_a_worker_reports_an_error():
    """One worker is not coming back and another has already failed.

    Joining the fleet in order held the parent on the first worker forever, so
    the error the second had already reported was never seen and the cleanup
    after it never ran.
    """
    stuck, failed = _Worker(alive=True), _Worker(exitcode=1)

    mc._wait_for_workers([stuck, failed], _Event(flag=True))

    assert stuck.is_alive(), "the wait should return, not stop the workers itself"


def test_the_wait_is_bounded_when_it_is_given_a_deadline():
    """The interrupt path waits a short while for workers to notice the event.
    Without a deadline that wait was the second place Ctrl-C could hang."""
    stuck = _Worker(alive=True)

    mc._wait_for_workers([stuck], timeout=0.2)

    assert stuck.is_alive()


def test_the_wait_reaps_workers_that_had_already_finished():
    """A worker gone before the loop starts is never joined by it, and an
    unjoined child has no exit code, which the crash check downstream reads as
    a crash."""
    done = _Worker(alive=False)

    mc._wait_for_workers([done], _Event())

    assert done.joins, "a finished worker was never joined, so it was not reaped"


def test_every_worker_is_signalled_before_any_of_them_is_waited_on():
    """Signal the whole fleet, then wait on it.

    Terminating one and joining it before reaching the next made every worker
    wait out the grace period of the ones ahead of it in the list, so a single
    worker that ignores the signal delays the rest by that much each.
    """
    trace = []
    deaf = _Worker(name="deaf", alive=True, deaf=True, trace=trace)
    ordinary = _Worker(name="ordinary", alive=True, trace=trace)

    mc._stop_any_worker_still_running([deaf, ordinary], grace=0.01)

    first_join = next(i for i, (call, _) in enumerate(trace) if call == "join")
    assert not [c for c in trace[first_join:] if c[0] == "terminate"], (
        f"a worker was signalled only after another had been waited on: {trace}"
    )
    assert ordinary.terminated == 1, "the second worker was never signalled"
    assert deaf.killed == 1, "the worker that sat through terminate was not killed"
    assert not deaf.is_alive()


def test_a_worker_that_already_exited_is_left_alone():
    """The control: cleanup runs on every path, including the ones where
    nothing went wrong."""
    done = _Worker(alive=False)

    mc._stop_any_worker_still_running([done], grace=0.01)

    assert done.terminated == 0 and done.killed == 0


class _RecordingMutex:
    def __init__(self, fail_on_acquire=False):
        self.acquired = 0
        self.released = 0
        self._fail = fail_on_acquire
        self._lock = threading.Lock()

    def acquire(self, *args, **kwargs):
        self.acquired += 1
        if self._fail:
            raise ConnectionResetError("the manager is gone")
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self.released += 1
        return self._lock.release()


class _Boom(RuntimeError):
    """The failure under test, so it cannot be confused with an incidental one."""


class _Monitor:
    """Enough of a monitor for a worker that completes an iteration."""

    count = 0

    def print_update_status(self):
        pass


def _sim_worker(tmp_path, **overrides):
    attributes = {
        "error_file": tmp_path / "errors.txt",
        "input_file": tmp_path / "inputs.txt",
        "output_file": tmp_path / "outputs.txt",
        "_MonteCarlo__child_seed": lambda index: index,
        "_MonteCarlo__seed_simulation": lambda seed: None,
        "_MonteCarlo__run_single_simulation": object,
        "_MonteCarlo__evaluate_flight_inputs": lambda index: '{"index": 0}\n',
        "_MonteCarlo__evaluate_flight_outputs": lambda flight, index: '{"index": 0}\n',
    }
    attributes.update(overrides)
    return types.SimpleNamespace(**attributes)


def test_a_claim_that_fails_after_a_completed_run_does_not_report_that_run(
    tmp_path, monkeypatch
):
    """The state was cleared after the claim rather than before it.

    So a claim that failed on the second lap reached the handler still holding
    the row that had just been written successfully, and the error file got a
    second copy of a simulation that never failed.
    """
    claims = iter([0])

    def claim_once_then_fail(*_args, **_kwargs):
        try:
            return next(claims)
        except StopIteration:
            raise _Boom("the claim failed on the second lap") from None

    monkeypatch.setattr(mc, "_claim_next_index", claim_once_then_fail)
    reported = []
    monkeypatch.setattr(mc._SimMonitor, "reprint", staticmethod(reported.append))
    worker = _sim_worker(tmp_path)

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__sim_producer(
            worker, _Monitor(), _RecordingMutex(), _Event()
        )

    written = (tmp_path / "errors.txt").read_text()
    assert '"index": 0' not in written, (
        f"the completed simulation was written to the error file again: {written!r}"
    )
    assert "the claim failed on the second lap" in written


def test_a_mutex_that_cannot_be_taken_is_not_then_released(tmp_path, monkeypatch):
    """The normal write path released in ``finally`` whether or not it had the
    lock, so a manager that died during acquire raised a second error on the way
    out and buried the first."""
    monkeypatch.setattr(mc, "_claim_next_index", lambda *a, **k: 0)
    mutex = _RecordingMutex(fail_on_acquire=True)

    with pytest.raises(ConnectionResetError):
        mc.MonteCarlo._MonteCarlo__sim_producer(
            _sim_worker(tmp_path), _Monitor(), mutex, _Event()
        )

    assert mutex.released == 0, "released a lock it never held"


def _serial_runner(tmp_path, row=""):
    """A stand-in carrying only what ``__run_in_serial`` touches."""
    runner = types.SimpleNamespace(
        _initial_sim_idx=0,
        number_of_simulations=2,
        _interrupted=False,
        _error_file=tmp_path / "errors.txt",
        input_file=tmp_path / "inputs.txt",
        output_file=tmp_path / "outputs.txt",
        _MonteCarlo__child_seed=lambda index: index,
        _MonteCarlo__seed_simulation=lambda seed: None,
        _MonteCarlo__run_single_simulation=object,
        _MonteCarlo__evaluate_flight_inputs=lambda index: row,
        _MonteCarlo__evaluate_flight_outputs=lambda flight, index: row,
    )
    runner._MonteCarlo__keep_the_inputs_that_did_not_finish = lambda payload: (
        mc.MonteCarlo._MonteCarlo__keep_the_inputs_that_did_not_finish(runner, payload)
    )
    return runner


def test_ctrl_c_before_the_first_row_keeps_the_interrupt(tmp_path, monkeypatch):
    """Half of the fix: the payload is bound before the try.

    It was assigned inside the loop body, after the two monitor calls, so Ctrl-C
    in either of those reached the handler with it still unbound and the
    interrupt came out as an UnboundLocalError instead.
    """

    class _Monitor:
        count = 0

        def __init__(self, **_kwargs):
            pass

        def keep_simulating(self):
            raise KeyboardInterrupt("ctrl-c before the first simulation")

    monkeypatch.setattr(mc, "_SimMonitor", _Monitor)
    runner = _serial_runner(tmp_path)

    mc.MonteCarlo._MonteCarlo__run_in_serial(runner)

    assert runner._interrupted, "the interrupt was not recorded"


def test_ctrl_c_between_rows_does_not_report_the_row_that_succeeded(
    tmp_path, monkeypatch
):
    """The other half: the payload is cleared before each lap, not after.

    Binding it once before the try stops the UnboundLocalError but leaves it
    holding the last completed row, so an interrupt between two iterations
    wrote a simulation that had succeeded into the error file. Both halves are
    needed, and each one passes the other's test on its own.
    """

    class _Monitor:
        count = 0

        def __init__(self, **_kwargs):
            self.laps = 0

        def keep_simulating(self):
            self.laps += 1
            if self.laps > 1:
                raise KeyboardInterrupt("ctrl-c after the first simulation")
            return True

        def increment(self):
            return 1

        def print_update_status(self):
            pass

        def print_final_status(self):
            pass

    monkeypatch.setattr(mc, "_SimMonitor", _Monitor)
    runner = _serial_runner(tmp_path, row='{"index": 0}\n')

    mc.MonteCarlo._MonteCarlo__run_in_serial(runner)

    assert runner._interrupted
    assert (tmp_path / "inputs.txt").read_text() == '{"index": 0}\n', (
        "the simulation that completed should have been written normally"
    )
    assert (tmp_path / "errors.txt").read_text() == "", (
        "a completed simulation was written to the error file as if it failed"
    )
