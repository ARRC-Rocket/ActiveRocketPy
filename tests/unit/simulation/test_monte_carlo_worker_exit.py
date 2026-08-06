"""The parent has to notice a worker that died without saying so.

``join()`` returns None however the child ended, so the shared error event was
the only signal the run had. A worker can leave without setting it: ``SystemExit``,
``os._exit``, a segfault in a native extension, a target that will not unpickle
under spawn, or the error handler of the worker itself failing. The exit status
is what separates those from a clean finish.
"""

import types
from contextlib import contextmanager

import pytest

import rocketpy.simulation.monte_carlo as mc


class _Process:
    """A worker that does not run, and reports the exit code it was given."""

    instances = []

    def __init__(self, target=None, args=(), **_kwargs):  # pylint: disable=unused-argument
        self.name = f"worker-{len(self.instances)}"
        self.exitcode = None
        self.started = False
        self.terminated = False
        self._planned_exitcode = 0
        self.instances.append(self)

    def start(self):
        self.started = True

    def join(self, *_a, **_k):
        self.exitcode = self._planned_exitcode

    def is_alive(self):
        return False

    def terminate(self):
        self.terminated = True


class _Event:
    def __init__(self):
        self.flag = False

    def set(self):
        self.flag = True

    def is_set(self):
        return self.flag


class _Monitor:
    def __init__(self, **_kwargs):
        pass

    def print_final_status(self):
        pass


@pytest.fixture
def parallel_runner(monkeypatch, tmp_path):
    """Run ``__run_in_parallel`` over stub workers and hand back the stubs."""
    _Process.instances = []
    fake_multiprocess = types.SimpleNamespace(Process=_Process)

    class _Manager:  # pylint: disable=invalid-name
        """Method names mirror the multiprocess manager API."""

        def Lock(self):  # noqa: N802
            return types.SimpleNamespace(acquire=lambda: None, release=lambda: None)

        def Event(self):  # noqa: N802
            return _Event()

        def _SimMonitor(self, **kwargs):  # noqa: N802
            return _Monitor(**kwargs)

    @contextmanager
    def fake_manager(*_a, **_k):
        yield _Manager()

    monkeypatch.setattr(mc, "_import_multiprocess", lambda: (fake_multiprocess, None))
    monkeypatch.setattr(mc, "_create_multiprocess_manager", fake_manager)

    runner = types.SimpleNamespace(
        error_file=tmp_path / "errors.txt",
        input_file=tmp_path / "inputs.txt",
        output_file=tmp_path / "outputs.txt",
        _initial_sim_idx=0,
        number_of_simulations=4,
        _interrupted=False,
        _MonteCarlo__validate_number_of_workers=lambda n: 2,
        _MonteCarlo__sim_producer=lambda *a: None,
    )
    runner.input_file.write_text("")
    runner.output_file.write_text("")
    return runner


def test_a_worker_that_crashes_without_setting_the_event_fails_the_run(
    parallel_runner,
):
    """The case the event alone cannot see."""
    original_join = _Process.join

    def crash(self, *a, **k):
        original_join(self, *a, **k)
        self.exitcode = -11  # SIGSEGV

    _Process.join = crash
    try:
        with pytest.raises(RuntimeError, match="did not exit cleanly"):
            mc.MonteCarlo._MonteCarlo__run_in_parallel(parallel_runner, n_workers=2)
    finally:
        _Process.join = original_join


def test_a_clean_run_is_not_reported_as_a_crash(parallel_runner):
    """The other half: every worker exits 0, so nothing is raised."""
    mc.MonteCarlo._MonteCarlo__run_in_parallel(parallel_runner, n_workers=2)

    assert all(p.exitcode == 0 for p in _Process.instances)


def test_a_failed_start_still_cleans_up_the_workers_already_running(
    parallel_runner, monkeypatch
):
    """The start loop is inside the try for this. A ``start()`` that fails part
    way through used to leave the ones already running with nobody to reap
    them."""
    started = []
    original_start = _Process.start

    def start_then_fail(self):
        if len(started) >= 1:
            raise OSError("cannot allocate a process")
        original_start(self)
        started.append(self)

    monkeypatch.setattr(_Process, "start", start_then_fail)
    monkeypatch.setattr(_Process, "is_alive", lambda self: not self.terminated)

    with pytest.raises(OSError):
        mc.MonteCarlo._MonteCarlo__run_in_parallel(parallel_runner, n_workers=3)

    assert started, "the fixture never started anything"
    assert all(p.terminated for p in started), "a started worker was left running"


def test_an_interrupted_run_is_not_then_reported_as_incomplete(
    parallel_runner, monkeypatch
):
    """Ctrl-C in the parent is caught and deliberately not re-raised, so
    ``simulate`` carries on to the completeness check with the run unfinished.
    The two are composed here in that order, since the check has to be able to
    tell a run the user stopped from a worker that went missing.
    """
    interrupted = []
    original_join = _Process.join

    def ctrl_c(self, *a, **k):
        # Once: the handler joins again on its way out, and that has to work.
        if not interrupted:
            interrupted.append(True)
            raise KeyboardInterrupt("user pressed ctrl-c")
        original_join(self, *a, **k)

    monkeypatch.setattr(_Process, "join", ctrl_c)

    mc.MonteCarlo._MonteCarlo__run_in_parallel(parallel_runner, n_workers=2)
    mc.MonteCarlo._MonteCarlo__check_each_index_was_recorded_once(parallel_runner)

    assert interrupted, "the run was never interrupted"
    assert parallel_runner.input_file.read_text() == "", (
        "nothing was written, so the check really was in a position to reject this"
    )
