"""What a worker does when something fails part way through.

A worker that dies has to leave three things true: the failure that started it
is the one that escapes, the shared mutex is not still held, and the error event
is set. Before this, a failure in the claim itself broke all three at once --
``sim_idx`` and ``inputs_json`` were only bound inside the loop, so the handler
raised ``UnboundLocalError`` over the real error while holding the mutex, and
``error_event.set()`` sat after the write that never ran.
"""

import json
import threading
import types

import pytest

import rocketpy.simulation.monte_carlo as mc


class _RecordingMutex:
    """A real lock that counts acquires and releases."""

    def __init__(self):
        self.acquired = 0
        self.released = 0
        self._lock = threading.Lock()

    def acquire(self, *args, **kwargs):
        self.acquired += 1
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self.released += 1
        return self._lock.release()


class _Event:
    def __init__(self):
        self.flag = False

    def set(self):
        self.flag = True

    def is_set(self):
        return self.flag


class _Boom(RuntimeError):
    """The failure under test, so it cannot be confused with an incidental one."""


def _worker(tmp_path, **overrides):
    """A stand-in carrying only the attributes ``__sim_producer`` touches."""
    attributes = {
        "error_file": tmp_path / "errors.txt",
        "input_file": tmp_path / "inputs.txt",
        "output_file": tmp_path / "outputs.txt",
        "_MonteCarlo__child_seed": lambda index: index,
        "_MonteCarlo__seed_simulation": lambda seed: None,
        "_MonteCarlo__run_single_simulation": object,
        "_MonteCarlo__evaluate_flight_inputs": lambda index: "{}\n",
        "_MonteCarlo__evaluate_flight_outputs": lambda flight, index: "{}\n",
    }
    attributes.update(overrides)
    return types.SimpleNamespace(**attributes)


def _raise(*_args, **_kwargs):
    raise _Boom("injected")


@pytest.mark.parametrize(
    "stage",
    ["claim", "reseed", "flight", "inputs", "outputs"],
    ids=["claim", "reseed", "flight", "input_eval", "output_eval"],
)
def test_a_failure_anywhere_keeps_the_cause_and_frees_the_mutex(
    tmp_path, monkeypatch, stage
):
    """Whichever stage fails, the same three things have to hold."""
    indices = iter([0, None])
    monkeypatch.setattr(mc, "_claim_next_index", lambda *a, **k: next(indices))
    overrides = {}
    if stage == "claim":
        monkeypatch.setattr(mc, "_claim_next_index", _raise)
    elif stage == "reseed":
        overrides["_MonteCarlo__seed_simulation"] = _raise
    elif stage == "flight":
        overrides["_MonteCarlo__run_single_simulation"] = _raise
    elif stage == "inputs":
        overrides["_MonteCarlo__evaluate_flight_inputs"] = _raise
    elif stage == "outputs":
        overrides["_MonteCarlo__evaluate_flight_outputs"] = _raise

    mutex, event = _RecordingMutex(), _Event()

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__sim_producer(
            _worker(tmp_path, **overrides), object(), mutex, event
        )

    assert mutex.acquired == mutex.released, "the mutex was left held"
    assert event.flag, "the parent was never told an error happened"


def test_the_cause_survives_even_when_the_error_report_also_fails(
    tmp_path, monkeypatch
):
    """Reporting is best effort. If the error file is unwritable too, the
    failure that started it is still what comes out, and the mutex is still
    released."""
    monkeypatch.setattr(mc, "_claim_next_index", lambda *a, **k: 0)
    monkeypatch.setattr(
        "builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("no disk"))
    )
    worker = _worker(tmp_path, _MonteCarlo__run_single_simulation=_raise)
    mutex, event = _RecordingMutex(), _Event()

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__sim_producer(worker, object(), mutex, event)

    assert mutex.acquired == mutex.released, "the mutex was left held"
    assert event.flag


def test_the_failure_is_still_reported_when_the_claim_itself_failed(
    tmp_path, monkeypatch
):
    """The report has to survive a failure before the loop body ran.

    ``sim_idx`` and ``inputs_json`` are bound before the try for this reason.
    Left to the loop, the handler raised ``UnboundLocalError`` at its first
    write, so nothing was written and nothing was printed: the run ended with
    no record of what went wrong.
    """
    monkeypatch.setattr(mc, "_claim_next_index", _raise)
    reported = []
    monkeypatch.setattr(mc._SimMonitor, "reprint", staticmethod(reported.append))
    mutex, event = _RecordingMutex(), _Event()

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__sim_producer(
            _worker(tmp_path), object(), mutex, event
        )

    assert reported, "the worker died without reporting anything"
    assert "injected" in reported[0], f"the report does not name the cause: {reported}"


def test_an_interrupt_while_reporting_does_not_leave_the_mutex_held(
    tmp_path, monkeypatch
):
    """``except Exception`` does not catch ``KeyboardInterrupt``, so the release
    has to be in a ``finally``. Ctrl-C between the acquire and the release would
    otherwise leave every other worker blocked on it for good."""
    monkeypatch.setattr(mc, "_claim_next_index", lambda *a, **k: 0)

    def interrupt(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(mc._SimMonitor, "reprint", staticmethod(interrupt))
    worker = _worker(tmp_path, _MonteCarlo__run_single_simulation=_raise)
    mutex, event = _RecordingMutex(), _Event()

    with pytest.raises(KeyboardInterrupt):
        mc.MonteCarlo._MonteCarlo__sim_producer(worker, object(), mutex, event)

    assert mutex.acquired == mutex.released, "the mutex was left held on interrupt"


def test_a_failure_before_the_inputs_exist_still_leaves_a_readable_record(
    tmp_path, monkeypatch
):
    """The run tells the user to check the error file, so it has to say
    something. A failure in the claim has no inputs to write, and the file was
    left empty while the traceback went only to a worker's stdout, which under
    ``spawn`` on Windows the user may never see.

    It has to stay a JSON line: ``_read_log_file`` parses this file with
    ``json.loads`` per line, so free text would make the whole log unreadable.
    """
    monkeypatch.setattr(mc, "_claim_next_index", _raise)
    worker = _worker(tmp_path)
    mutex, event = _RecordingMutex(), _Event()

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__sim_producer(worker, object(), mutex, event)

    lines = [
        line
        for line in worker.error_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert lines, "the error file was left empty"
    record = json.loads(lines[0])
    assert record["index"] is None, "an early failure has no simulation index"
    assert "injected" in record["error"], "the record does not carry the cause"


@pytest.mark.parametrize("failing_file", ["input_file", "output_file"])
def test_a_failed_write_is_reported_like_any_other_failure(
    tmp_path, monkeypatch, failing_file
):
    """A disk that fills up part way through is a failure like any other: the
    cause has to escape, the mutex has to come back, and the event has to be
    set. These two writes sit inside the loop's own mutex block rather than the
    handler, so they are worth exercising separately from the stages above.
    """
    monkeypatch.setattr(mc, "_claim_next_index", lambda *a, **k: 0)
    worker = _worker(tmp_path)
    blocked = str(getattr(worker, failing_file))
    real_open = open

    def selective_open(path, *args, **kwargs):
        if str(path) == blocked:
            raise OSError("no space left on device")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", selective_open)
    mutex, event = _RecordingMutex(), _Event()

    with pytest.raises(OSError, match="no space left"):
        mc.MonteCarlo._MonteCarlo__sim_producer(worker, object(), mutex, event)

    assert mutex.acquired == mutex.released, "the mutex was left held"
    assert event.flag


def test_an_unreachable_error_event_does_not_replace_the_failure_it_reports(
    tmp_path, monkeypatch
):
    """The event is a manager proxy, so notifying can fail on its own.

    It is set first and every other report is guarded, which left this one
    statement able to do the thing the guards exist to prevent: raise over the
    failure being reported, so the parent sees a connection error instead.
    """

    class _UnreachableEvent:
        def set(self):
            raise ConnectionResetError("the manager is gone")

        def is_set(self):
            raise ConnectionResetError("the manager is gone")

    monkeypatch.setattr(mc, "_claim_next_index", lambda *a, **k: 0)
    worker = _worker(tmp_path, _MonteCarlo__run_single_simulation=_raise)
    mutex = _RecordingMutex()

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__sim_producer(
            worker, object(), mutex, _UnreachableEvent()
        )

    assert mutex.acquired == mutex.released, "the mutex was left held"
