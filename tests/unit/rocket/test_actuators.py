import math
import subprocess
import sys

import pytest

from rocketpy.rocket.actuator.roll import RollActuator
from rocketpy.rocket.actuator.throttle import ThrottleActuator
from rocketpy.rocket.actuator.thrust_vector import (
    ThrustVectorActuator,
    ThrustVectorActuator2D,
)


class TestRollActuator:
    """Test suite for RollActuator class."""

    def test_initialization_defaults(self):
        """Test RollActuator initialization with default parameters."""
        actuator = RollActuator()
        assert actuator.name == "Roll Control"
        assert actuator.demand_rate == 100
        assert actuator.actuator_range == (0, 0)
        assert actuator.roll_torque == 0.0
        assert actuator.actuator_rate_limit is None
        assert actuator.clamp is True

    def test_initialization_custom(self):
        """Test RollActuator initialization with custom parameters."""
        actuator = RollActuator(
            name="Custom Roll",
            demand_rate=50,
            max_roll_torque=10.0,
            torque_rate_limit=5.0,
            clamp=False,
            initial_roll_torque=2.0,
        )
        assert actuator.name == "Custom Roll"
        assert actuator.demand_rate == 50
        assert actuator.actuator_range == (-10.0, 10.0)
        assert actuator.roll_torque == 2.0
        assert actuator.actuator_rate_limit == 5.0
        assert actuator.clamp is False

    def test_roll_torque_property(self):
        """Test roll torque getter and setter."""
        actuator = RollActuator(max_roll_torque=10.0)
        actuator.roll_torque = 5.0
        assert actuator.roll_torque == 5.0

    def test_roll_torque_clamping(self):
        """Test roll torque clamping to range."""
        actuator = RollActuator(max_roll_torque=10.0, clamp=True)
        actuator.actuator_output = 15.0
        assert actuator.roll_torque == 10.0
        actuator.actuator_output = -15.0
        assert actuator.roll_torque == -10.0

    def test_to_dict(self):
        """Test to_dict serialization."""
        actuator = RollActuator(
            name="Test Roll",
            max_roll_torque=5.0,
            torque_rate_limit=2.0,
            initial_roll_torque=1.0,
        )
        data = actuator.to_dict()
        assert data["name"] == "Test Roll"
        assert data["max_roll_torque"] == 5.0
        assert data["torque_rate_limit"] == 2.0
        assert data["initial_roll_torque"] == 1.0

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "name": "Test Roll",
            "demand_rate": 50,
            "max_roll_torque": 5.0,
            "torque_rate_limit": 2.0,
            "clamp": True,
            "initial_roll_torque": 1.0,
            "roll_torque_time_constant": None,
        }
        actuator = RollActuator.from_dict(data)
        assert actuator.name == "Test Roll"
        assert actuator.demand_rate == 50
        assert actuator.roll_torque == 1.0

    def test_reset(self):
        """Test actuator reset functionality."""
        actuator = RollActuator(max_roll_torque=10.0, initial_roll_torque=2.0)
        actuator.roll_torque = 5.0
        actuator._reset()
        assert actuator.roll_torque == 2.0

    def test_info_methods(self):
        """Test info and all_info methods."""
        actuator = RollActuator()
        # Just verify methods exist and don't raise exceptions
        actuator.info()
        actuator.all_info()


class TestThrottleActuator:
    """Test suite for ThrottleActuator class."""

    def test_initialization_defaults(self):
        """Test ThrottleActuator initialization with default parameters."""
        actuator = ThrottleActuator()
        assert actuator.name == "Throttle Control"
        assert actuator.demand_rate == 100
        assert actuator.actuator_range == (0, 1)
        assert actuator.throttle == 1.0

    def test_initialization_custom(self):
        """Test ThrottleActuator initialization with custom parameters."""
        actuator = ThrottleActuator(
            name="Custom Throttle",
            demand_rate=50,
            throttle_range=(0, 0.8),
            throttle_rate_limit=0.1,
            initial_throttle=0.5,
        )
        assert actuator.name == "Custom Throttle"
        assert actuator.actuator_range == (0, 0.8)
        assert actuator.throttle == 0.5

    def test_throttle_property(self):
        """Test throttle getter and setter."""
        actuator = ThrottleActuator(throttle_range=(0, 1.0))
        actuator.throttle = 0.5
        assert actuator.throttle == 0.5

    def test_throttle_clamping(self):
        """Test throttle clamping to range."""
        actuator = ThrottleActuator(throttle_range=(0, 1.0), clamp=True)
        actuator.actuator_output = 1.5
        assert actuator.throttle == 1.0
        actuator.actuator_output = -0.5
        assert actuator.throttle == 0.0

    def test_to_dict(self):
        """Test to_dict serialization."""
        actuator = ThrottleActuator(
            name="Test Throttle",
            throttle_range=(0, 0.8),
            throttle_rate_limit=0.1,
            initial_throttle=0.3,
        )
        data = actuator.to_dict()
        assert data["name"] == "Test Throttle"
        assert data["throttle_range"] == (0, 0.8)
        assert data["throttle_rate_limit"] == 0.1
        assert data["initial_throttle"] == 0.3

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "name": "Test Throttle",
            "demand_rate": 50,
            "throttle_range": (0, 0.8),
            "throttle_rate_limit": 0.1,
            "clamp": True,
            "initial_throttle": 0.3,
            "throttle_time_constant": None,
        }
        actuator = ThrottleActuator.from_dict(data)
        assert actuator.name == "Test Throttle"
        assert actuator.throttle == 0.3

    def test_reset(self):
        """Test actuator reset functionality."""
        actuator = ThrottleActuator(initial_throttle=0.3)
        actuator.throttle = 0.7
        actuator._reset()
        assert actuator.throttle == 0.3

    def test_info_methods(self):
        """Test info and all_info methods."""
        actuator = ThrottleActuator()
        actuator.info()
        actuator.all_info()


class TestThrustVectorActuator:
    """Test suite for ThrustVectorActuator class."""

    def test_initialization_defaults(self):
        """Test ThrustVectorActuator initialization with default parameters."""
        actuator = ThrustVectorActuator()
        assert actuator.name == "Thrust Vector Control"
        assert actuator.demand_rate == 100
        assert actuator.actuator_range == (-10, 10)
        assert actuator.gimbal_angle == 0.0

    def test_initialization_custom(self):
        """Test ThrustVectorActuator initialization with custom parameters."""
        actuator = ThrustVectorActuator(
            name="Custom TVC",
            demand_rate=50,
            max_gimbal_angle=15.0,
            gimbal_rate_limit=5.0,
            initial_gimbal_angle=5.0,
        )
        assert actuator.name == "Custom TVC"
        assert actuator.actuator_range == (-15.0, 15.0)
        assert actuator.gimbal_angle == 5.0

    def test_gimbal_angle_property(self):
        """Test gimbal angle getter and setter."""
        actuator = ThrustVectorActuator(max_gimbal_angle=10.0)
        actuator.gimbal_angle = 5.0
        assert actuator.gimbal_angle == 5.0

    def test_gimbal_angle_clamping(self):
        """Test gimbal angle clamping to range."""
        actuator = ThrustVectorActuator(max_gimbal_angle=10.0, clamp=True)
        actuator.actuator_output = 15.0
        assert actuator.gimbal_angle == 10.0
        actuator.actuator_output = -15.0
        assert actuator.gimbal_angle == -10.0

    def test_to_dict(self):
        """Test to_dict serialization."""
        actuator = ThrustVectorActuator(
            name="Test TVC",
            max_gimbal_angle=8.0,
            gimbal_rate_limit=3.0,
            initial_gimbal_angle=2.0,
        )
        data = actuator.to_dict()
        assert data["name"] == "Test TVC"
        assert data["max_gimbal_angle"] == 8.0
        assert data["gimbal_rate_limit"] == 3.0
        assert data["initial_gimbal_angle"] == 2.0

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "name": "Test TVC",
            "demand_rate": 50,
            "max_gimbal_angle": 8.0,
            "gimbal_rate_limit": 3.0,
            "clamp": True,
            "initial_gimbal_angle": 2.0,
            "gimbal_time_constant": None,
        }
        actuator = ThrustVectorActuator.from_dict(data)
        assert actuator.name == "Test TVC"
        assert actuator.gimbal_angle == 2.0

    def test_reset(self):
        """Test actuator reset functionality."""
        actuator = ThrustVectorActuator(initial_gimbal_angle=2.0)
        actuator.gimbal_angle = 5.0
        actuator._reset()
        assert actuator.gimbal_angle == 2.0

    def test_info_methods(self):
        """Test info and all_info methods."""
        actuator = ThrustVectorActuator()
        actuator.info()
        actuator.all_info()


class TestThrustVectorActuator2D:
    """Test suite for ThrustVectorActuator2D class."""

    def test_initialization_defaults(self):
        """Test ThrustVectorActuator2D initialization with default parameters."""
        actuator = ThrustVectorActuator2D()
        assert actuator.gimbal_angle_x == 0.0
        assert actuator.gimbal_angle_y == 0.0
        assert actuator.gimbal_angles == (0.0, 0.0)

    def test_initialization_custom(self):
        """Test ThrustVectorActuator2D initialization with custom parameters."""
        actuator = ThrustVectorActuator2D(
            name="Custom 2D TVC",
            max_gimbal_angle=15.0,
            initial_gimbal_angle=5.0,
        )
        assert actuator.x.name == "Custom 2D TVC X-axis"
        assert actuator.y.name == "Custom 2D TVC Y-axis"
        assert actuator.gimbal_angles == (5.0, 5.0)

    def test_gimbal_angle_x_property(self):
        """Test gimbal angle X getter and setter."""
        actuator = ThrustVectorActuator2D()
        actuator.gimbal_angle_x = 5.0
        assert actuator.gimbal_angle_x == 5.0

    def test_gimbal_angle_y_property(self):
        """Test gimbal angle Y getter and setter."""
        actuator = ThrustVectorActuator2D()
        actuator.gimbal_angle_y = 7.0
        assert actuator.gimbal_angle_y == 7.0

    def test_gimbal_angles_property_get(self):
        """Test gimbal angles tuple getter."""
        actuator = ThrustVectorActuator2D()
        actuator.gimbal_angle_x = 3.0
        actuator.gimbal_angle_y = 4.0
        assert actuator.gimbal_angles == (3.0, 4.0)

    def test_gimbal_angles_property_set(self):
        """Test gimbal angles tuple setter."""
        actuator = ThrustVectorActuator2D()
        actuator.gimbal_angles = (5.0, 7.0)
        assert actuator.gimbal_angle_x == 5.0
        assert actuator.gimbal_angle_y == 7.0
        assert actuator.gimbal_angles == (5.0, 7.0)

    def test_independent_axes(self):
        """Test that X and Y axes are independent."""
        actuator = ThrustVectorActuator2D(max_gimbal_angle=10.0)
        actuator.gimbal_angle_x = 5.0
        actuator.gimbal_angle_y = -3.0
        assert actuator.gimbal_angle_x == 5.0
        assert actuator.gimbal_angle_y == -3.0
        assert actuator.gimbal_angles == (5.0, -3.0)

    def test_clamping_individual_axes(self):
        """Test clamping on individual axes."""
        actuator = ThrustVectorActuator2D(max_gimbal_angle=10.0, clamp=True)
        # Test X axis clamping
        actuator.x.actuator_output = 15.0
        assert actuator.gimbal_angle_x == 10.0
        # Test Y axis clamping
        actuator.y.actuator_output = -15.0
        assert actuator.gimbal_angle_y == -10.0

    def test_to_dict(self):
        """Test to_dict serialization for ThrustVectorActuator2D."""
        actuator = ThrustVectorActuator2D(
            name="Custom 2D TVC",
            demand_rate=50,
            max_gimbal_angle=8.0,
            gimbal_rate_limit=3.0,
            clamp=True,
            initial_gimbal_angle=2.0,
            gimbal_time_constant=0.1,
        )
        data = actuator.to_dict()
        assert data["name"] == "Custom 2D TVC"
        assert data["demand_rate"] == 50
        assert data["max_gimbal_angle"] == 8.0
        assert data["gimbal_rate_limit"] == 3.0
        assert data["clamp"] is True
        assert data["initial_gimbal_angle"] == 2.0
        assert data["gimbal_time_constant"] == 0.1

    def test_from_dict(self):
        """Test from_dict deserialization for ThrustVectorActuator2D."""
        data = {
            "name": "Test 2D TVC",
            "demand_rate": 75,
            "max_gimbal_angle": 12.0,
            "gimbal_rate_limit": 4.0,
            "clamp": False,
            "initial_gimbal_angle": 3.0,
            "gimbal_time_constant": 0.2,
        }
        actuator = ThrustVectorActuator2D.from_dict(data)
        assert actuator.x.name == "Test 2D TVC X-axis"
        assert actuator.y.name == "Test 2D TVC Y-axis"
        assert actuator.x.demand_rate == 75
        assert actuator.y.demand_rate == 75
        assert actuator.gimbal_angles == (3.0, 3.0)
        assert actuator.x.actuator_range == (-12.0, 12.0)

    def test_from_dict_roundtrip(self):
        """Test roundtrip serialization/deserialization."""
        original = ThrustVectorActuator2D(
            name="Roundtrip Test",
            demand_rate=60,
            max_gimbal_angle=9.0,
            gimbal_rate_limit=2.5,
            clamp=True,
            initial_gimbal_angle=1.5,
            gimbal_time_constant=0.15,
        )
        data = original.to_dict()
        reconstructed = ThrustVectorActuator2D.from_dict(data)
        assert reconstructed.x.name == original.x.name
        assert reconstructed.x.demand_rate == original.x.demand_rate
        assert reconstructed.gimbal_angles == original.gimbal_angles

    def test_info_methods(self):
        """Test info and all_info methods for ThrustVectorActuator2D."""
        actuator = ThrustVectorActuator2D(
            name="Test 2D TVC",
            max_gimbal_angle=10.0,
            initial_gimbal_angle=2.0,
        )
        # These should not raise exceptions
        actuator.info()
        actuator.all_info()

    def test_info_methods_with_custom_parameters(self):
        """Test info methods with custom gimbal angles."""
        actuator = ThrustVectorActuator2D(
            name="Custom Info Test",
            demand_rate=200,
            max_gimbal_angle=15.0,
            gimbal_rate_limit=10.0,
            initial_gimbal_angle=5.0,
            gimbal_time_constant=0.05,
        )
        actuator.gimbal_angle_x = 7.5
        actuator.gimbal_angle_y = -3.2
        # These should not raise exceptions and reflect current state
        actuator.info()
        actuator.all_info()


class TestActuatorDynamics:
    """Test suite for actuator dynamics (rate limiting, time constants)."""

    def test_rate_limiting_roll(self):
        """Test rate limiting on roll actuator."""
        actuator = RollActuator(
            max_roll_torque=10.0,
            demand_rate=100,
            torque_rate_limit=5.0,
        )
        # Change should be limited
        actuator.actuator_output = 10.0  # Try to jump to 10
        # At 100 Hz demand rate: max_change = 5.0 / 100 = 0.05
        # So output should be clamped to 0.05
        assert actuator.roll_torque <= 0.1  # Small due to rate limit

    def test_no_rate_limiting_when_none(self):
        """Test that no rate limiting occurs when set to None."""
        actuator = RollActuator(
            max_roll_torque=10.0,
            torque_rate_limit=None,
        )
        actuator.actuator_output = 5.0
        assert actuator.roll_torque == 5.0

    def test_time_constant_iir_filter(self):
        """Test IIR filter behavior with time constant."""
        actuator = ThrottleActuator(
            throttle_range=(0, 1.0),
            demand_rate=100,
            throttle_time_constant=0.1,
            initial_throttle=0.0,
        )
        # With time constant, output should be filtered
        # alpha = Ts / (tau + Ts) = 0.01 / (0.1 + 0.01) ≈ 0.0909
        initial_output = actuator.throttle
        actuator.actuator_output = 1.0
        filtered_output = actuator.throttle
        # Output should be between initial and 1.0 due to filtering
        assert initial_output < filtered_output < 1.0


NAN = float("nan")
INF = float("inf")


def _as_source(value):
    """Render a value as source the ``-O`` subprocess can evaluate.

    ``repr`` is almost enough, except that it renders NaN as the bare name
    ``nan`` and infinity as ``inf``, neither of which the subprocess has bound.
    Left as ``repr`` those cases died on NameError, and a test that only checked
    the return code would have called that a pass.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return f'float("{value}")'
    if isinstance(value, tuple):
        return "(" + ", ".join(_as_source(item) for item in value) + ",)"
    return repr(value)


# One table, walked twice: once in-process for the message, once under ``-O``.
# Keeping them in step is the point. An argument that is only rejected in the
# default interpreter is not rejected, because the checks these replaced were
# asserts and asserts are what ``-O`` removes.
#
# Each entry names the message it expects, so a case cannot pass on some other
# argument's check. Every NaN case is here because NaN fails every ordered
# comparison: `nan <= 0` is false just as `nan > 0` is, so a check written as the
# inverted comparison accepts it while the assert it replaces rejected it.
INVALID_ARGUMENTS = [
    (RollActuator, {"demand_rate": -1}, "demand_rate"),
    (RollActuator, {"demand_rate": 0}, "demand_rate"),
    (RollActuator, {"demand_rate": NAN}, "demand_rate"),
    (RollActuator, {"max_roll_torque": -5}, "actuator_range"),
    (ThrottleActuator, {"throttle_range": (NAN, 1.0)}, "actuator_range"),
    (ThrottleActuator, {"throttle_range": (0.0, NAN)}, "actuator_range"),
    (ThrustVectorActuator, {"gimbal_rate_limit": -1.0}, "rate_limit"),
    (ThrustVectorActuator, {"gimbal_rate_limit": NAN}, "rate_limit"),
    (ThrottleActuator, {"throttle_time_constant": -0.1}, "time_constant"),
    (ThrottleActuator, {"throttle_time_constant": NAN}, "time_constant"),
    (ThrottleActuator, {"initial_throttle": NAN}, "initial output"),
    # clamp is what would otherwise absorb an out-of-range initial value, and
    # np.clip returns NaN for NaN, so both settings have to refuse it.
    (ThrottleActuator, {"initial_throttle": NAN, "clamp": False}, "initial output"),
    # Infinity was never named by the comparisons. An actuator cannot start at
    # one, and clamping would quietly turn it into a range endpoint.
    (ThrottleActuator, {"initial_throttle": INF}, "initial output"),
    (RollActuator, {"demand_rate": INF}, "demand_rate"),
]
INVALID_IDS = [
    f"{cls.__name__}-{'-'.join(kwargs)}-{'clamped' if kwargs.get('clamp', True) else 'unclamped'}"
    for cls, kwargs, _ in INVALID_ARGUMENTS
]


class TestTheOutputSetterRefusesWhatTheConstructorDoes:
    """The two used to disagree, and the setter is the one an agent writes to.

    A NaN handed to the constructor was rejected; the same NaN handed to the
    setter on the next timestep was stored. ``np.clip`` returns NaN for NaN, and
    both range comparisons are false for NaN, so neither the clamped nor the
    unclamped branch noticed.

    BalloonPoppingChallenge assigns agent actions straight into these setters,
    and a policy that goes unstable emits NaN, which from there reaches the
    forces, the moments and the integrator state.
    """

    @pytest.mark.parametrize("clamp", [True, False], ids=["clamped", "unclamped"])
    @pytest.mark.parametrize("value", [NAN, INF, -INF], ids=["nan", "inf", "-inf"])
    def test_a_non_finite_command_is_refused(self, clamp, value):
        actuator = ThrottleActuator(clamp=clamp)

        with pytest.raises(ValueError, match="output"):
            actuator.actuator_output = value

    @pytest.mark.parametrize("clamp", [True, False], ids=["clamped", "unclamped"])
    def test_an_ordinary_command_still_goes_through(self, clamp):
        """Or refusing everything would satisfy the test above."""
        actuator = ThrottleActuator(clamp=clamp)

        actuator.actuator_output = 0.5

        assert actuator.actuator_output == pytest.approx(0.5)

    def test_the_stored_output_is_untouched_by_a_refused_command(self):
        actuator = ThrottleActuator()
        actuator.actuator_output = 0.5

        with pytest.raises(ValueError):
            actuator.actuator_output = NAN

        assert actuator.actuator_output == pytest.approx(0.5)


class TestActuatorValidation:
    """Test suite for actuator parameter validation."""

    @pytest.mark.parametrize(
        "actuator_class, kwargs, message", INVALID_ARGUMENTS, ids=INVALID_IDS
    )
    def test_invalid_arguments_are_rejected(self, actuator_class, kwargs, message):
        with pytest.raises(ValueError, match=message):
            actuator_class(**kwargs)

    @pytest.mark.parametrize(
        "actuator_class, kwargs, message", INVALID_ARGUMENTS, ids=INVALID_IDS
    )
    def test_validation_survives_optimized_mode(self, actuator_class, kwargs, message):
        """``python -O`` drops assert statements, so these must not be asserts.

        Run in a subprocess because the flag is set at interpreter startup. Under
        the old bare asserts every one of these was accepted in silence.
        """
        arguments = ", ".join(f"{k}={_as_source(v)}" for k, v in kwargs.items())
        source = (
            "from rocketpy.rocket.actuator import "
            f"{actuator_class.__name__} as A; A({arguments})"
        )
        result = subprocess.run(
            [sys.executable, "-O", "-c", source],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0, "invalid arguments were accepted under -O"
        assert "ValueError" in result.stderr
        assert message in result.stderr

    def test_demand_rate_none_builds_a_continuous_actuator(self):
        """None is the documented continuous-time mode and must be accepted.

        The check used to read ``demand_rate > 0 or demand_rate is None``, and
        Python evaluates the left operand first, so this raised TypeError and the
        mode could not be constructed at all.
        """
        actuator = RollActuator(demand_rate=None)

        assert actuator.demand_rate is None

    @pytest.mark.parametrize(
        "actuator_class, kwargs",
        [
            (RollActuator, {"torque_rate_limit": 0.0}),
            (ThrottleActuator, {"throttle_time_constant": 0.0}),
            (ThrustVectorActuator, {"gimbal_rate_limit": 0.0}),
        ],
    )
    def test_zero_is_accepted_where_the_bound_is_non_negative(
        self, actuator_class, kwargs
    ):
        """Zero is on the legal side of every non-negative bound.

        Worth its own case because the fix moved these from ``x < 0`` to
        ``not x >= 0``, and an off-by-one there would turn the documented "no
        dynamics" and "no rate limit" settings into errors. A zero rate limit
        does freeze the actuator, but that is the caller's business, and the
        constructor is not where that is decided.
        """
        actuator = actuator_class(**kwargs)

        assert actuator is not None


class TestActuatorInitialOutput:
    """An actuator must not start outside its own range.

    The output setter already refuses to leave the range, but the initial value
    bypassed it and ``_reset()`` restored that same value, so a reset actuator
    ended up somewhere the setter would never have put it.
    """

    def test_an_out_of_range_initial_value_is_clamped(self):
        actuator = ThrottleActuator(throttle_range=(0.0, 1.0), initial_throttle=2.0)

        assert actuator.actuator_output == 1.0
        assert actuator.actuator_initial_output == 1.0

    def test_the_clamped_value_survives_a_reset(self):
        actuator = ThrottleActuator(throttle_range=(0.0, 1.0), initial_throttle=-3.0)
        actuator.actuator_output = 0.5
        actuator._reset()

        assert actuator.actuator_output == 0.0

    def test_a_value_inside_the_range_is_untouched(self):
        actuator = ThrottleActuator(throttle_range=(0.0, 1.0), initial_throttle=0.25)

        assert actuator.actuator_initial_output == 0.25

    def test_without_clamping_it_warns_instead(self):
        # clamp=False is the documented way to let an actuator report outside its
        # range, so the initial value follows the setter and only warns.
        with pytest.warns(UserWarning, match="outside its range"):
            actuator = ThrottleActuator(
                throttle_range=(0.0, 1.0), initial_throttle=2.0, clamp=False
            )

        assert actuator.actuator_initial_output == 2.0

    def test_the_warning_is_not_blamed_on_the_actuator_module(self):
        """``pytest.warns`` reads the message, and the message is not the whole
        warning. Without a stacklevel the report points at the ``warnings.warn``
        line inside ``actuator.py``, which is the same line for every caller, so
        ``-W`` filters keyed on a module and the printed location are both
        useless.

        Asserting the negative rather than a specific file because the warning is
        raised in a base ``__init__`` reached through ``super()``, so no single
        stacklevel lands on user code for every actuator: 2 reaches the concrete
        subclass, and the dual-axis actuator adds another frame on top of that.
        Not blaming the base module is the part that holds for all of them.
        """
        with pytest.warns(UserWarning, match="outside its range") as record:
            ThrottleActuator(
                throttle_range=(0.0, 1.0), initial_throttle=2.0, clamp=False
            )

        assert not record[0].filename.endswith("actuator.py")


class TestActuatorWarnings:
    """Test suite for actuator warning conditions."""

    def test_clamp_false_no_clamping(self):
        """Test that output is not clamped when clamp=False."""
        actuator = RollActuator(max_roll_torque=10.0, clamp=False)
        actuator.actuator_output = 15.0
        # Output should not be clamped when clamp=False
        assert actuator.roll_torque == 15.0

    def test_clamping_applied(self):
        """Test that clamping is applied when clamp=True."""
        actuator = RollActuator(max_roll_torque=10.0, clamp=True)
        actuator.actuator_output = 15.0
        # Output should be clamped to range
        assert actuator.roll_torque == 10.0


class TestARangeThatCannotClampIsRefused:
    """The range was checked for ordering only, and stored by reference.

    Both halves let a finite value become a non-finite stored output, which is
    the one thing the validation around it exists to prevent. Measured before
    the fix: ``(inf, inf)`` passed ``lower <= upper``, ``np.clip(0.5, inf, inf)``
    stored ``inf``, and the next ordinary command reached
    ``(1 - alpha) * inf``, which is ``0.0 * inf``, and stored NaN. ``_reset()``
    put it back at the start of every later flight.
    """

    @pytest.mark.parametrize("limits", [(math.inf, math.inf), (-math.inf, -math.inf)])
    def test_a_range_with_nothing_finite_in_it_is_refused(self, limits):
        with pytest.raises(ValueError, match="clamp"):
            ThrottleActuator(throttle_range=limits, initial_throttle=0.5)

    def test_a_range_of_nans_is_refused(self):
        """NaN fails every ordered comparison, so the ordering check catches it
        only because that check is written as a positive test."""
        with pytest.raises(ValueError):
            ThrottleActuator(throttle_range=(math.nan, math.nan))

    @pytest.mark.parametrize("limits", [(0.0,), (0.0, 1.0, 2.0), 1.0])
    def test_a_range_that_is_not_a_pair_is_refused(self, limits):
        with pytest.raises(ValueError):
            ThrottleActuator(throttle_range=limits)

    @pytest.mark.parametrize(
        "limits",
        [(-math.inf, math.inf), (0.0, math.inf), (-math.inf, 1.0), (0.0, 1.0)],
    )
    def test_a_range_that_can_clamp_still_builds(self, limits):
        """The half that stops this being satisfied by refusing everything.

        An infinite bound is how this class says "unbounded on that side", and
        the base default really is ``(-inf, inf)``, so only a range with no
        finite value on the clamping side may be refused.
        """
        actuator = ThrottleActuator(throttle_range=limits, initial_throttle=0.5)
        actuator.throttle = 0.25

        assert math.isfinite(actuator.throttle)

    def test_mutating_the_range_passed_in_does_not_move_the_actuator(self):
        """It was the caller's own list, so a validated invariant could be
        edited away after the fact."""
        limits = [0.0, 1.0]
        actuator = ThrottleActuator(throttle_range=limits, initial_throttle=0.5)

        limits[:] = [math.inf, math.inf]
        actuator.throttle = 0.25

        assert actuator.actuator_range == (0.0, 1.0)
        assert actuator.throttle == 0.25


class TestTheFilterCoefficientStaysFinite:
    """``alpha`` is derived, and validating only the inputs left it unchecked.

    ``Ts / (tau + Ts)`` with ``Ts = 1 / demand_rate`` forms an intermediate that
    the arguments themselves never contain. Both arguments below are finite and
    pass every check at the boundary.
    """

    def test_a_subnormal_demand_rate_does_not_give_a_nan_coefficient(self):
        """Measured before the fix: ``1.0 / 5e-324`` is inf, ``inf / (1.0 + inf)``
        is NaN, and the first finite command then stored NaN and stayed there."""
        actuator = ThrottleActuator(
            demand_rate=5e-324,
            throttle_time_constant=1.0,
            throttle_range=(0.0, 1.0),
            initial_throttle=0.5,
        )
        actuator.throttle = 0.25

        assert math.isfinite(actuator._alpha)
        assert math.isfinite(actuator.throttle)

    def test_a_huge_time_constant_does_not_pin_the_coefficient_to_zero(self):
        """The other end of the same overflow. The denominator went infinite and
        alpha came out 0, which freezes the actuator at its initial value; the
        answer here is about 0.5."""
        actuator = ThrottleActuator(
            demand_rate=1e-308,
            throttle_time_constant=1e308,
            throttle_range=(0.0, 1.0),
            initial_throttle=0.5,
        )

        assert actuator._alpha == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "demand_rate, time_constant",
        [(1e-3, 1e-6), (1.0, 1.0), (100.0, 0.05), (1e4, 1e3)],
    )
    def test_the_two_forms_agree_where_both_work(self, demand_rate, time_constant):
        """So the rewrite is a rewrite and not a change of behaviour. Measured
        over 2000 random pairs across these ranges: the largest difference is
        2.2e-16."""
        actuator = ThrottleActuator(
            demand_rate=demand_rate,
            throttle_time_constant=time_constant,
            throttle_range=(0.0, 1.0),
        )
        demand_period = 1.0 / demand_rate

        assert actuator._alpha == pytest.approx(
            demand_period / (time_constant + demand_period), rel=1e-12
        )


def _no_op_controller(time, sampling_rate, state, state_history, observed, interactive):  # pylint: disable=unused-argument
    """A controller that commands nothing, so these tests are about the wiring."""
    return None


class TestTheControllerAndTheActuatorShareOneSamplingRate:
    """``add_*_control`` built the two halves from the same argument.

    The actuator normalizes what it is given and stores a float. The controller
    was handed the original object, so one quantity ended up held twice, in two
    types. Nothing complained: the call returned a rocket that looked complete,
    and the failure surfaced later inside Flight at
    ``controller_time_step = 1 / controller.sampling_rate``, by which point the
    actuator and the controller were already attached.

    Passing the actuator's normalized value is the smaller of the two available
    fixes. Refusing strings and bools outright is the other, and it is a wider
    behaviour change than this needs.
    """

    # The third entry reaches through .x because ThrustVectorActuator2D keeps
    # no demand_rate of its own, only the two axes it builds from the argument.
    ADDERS = [
        ("add_roll_control", "roll_control", {"max_roll_torque": 10.0}, False),
        ("add_throttle_control", "throttle_control", {}, False),
        (
            "add_thrust_vector_control",
            "thrust_vector_control",
            {"max_gimbal_angle": 5.0},
            True,
        ),
    ]

    @staticmethod
    def _rate_of(rocket, attribute, per_axis):
        actuator = getattr(rocket, attribute)
        return (actuator.x if per_axis else actuator).demand_rate

    @pytest.mark.parametrize("adder, attribute, extra, per_axis", ADDERS)
    @pytest.mark.parametrize("given", ["100", True, 100])
    def test_both_halves_hold_the_same_normalized_value(
        self, calisto, adder, attribute, extra, per_axis, given
    ):
        """``True`` is in here because ``float(True)`` is 1.0, so a bool reaches
        the actuator as a rate and used to reach the controller as a bool."""
        getattr(calisto, adder)(
            controller_function=_no_op_controller, sampling_rate=given, **extra
        )
        rate = self._rate_of(calisto, attribute, per_axis)
        controller = calisto._controllers[-1]

        assert controller.sampling_rate == rate
        assert type(controller.sampling_rate) is type(rate)

    @pytest.mark.parametrize(
        "adder, extra", [(adder, extra) for adder, _, extra, _ in ADDERS]
    )
    def test_the_controller_rate_is_usable_as_a_time_step(self, calisto, adder, extra):
        """The exact expression Flight evaluates, which is where a string blew
        up with a TypeError after the rocket had already been assembled."""
        getattr(calisto, adder)(
            controller_function=_no_op_controller, sampling_rate="100", **extra
        )
        controller = calisto._controllers[-1]

        assert 1 / controller.sampling_rate == pytest.approx(0.01)


class TestAFilterThatCannotRespondIsRefused:
    """The rewritten coefficient removed one overflow and left a second.

    ``tau * demand_rate`` can overflow where neither factor does, and ``1 / inf``
    is 0, which is a filter that never moves. Measured: an actuator built that
    way holds its initial output against every command for the whole flight.
    """

    @pytest.mark.parametrize(
        "time_constant, demand_rate", [(1e200, 1e200), (1e308, 1e10), (1e154, 1e155)]
    )
    def test_a_coefficient_that_overflows_to_zero_is_refused(
        self, time_constant, demand_rate
    ):
        with pytest.raises(ValueError, match="cannot respond"):
            ThrottleActuator(
                demand_rate=demand_rate,
                throttle_time_constant=time_constant,
                throttle_range=(0.0, 1.0),
                initial_throttle=0.5,
            )

    @pytest.mark.parametrize(
        "time_constant, demand_rate, expected",
        [(0.01, 100.0, 0.5), (10.0, 1000.0, 1e-4)],
    )
    def test_a_slow_actuator_is_still_allowed(
        self, time_constant, demand_rate, expected
    ):
        """The half that stops this being satisfied by refusing slow actuators.
        A small coefficient is what a slow actuator is; only zero is broken."""
        actuator = ThrottleActuator(
            demand_rate=demand_rate,
            throttle_time_constant=time_constant,
            throttle_range=(0.0, 1.0),
        )

        assert actuator._alpha == pytest.approx(expected, rel=1e-3)
