import warnings
from abc import ABC, abstractmethod

import numpy as np


def _finite_or_raise(value, description):
    """Return ``value`` as a float, refusing anything that is not finite.

    A positive test on the value rather than a negated comparison. NaN fails
    every ordered comparison, so a check written as ``value <= 0`` accepts it
    where the assert it replaced rejected it; and the self-comparison that
    caught it, ``not value == value``, reads as a redundant comparison to a
    reader and to pylint alike.

    This also settles two cases the comparisons never named. Infinity is
    refused, since an actuator cannot start at or be driven to one, and a value
    that is not a number at all is refused here rather than a few lines later
    inside ``np.clip``.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be a number.") from error
    if not np.isfinite(number):
        raise ValueError(f"{description} must be a finite number.")
    return number


def _positive_or_none(value, description):
    """An optional number that has to be finite and greater than zero."""
    if value is None:
        return None
    number = _finite_or_raise(value, description)
    if number <= 0:
        raise ValueError(f"{description} must be positive or None.")
    return number


def _non_negative_or_none(value, description):
    """An optional number that has to be finite and not below zero."""
    if value is None:
        return None
    number = _finite_or_raise(value, description)
    if number < 0:
        raise ValueError(f"{description} must be non-negative or None.")
    return number


def _range_or_raise(value, description):
    """Return the two endpoints as a tuple, refusing a range that cannot clamp.

    Endpoints are not held to ``_finite_or_raise``. An infinite bound is how
    this class says "unbounded on that side", and the base default really is
    ``(-inf, inf)``. What has to hold instead is weaker and is the property the
    range is used for: clamping a finite value against it has to give back a
    finite value.

    That fails in exactly two cases, and ordering makes them one. Given
    ``lower <= upper``, a lower bound of ``+inf`` forces the upper bound to
    match, and an upper bound of ``-inf`` forces the lower bound to match; in
    both, ``np.clip`` returns the infinity. So ``(inf, inf)`` passed the
    ordering check, took a finite initial output of 0.5, and stored ``inf``,
    after which the first ordinary command landed on
    ``(1 - alpha) * inf == 0.0 * inf`` and stored NaN, which ``_reset()`` then
    restored on every subsequent flight. A one-sided bound such as ``(0, inf)``
    is unaffected and stays allowed.

    The endpoints are also copied into a new tuple. They were stored as the
    caller's own object, so a list mutated after construction moved the range
    out from under a validation that had already run.
    """
    try:
        lower, upper = value
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be a pair of numbers.") from error
    try:
        lower = float(lower)
        upper = float(upper)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} endpoints must be numbers.") from error
    # Written as a positive test because NaN fails every ordered comparison, so
    # a range of two NaNs would slip through a check phrased as a negation.
    if not lower <= upper:
        raise ValueError(f"{description}[0] must be <= {description}[1].")
    if lower == np.inf or upper == -np.inf:
        raise ValueError(
            f"{description} {(lower, upper)} cannot clamp anything to a finite value."
        )
    return (lower, upper)


class Actuator(ABC):
    """Abstract class used to define actuators.

    Actuators are used to model the dynamics of control systems such as
    throttle, thrust vector, and roll control. They can be used to simulate the response of
    the control system to changes in throttle, thrust vector, or roll torque commands."""

    def __init__(
        self,
        name,
        demand_rate=None,
        actuator_range=(-np.inf, np.inf),
        actuator_rate_limit=None,
        clamp=True,
        actuator_initial_output=0.0,
        actuator_time_constant=None,
    ):
        """Initializes the Actuator class.

        Parameters
        ----------
        name : str
            Name of the actuator.
        demand_rate : float, optional
            Demand rate (Hz) of the actuator. Default is None for continuous-time actuator.
        actuator_range : tuple, optional
            Range of the actuator output. Default is (-np.Inf, np.Inf).
        actuator_rate_limit : float, optional
            Rate limit of the actuator per second. Default is None.
        clamp : bool, optional
            Whether to clamp the actuator output. Default is True.
        actuator_initial_output : float, optional
            Initial output of the actuator. Default is 0.0. A value outside
            actuator_range is treated the way the output setter treats one: it is
            clamped when clamp is True, and warned about otherwise.
        actuator_time_constant : float, optional
            Time constant of the actuator, implemented as a discrete IIR filter. Default is None.

        Returns
        -------
        None
        """

        self.name = name

        # These are argument checks rather than internal invariants, so they raise
        # instead of asserting: python -O drops assert statements, and a negative
        # time constant or rate limit would then be accepted in silence.
        # Finite first, then the bound, so each check says one thing. Written
        # as a negated comparison these accepted NaN, which fails every ordered
        # comparison, and infinity, which passes them: a demand rate of infinity
        # makes the sampling period zero, which drives the IIR coefficient to
        # zero and freezes the output.
        #
        # The range endpoints are deliberately not put through this. The base
        # default really is (-inf, inf), meaning an actuator with no range, so
        # they get the weaker check in _range_or_raise instead.
        self.demand_rate = _positive_or_none(demand_rate, "demand_rate")

        self.actuator_range = _range_or_raise(actuator_range, "actuator_range")
        actuator_range = self.actuator_range

        self.actuator_rate_limit = _non_negative_or_none(
            actuator_rate_limit, "actuator_rate_limit"
        )

        self.clamp = clamp

        self.actuator_time_constant = _non_negative_or_none(
            actuator_time_constant, "actuator_time_constant"
        )
        self._update_iir_coefficients()

        # An initial output outside the range used to survive here and come back
        # on every _reset(), even though the output setter would never let the
        # actuator reach such a value afterwards. Treat it the way the setter
        # treats any other out-of-range value, so the two agree.
        # Refused before clamping: np.clip propagates NaN, so clamping would
        # store it and _reset() would restore it, and there is no direction to
        # clamp it towards in any case.
        actuator_initial_output = _finite_or_raise(
            actuator_initial_output, f"Actuator '{name}' initial output"
        )
        if self.clamp:
            actuator_initial_output = float(
                np.clip(actuator_initial_output, actuator_range[0], actuator_range[1])
            )
        elif not actuator_range[0] <= actuator_initial_output <= actuator_range[1]:
            warnings.warn(
                f"Actuator '{name}' initial output {actuator_initial_output} "
                f"is outside its range {actuator_range}.",
                stacklevel=2,
            )

        self.actuator_initial_output = actuator_initial_output
        self._actuator_output = actuator_initial_output

    def _update_iir_coefficients(self):
        """Updates the IIR filter coefficient based on time constant and
        demand rate. Uses first-order discrete-time system:
        y[n] = alpha * u[n] + (1 - alpha) * y[n-1]
        where alpha = Ts / (tau + Ts)
        """

        if self.actuator_time_constant is not None and self.actuator_time_constant > 0:
            if self.demand_rate is not None:
                # Algebraically Ts / (tau + Ts) with Ts = 1 / demand_rate, but
                # written without forming Ts. That intermediate overflows for a
                # subnormal demand rate: 1.0 / 5e-324 is inf, inf / (1.0 + inf)
                # is NaN, and the filter then stores NaN out of a command that
                # passed every check on the way in. At the other end a huge time
                # constant makes the denominator overflow and pins alpha to
                # zero, freezing the actuator, where 1e-308 with 1e308 should
                # give about 0.5. Neither shape is reachable from a physical
                # configuration, and neither costs anything to rule out: over
                # 2000 random pairs across the ranges that are, the two forms
                # agree to 2.2e-16.
                self._alpha = 1.0 / (
                    1.0 + self.actuator_time_constant * self.demand_rate
                )
            else:
                warnings.warn(
                    f"Actuator time constant currently only implemented on discrete controllers. '{self.name}' dynamics not applied."
                )
                self._alpha = 1.0  # No filtering, direct pass-through
        else:
            self._alpha = 1.0  # No filtering, direct pass-through

    @property
    def actuator_output(self):
        return self._actuator_output

    @actuator_output.setter
    def actuator_output(self, value):
        """Sets the actuator output with optional clamping or warning.

        Parameters
        ----------
        value : float
            Desired actuator output.

        Returns
        -------
        None
        """
        # The same rule as the initial output, so the two agree. It used to
        # reject a NaN handed to the constructor and accept one handed to the
        # setter on the next timestep: np.clip returns NaN for NaN, and both
        # range comparisons are false for NaN, so neither branch noticed and it
        # was stored.
        #
        # This is the path an agent writes to. BalloonPoppingChallenge assigns
        # its actions straight into these setters, and a policy that goes
        # unstable emits NaN, which from here reaches the forces, the moments
        # and the integrator state.
        value = _finite_or_raise(value, f"Actuator '{self.name}' output")

        # Apply first-order IIR actuator dynamics
        value = self._alpha * value + (1 - self._alpha) * self._actuator_output

        # Apply rate limit if specified
        if self.actuator_rate_limit is not None:
            if self.demand_rate is not None:
                max_change = self.actuator_rate_limit / self.demand_rate
                change = value - self._actuator_output
                if abs(change) > max_change:
                    value = self._actuator_output + np.sign(change) * max_change
                    warnings.warn(
                        f"Actuator '{self.name}' output change {change:.3f} exceeds rate limit of {max_change:.3f} per time step."
                    )
            else:
                warnings.warn(
                    f"Actuator rate limit currently only implemented for discrete controllers. '{self.name}' rate limit not applied."
                )

        # Apply range limits if specified
        if self.clamp:
            value = np.clip(value, self.actuator_range[0], self.actuator_range[1])
        else:
            if value < self.actuator_range[0] or value > self.actuator_range[1]:
                warnings.warn(
                    f"Actuator '{self.name}' output {value:.3f} exceeds range limits {self.actuator_range}."
                )

        self._actuator_output = value

    def _reset(self):
        """Resets the actuator to its initial state. This method
        is called at the beginning of each simulation to ensure the actuator
        is in the correct state."""
        self._actuator_output = self.actuator_initial_output

    @abstractmethod
    def info(self):
        """Prints summarized information of the actuator.

        Returns
        -------
        None
        """

    @abstractmethod
    def all_info(self):
        """Prints all information of the actuator.

        Returns
        -------
        None
        """
