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


def _two_numbers_or_raise(value, description):
    """A pair of numbers. Infinite is allowed here: it means unbounded."""
    try:
        lower, upper = value
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be exactly two numbers.") from error
    numbers = []
    for endpoint, where in ((lower, 0), (upper, 1)):
        # Not float(), which takes "0" and would leave a YAML string range half
        # converted while the rest of the config kept its strings.
        if isinstance(endpoint, bool) or not isinstance(endpoint, (int, float)):
            raise ValueError(f"{description}[{where}] must be a number.")
        numbers.append(float(endpoint))
    return numbers[0], numbers[1]


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
        # default really is (-inf, inf), meaning an actuator with no range.
        self.demand_rate = _positive_or_none(demand_rate, "demand_rate")

        # A range read out of a config file arrives as anything. Without the
        # shape and number checks, three endpoints drop the third in silence and
        # string endpoints fail inside np.clip with a ufunc loop error.
        lower, upper = _two_numbers_or_raise(actuator_range, "actuator_range")
        if not lower <= upper:
            raise ValueError("actuator_range[0] must be <= actuator_range[1].")
        self.actuator_range = (lower, upper)
        actuator_range = self.actuator_range

        self.actuator_rate_limit = _non_negative_or_none(
            actuator_rate_limit, "actuator_rate_limit"
        )

        self.clamp = clamp

        self.actuator_time_constant = _non_negative_or_none(
            actuator_time_constant, "actuator_time_constant"
        )

        # Neither is implemented for a continuous actuator, and both were
        # accepted with a warning. Asking for a 0.1 s lag and getting an
        # instant response is worth stopping for rather than mentioning.
        if self.demand_rate is None:
            for option, given in (
                ("actuator_rate_limit", self.actuator_rate_limit),
                ("actuator_time_constant", self.actuator_time_constant),
            ):
                if given:
                    raise ValueError(
                        f"{option} needs a demand_rate: it is not applied to a "
                        f"continuous actuator."
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
                # Algebraically Ts / (tau + Ts) with Ts = 1 / demand_rate,
                # written without forming Ts. One expression instead of two,
                # and the two agree to 2.2e-16 over the range of time
                # constants and rates a real actuator uses.
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
