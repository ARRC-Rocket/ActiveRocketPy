import warnings
from abc import ABC, abstractmethod

import numpy as np


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
        if demand_rate is not None and demand_rate <= 0:
            raise ValueError("demand_rate must be positive or None.")
        self.demand_rate = demand_rate

        if actuator_range[0] > actuator_range[1]:
            raise ValueError("actuator_range[0] must be <= actuator_range[1].")
        self.actuator_range = actuator_range

        if actuator_rate_limit is not None and actuator_rate_limit < 0:
            raise ValueError("actuator_rate_limit must be non-negative or None.")
        self.actuator_rate_limit = actuator_rate_limit

        self.clamp = clamp

        if actuator_time_constant is not None and actuator_time_constant < 0:
            raise ValueError("actuator_time_constant must be non-negative or None.")
        self.actuator_time_constant = actuator_time_constant
        self._update_iir_coefficients()

        # An initial output outside the range used to survive here and come back
        # on every _reset(), even though the output setter would never let the
        # actuator reach such a value afterwards. Treat it the way the setter
        # treats any other out-of-range value, so the two agree.
        if self.clamp:
            actuator_initial_output = float(
                np.clip(actuator_initial_output, actuator_range[0], actuator_range[1])
            )
        elif not actuator_range[0] <= actuator_initial_output <= actuator_range[1]:
            warnings.warn(
                f"Actuator '{name}' initial output {actuator_initial_output} "
                f"is outside its range {actuator_range}."
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
                demand_period = 1.0 / self.demand_rate
                self._alpha = demand_period / (
                    self.actuator_time_constant + demand_period
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
