import warnings

import numpy as np

from ..prints.throttle_control_prints import _ThrottleControlPrints


class ThrottleControl:
    """Throttle Control system class for managing rocket throttle.

    This class represents a throttle control system that allows the application
    of throttle around the rocket's engine. Ideal throttle is assumed.

    Attributes
    ----------
    ThrottleControl.throttle : float
        Current throttle magnitude as a fraction of maximum thrust.
        Positive values indicate forward thrust.
    ThrottleControl.throttle_range : tuple of float
        Tuple representing the minimum and maximum throttle values.
        The throttle is clamped to this range if clamp is True.
    ThrottleControl.clamp : bool, optional
        If True, throttle is clamped to throttle_range.
        If False, a warning is issued when throttle exceeds the range.
    ThrottleControl.name : str
        Name of the throttle control system.
    """

    def __init__(
        self,
        sampling_rate=100,
        throttle_range=(0.0, 1.0),
        throttle_rate_limit=0,
        clamp=True,
        throttle=1.0,
        name="Throttle Control",
    ):
        """Initializes the ThrottleControl class.

        Parameters
        ----------
        sampling_rate : int, optional
            Sampling rate of the throttle control system in Hz. Default is 100 Hz.
        throttle_range : tuple of float, optional
            Tuple representing the minimum and maximum throttle values.
            The throttle is clamped to this range if clamp is True.
            Default is (0.0, 1.0).
        throttle_rate_limit : float, int
            Maximum throttle rate in units per second. Throttle is limited to this
            rate. Must be non-negative. Default is 0 (no throttle change).
        clamp : bool, optional
            If True, the simulation will clamp throttle to the range
            [throttle_range[0], throttle_range[1]] if it exceeds this range.
            If False, the simulation will issue a warning if throttle
            exceeds the range. Default is True.
        throttle : float, optional
            Initial throttle value. Default is 1.0 (full throttle).
        name : str, optional
            Name of the throttle control system. Default is "Throttle Control".

        Returns
        -------
        None
        """
        self.name = name
        self.sampling_rate = sampling_rate
        assert throttle_range[0] <= throttle_range[1], (
            "throttle_range[0] must be <= throttle_range[1]"
        )
        self.throttle_range = throttle_range
        assert throttle_rate_limit >= 0, "throttle_rate_limit must be non-negative."
        self.throttle_rate_limit = throttle_rate_limit
        self.clamp = clamp
        self.initial_throttle = throttle
        self.throttle_prev = throttle
        self.throttle = throttle
        self.prints = _ThrottleControlPrints(self)

    @property
    def throttle(self):
        return self._throttle

    @throttle.setter
    def throttle(self, value):
        """Sets the throttle with optional clamping or warning.

        Parameters
        ----------
        value : float
            Throttle value as a fraction of maximum thrust.
        """
        if value < self.throttle_range[0] or value > self.throttle_range[1]:
            if self.clamp:
                value = np.clip(value, self.throttle_range[0], self.throttle_range[1])
            else:
                warnings.warn(
                    f"Throttle of {self.name} is {value:.4f}, "
                    f"which exceeds bounds "
                    f"[{self.throttle_range[0]:.4f}, {self.throttle_range[1]:.4f}].",
                    UserWarning,
                )
        # Limit the throttle rate
        max_throttle_change = self.throttle_rate_limit / self.sampling_rate
        throttle_change = value - self.throttle_prev
        if abs(throttle_change) > max_throttle_change:
            value = self.throttle_prev + np.sign(throttle_change) * max_throttle_change
        self.throttle_prev = value
        self._throttle = value

    def _reset(self):
        """Resets the throttle control system to its initial state. This method
        is called at the beginning of each simulation to ensure the throttle
        control system is in the correct state."""
        self.throttle = self.initial_throttle
        self.throttle_prev = self.initial_throttle

    def info(self):
        """Prints summarized information of the throttle control system.

        Returns
        -------
        None
        """
        self.prints.basics()

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        return {
            "sampling_rate": self.sampling_rate,
            "throttle_range": self.throttle_range,
            "throttle_rate_limit": self.throttle_rate_limit,
            "clamp": self.clamp,
            "throttle": self.throttle,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            sampling_rate=data.get("sampling_rate"),
            throttle_range=data.get("throttle_range"),
            throttle_rate_limit=data.get("throttle_rate_limit"),
            clamp=data.get("clamp"),
            throttle=data.get("throttle"),
            name=data.get("name"),
        )
