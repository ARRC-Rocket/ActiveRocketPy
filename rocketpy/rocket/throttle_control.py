import warnings

import numpy as np

from ..prints.roll_control_prints import _RollControlPrints


class ThrottleControl:


    def __init__(
        self,
        max_throttle=0,
        clamp=True,
        throttle=0.0,
        name="Throttle Control",
    ):

        self.name = name
        self.max_throttle = max_throttle
        self.clamp = clamp
        self.initial_throttle = throttle
        self.throttle = throttle
        self.prints = _RollControlPrints(self)

    @property
    def throttle(self):
        """Returns the current roll torque in N·m."""
        return self._throttle

    @throttle.setter
    def throttle(self, value):
        """Sets the roll torque with optional clamping or warning.

        Parameters
        ----------
        value : float
            Roll torque in N·m.
        """
        if abs(value) > self.max_throttle:
            if self.clamp:
                value = np.clip(value, -self.max_throttle, self.max_throttle)
            else:
                warnings.warn(
                    f"Roll torque of {self.name} is {value:.4f} N·m, "
                    f"which exceeds the maximum of {self.max_throttle:.4f} N·m.",
                    UserWarning,
                )
        self._throttle = value

    def _reset(self):
        """Resets the roll control system to its initial state. This method
        is called at the beginning of each simulation to ensure the roll
        control system is in the correct state."""
        self.throttle = self.initial_throttle

    def info(self):
        """Prints summarized information of the roll control system.

        Returns
        -------
        None
        """
        self.prints.basics()

    def all_info(self):
        """Prints all information of the roll control system.

        Returns
        -------
        None
        """
        self.info()

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        return {
            "max_throttle": self.max_throttle,
            "clamp": self.clamp,
            "throttle": self.initial_throttle,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            max_throttle=data.get("max_throttle"),
            clamp=data.get("clamp"),
            throttle=data.get("throttle"),
            name=data.get("name"),
        )
