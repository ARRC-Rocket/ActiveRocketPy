import warnings

import numpy as np

from ..prints.roll_control_prints import _RollControlPrints


class RollControl:
    """Roll Control system class for managing rocket roll torque.

    This class represents a roll control system that allows the application
    of roll torque around the rocket's X-axis. Ideal roll torque is assumed.

    Attributes
    ----------
    RollControl.roll_torque : float
        Current roll torque magnitude in N·m (Newton-meters).
        Positive values indicate counter-clockwise rotation when viewed
        from the nose of the rocket.
    RollControl.max_roll_torque : float
        Maximum roll torque magnitude in N·m. The roll torque is clamped
        to this value if clamp is True.
    RollControl.clamp : bool, optional
        If True, roll torque is clamped to [-max_roll_torque, max_roll_torque].
        If False, a warning is issued when roll torque exceeds the max value.
    RollControl.name : str
        Name of the roll control system.
    """

    def __init__(
        self,
        max_roll_torque=0,
        clamp=True,
        roll_torque=0.0,
        name="Roll Control",
    ):
        """Initializes the RollControl class.

        Parameters
        ----------
        max_roll_torque : float, int
            Maximum roll torque magnitude in N·m. Must be non-negative.
            Default is 0 (no roll control).
        clamp : bool, optional
            If True, the simulation will clamp roll torque to the range
            [-max_roll_torque, max_roll_torque] if it exceeds this range.
            If False, the simulation will issue a warning if roll torque
            exceeds the maximum value. Default is True.
        roll_torque : float, optional
            Initial roll torque in N·m. Default is 0.0 (no torque).
        name : str, optional
            Name of the roll control system. Default is "Roll Control".

        Returns
        -------
        None
        """
        self.name = name
        self.max_roll_torque = max_roll_torque
        self.clamp = clamp
        self.initial_roll_torque = roll_torque
        self.roll_torque = roll_torque
        self.prints = _RollControlPrints(self)

    @property
    def roll_torque(self):
        """Returns the current roll torque in N·m."""
        return self._roll_torque

    @roll_torque.setter
    def roll_torque(self, value):
        """Sets the roll torque with optional clamping or warning.

        Parameters
        ----------
        value : float
            Roll torque in N·m.
        """
        if abs(value) > self.max_roll_torque:
            if self.clamp:
                value = np.clip(value, -self.max_roll_torque, self.max_roll_torque)
            else:
                warnings.warn(
                    f"Roll torque of {self.name} is {value:.4f} N·m, "
                    f"which exceeds the maximum of {self.max_roll_torque:.4f} N·m.",
                    UserWarning,
                )
        self._roll_torque = value

    def _reset(self):
        """Resets the roll control system to its initial state. This method
        is called at the beginning of each simulation to ensure the roll
        control system is in the correct state."""
        self.roll_torque = self.initial_roll_torque

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
            "max_roll_torque": self.max_roll_torque,
            "clamp": self.clamp,
            "roll_torque": self.initial_roll_torque,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            max_roll_torque=data.get("max_roll_torque"),
            clamp=data.get("clamp"),
            roll_torque=data.get("roll_torque"),
            name=data.get("name"),
        )
