import warnings

import numpy as np

from ..prints.tvc_prints import _TVCPrints


class TVC:
    """Thrust Vector Control (TVC) system class used as a controllable component.

    This class represents a thrust vector control system that allows deflection
    of the thrust vector through gimbal angles. TVC is typically controlled
    by a controller function similar to air brakes and is used by ``Flight``
    to model thrust vectoring behavior.

    Attributes
    ----------
    TVC.gimbal_angle_x : float
        Current gimbal angle around the x-axis (pitch), in degrees.
        Positive values provides positive M1 (pitch moment).
    TVC.gimbal_angle_y : float
        Current gimbal angle around the y-axis (yaw), in degrees.
        Positive values provides positive M2 (yaw moment).
    TVC.gimbal_range : float
        Maximum gimbal angle magnitude in degrees. Both x and y gimbal angles
        are clamped to this value if clamp is True.
    TVC.clamp : bool, optional
        If True, gimbal angles are clamped to [-gimbal_range, gimbal_range].
        If False, a warning is issued when gimbal angles exceed the max value.
    TVC.name : str
        Name of the TVC system.
    """

    def __init__(
        self,
        gimbal_range=0,
        clamp=True,
        gimbal_angle_x=0.0,
        gimbal_angle_y=0.0,
        name="TVC",
    ):
        """Initializes the TVC class.

        Parameters
        ----------
        gimbal_range : int, float
            Maximum gimbal angle magnitude in degrees. Both x and y gimbal
            angles are clamped to this value if clamp is True. Must be
            positive.
        clamp : bool, optional
            If True, the simulation will clamp gimbal angles to the range
            [-gimbal_range, gimbal_range] if they exceed this range.
            If False, the simulation will issue a warning if gimbal angles
            exceed the maximum value. Default is True.
        gimbal_angle_x : float, optional
            Initial gimbal angle around the x-axis (pitch) in degrees.
            Default is 0.0 (no deflection).
        gimbal_angle_y : float, optional
            Initial gimbal angle around the y-axis (yaw) in degrees.
            Default is 0.0 (no deflection).
        name : str, optional
            Name of the TVC system. Default is "TVC".

        Returns
        -------
        None
        """
        self.name = name
        self.gimbal_range = gimbal_range
        self.clamp = clamp
        self.initial_gimbal_angle_x = gimbal_angle_x
        self.initial_gimbal_angle_y = gimbal_angle_y
        self.gimbal_angle_x = gimbal_angle_x
        self.gimbal_angle_y = gimbal_angle_y
        self.prints = _TVCPrints(self)

    @property
    def gimbal_angle_x(self):
        """Returns the current gimbal angle around the x-axis (pitch)."""
        return self._gimbal_angle_x

    @gimbal_angle_x.setter
    def gimbal_angle_x(self, value):
        # Check if deployment level is within bounds and warn user if not
        if abs(value) > self.gimbal_range:
            if self.clamp:
                value = np.clip(value, -self.gimbal_range, self.gimbal_range)
            else:
                warnings.warn(
                    f"Gimbal angle x of {self.name} is {value:.4f} deg, "
                    f"which exceeds the maximum of {self.gimbal_range:.4f} deg.",
                    UserWarning,
                )
        self._gimbal_angle_x = value

    @property
    def gimbal_angle_y(self):
        """Returns the current gimbal angle around the y-axis (yaw)."""
        return self._gimbal_angle_y

    @gimbal_angle_y.setter
    def gimbal_angle_y(self, value):
        # Check if deployment level is within bounds and warn user if not
        if abs(value) > self.gimbal_range:
            if self.clamp:
                value = np.clip(value, -self.gimbal_range, self.gimbal_range)
            else:
                warnings.warn(
                    f"Gimbal angle y of {self.name} is {value:.4f} deg, "
                    f"which exceeds the maximum of {self.gimbal_range:.4f} deg.",
                    UserWarning,
                )
        self._gimbal_angle_y = value

    @property
    def gimbal_angles(self):
        """Returns a tuple of the current gimbal angles (x, y) in degrees."""
        return (self.gimbal_angle_x, self.gimbal_angle_y)

    @gimbal_angles.setter
    def gimbal_angles(self, value):
        """Sets both gimbal angles from a tuple.

        Parameters
        ----------
        value : tuple
            Tuple of (gimbal_angle_x, gimbal_angle_y) in degrees.
        """
        self.gimbal_angle_x = value[0]
        self.gimbal_angle_y = value[1]

    def _reset(self):
        """Resets the TVC system to its initial state. This method is called
        at the beginning of each simulation to ensure the TVC system is in
        the correct state."""
        self.gimbal_angle_x = self.initial_gimbal_angle_x
        self.gimbal_angle_y = self.initial_gimbal_angle_y

    def info(self):
        """Prints summarized information of the TVC system.

        Returns
        -------
        None
        """
        self.prints.geometry()

    def all_info(self):
        """Prints all information of the TVC system.

        Returns
        -------
        None
        """
        self.info()

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        return {
            "gimbal_range": self.gimbal_range,
            "clamp": self.clamp,
            "gimbal_angle_x": self.initial_gimbal_angle_x,
            "gimbal_angle_y": self.initial_gimbal_angle_y,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            gimbal_range=data.get("gimbal_range"),
            clamp=data.get("clamp"),
            gimbal_angle_x=data.get("gimbal_angle_x"),
            gimbal_angle_y=data.get("gimbal_angle_y"),
            name=data.get("name"),
        )
