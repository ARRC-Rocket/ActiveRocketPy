import warnings
import numpy as np
from ..prints.throttle_control_prints import _ThrottleControlPrints

class ThrottleControl:
    def __init__(
        self,
        throttle_range=(0.0, 1.0),
        initial_throttle=1.0,
        clamp=True,
        name="Throttle Control",
    ):
        if throttle_range[0] > throttle_range[1]:
            raise ValueError("throttle_range[0] must be <= throttle_range[1]")
        self.throttle_range = throttle_range
        self.initial_throttle = float(initial_throttle)
        self.clamp = clamp
        self.name = name

        self._throttle = None
        self.throttle = initial_throttle

    @property
    def throttle(self):
        return self._throttle

    @throttle.setter
    def throttle(self, value):
        value = float(value)

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

        self._throttle = float(value)

    def _reset(self):
        self.throttle = self.initial_throttle

    def __repr__(self):
        return (
            f"<ThrottleControl("
            f"name={self.name}, "
            f"throttle={self.throttle:.3f}, "
            f"range=[{self.throttle_range[0]:.3f}, {self.throttle_range[1]:.3f}])>"
        )
