import warnings
import numpy as np
from ..prints.throttle_control_prints import _ThrottleControlPrints

class ThrottleControl:
    def __init__(
        self,
        min_throttle=0.0,
        max_throttle=1.0,
        initial_throttle=1.0,
        clamp=True,
        name="Throttle Control",
    ):
        if min_throttle > max_throttle:
            raise ValueError("min_throttle must be <= max_throttle")

        self.min_throttle = float(min_throttle)
        self.max_throttle = float(max_throttle)
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

        if value < self.min_throttle or value > self.max_throttle:
            if self.clamp:
                value = np.clip(value, self.min_throttle, self.max_throttle)
            else:
                warnings.warn(
                    f"Throttle of {self.name} is {value:.4f}, "
                    f"which exceeds bounds "
                    f"[{self.min_throttle:.4f}, {self.max_throttle:.4f}].",
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
            f"range=[{self.min_throttle:.3f}, {self.max_throttle:.3f}])>"
        )
