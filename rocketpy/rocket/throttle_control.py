import warnings

import numpy as np

from ..prints.throttle_control_prints import _ThrottleControlPrints


class ThrottleControl:


    def __init__(
        self,
        min_throttle=0.0,
        max_throttle=1.0,
        clamp=True,
        throttle=0.0,
        name="Throttle Control",
    ):

        self.name = name
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.clamp = clamp
        self.initial_throttle = throttle
        self.throttle = throttle
        self.prints = _ThrottleControlPrints(self)

    @property
    def throttle(self):
        return self._throttle

    @throttle.setter
    def throttle(self, value):
        if value < self.min_throttle or value > self.max_throttle:
            if self.clamp:
                value = np.clip(value, self.min_throttle, self.max_throttle)
            else:
                warnings.warn(
                    #f"Roll torque of {self.name} is {value:.4f} N·m, "
                    #f"which exceeds the maximum of {self.max_throttle:.4f} N·m.",
                    UserWarning,
                )
        self._throttle = float(value)

    def _reset(self):
        self.throttle = self.initial_throttle

    def info(self):
        self.prints.basics()

    def all_info(self):
        self.info()

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        return {
            "min_throttle":self.min_throttle,
            "max_throttle": self.max_throttle,
            "clamp": self.clamp,
            "throttle": self.initial_throttle,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            min_throttle=data.get("min_throttle"),
            max_throttle=data.get("max_throttle"),
            clamp=data.get("clamp"),
            throttle=data.get("throttle"),
            name=data.get("name"),
        )
