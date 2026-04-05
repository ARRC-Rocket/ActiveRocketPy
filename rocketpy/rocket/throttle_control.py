import warnings
import numpy as np
from ..prints.throttle_control_prints import _ThrottleControlPrints

class ThrottleControl:
    def __init__(
        self,
        throttle_range=(0.0, 1.0),
        initial_throttle=1.0,
        clamp=True,
        sampling_rate=100,
        throttle_rate_limit=0,
        name="Throttle Control",
    ):
        if throttle_range[0] > throttle_range[1]:
            raise ValueError("throttle_range[0] must be <= throttle_range[1]")
        self.throttle_range = throttle_range
        self.initial_throttle = float(initial_throttle)
        self.clamp = clamp
        self.sampling_rate = sampling_rate
        assert throttle_rate_limit >= 0, "throttle_rate_limit must be non-negative."
        self.throttle_rate_limit = throttle_rate_limit
        self.name = name

        self._throttle = None
        self.throttle_prev = initial_throttle
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

        # Limit the throttle rate
        max_throttle_change = self.throttle_rate_limit / self.sampling_rate
        throttle_change = value - self.throttle_prev
        if abs(throttle_change) > max_throttle_change:
            value = self.throttle_prev + np.sign(throttle_change) * max_throttle_change
        self.throttle_prev = value

        self._throttle = float(value)

    def _reset(self):
        self.throttle = self.initial_throttle
        self.throttle_prev = self.initial_throttle

    def __repr__(self):
        return (
            f"<ThrottleControl("
            f"name={self.name}, "
            f"throttle={self.throttle:.3f}, "
            f"range=[{self.throttle_range[0]:.3f}, {self.throttle_range[1]:.3f}])>"
        )

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        return {
            "throttle_range": self.throttle_range,
            "initial_throttle": self.initial_throttle,
            "clamp": self.clamp,
            "sampling_rate": self.sampling_rate,
            "throttle_rate_limit": self.throttle_rate_limit,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            throttle_range=data.get("throttle_range"),
            initial_throttle=data.get("initial_throttle"),
            clamp=data.get("clamp"),
            sampling_rate=data.get("sampling_rate"),
            throttle_rate_limit=data.get("throttle_rate_limit"),
            name=data.get("name"),
        )
