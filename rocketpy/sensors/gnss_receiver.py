import math

import numpy as np

from rocketpy.tools import inverted_haversine

from ..mathutils.vector_matrix import Matrix, Vector
from ..prints.sensors_prints import _GnssReceiverPrints
from .sensor import ScalarSensor


class GnssReceiver(ScalarSensor):
    """Class for the GNSS Receiver sensor.

    Attributes
    ----------
    prints : _GnssReceiverPrints
        Object that contains the print functions for the sensor.
    sampling_rate : float
        Sample rate of the sensor in Hz.
    position_accuracy : float
        Accuracy of the sensor interpreted as the standard deviation of the
        position in meters.
    altitude_accuracy : float
        Accuracy of the sensor interpreted as the standard deviation of the
        position in meters.
    velocity_accuracy : float
        Accuracy of the sensor interpreted as the standard deviation of the
        velocity in meters per second.
    name : str
        The name of the sensor.
    measurement : tuple
        The measurement of the sensor.
    measured_data : list
        The stored measured data of the sensor.
    """

    units = "°, m"

    def __init__(
        self,
        sampling_rate,
        position_accuracy=0,
        altitude_accuracy=0,
        velocity_accuracy=0,
        name="GnssReceiver",
    ):
        """Initialize the Gnss Receiver sensor.

        Parameters
        ----------
        sampling_rate : float
            Sample rate of the sensor in Hz.
        position_accuracy : float
            Accuracy of the sensor interpreted as the standard deviation of the
            position in meters. Default is 0.
        altitude_accuracy : float
            Accuracy of the sensor interpreted as the standard deviation of the
            position in meters. Default is 0.
        velocity_accuracy : float
            Accuracy of the sensor interpreted as the standard deviation of the
            velocity in meters per second. Default is 0.
        name : str
            The name of the sensor. Default is "GnssReceiver".
        """
        super().__init__(sampling_rate=sampling_rate, name=name)
        self.position_accuracy = position_accuracy
        self.altitude_accuracy = altitude_accuracy
        self.velocity_accuracy = velocity_accuracy

        self.prints = _GnssReceiverPrints(self)

    def measure(self, time, **kwargs):
        """Measure the position of the rocket in latitude, longitude and
        altitude.

        Parameters
        ----------
        time : float
            Current time in seconds.
        kwargs : dict
            Keyword arguments dictionary containing the following keys:

            - u : np.array
                State vector of the rocket.
            - u_dot : np.array
                Derivative of the state vector of the rocket.
            - relative_position : np.array
                Position of the sensor relative to the rocket center of mass.
            - environment : Environment
                Environment object containing the atmospheric conditions.
        """
        u = kwargs["u"]
        relative_position = kwargs["relative_position"]

        # Get from state u and add relative position
        x, y, z = (Matrix.transformation(u[6:10]) @ relative_position) + Vector(u[0:3])
        vx, vy, vz =  (Matrix.transformation(u[6:10]) @ Vector.cross(Vector(u[10:13]), relative_position)) + Vector(u[3:6])

        # Apply accuracy to the position
        x = np.random.normal(x, self.position_accuracy)
        y = np.random.normal(y, self.position_accuracy)
        z = np.random.normal(z, self.altitude_accuracy)
        vx = np.random.normal(vx, self.velocity_accuracy)
        vy = np.random.normal(vy, self.velocity_accuracy)
        vz = np.random.normal(vz, self.velocity_accuracy)

        self.measurement = (x, y, z, vx, vy, vz)
        self._save_data((time, *self.measurement))

    def export_measured_data(self, filename, file_format="csv"):
        """Export the measured values to a file

        Parameters
        ----------
        filename : str
            Name of the file to export the values to
        file_format : str
            Format of the file to export the values to. Options are "csv" and
            "json". Default is "csv".

        Returns
        -------
        None
        """
        self._generic_export_measured_data(
            filename=filename,
            file_format=file_format,
            data_labels=("t", "x", "y", "z", "vx", "vy", "vz"),
        )

    def to_dict(self, **kwargs):
        return {
            "sampling_rate": self.sampling_rate,
            "position_accuracy": self.position_accuracy,
            "altitude_accuracy": self.altitude_accuracy,
            "velocity_accuracy": self.velocity_accuracy,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            sampling_rate=data["sampling_rate"],
            position_accuracy=data["position_accuracy"],
            altitude_accuracy=data["altitude_accuracy"],
            name=data["name"],
        )
