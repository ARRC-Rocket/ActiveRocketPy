class _TVCPrints:
    """Class that contains all TVC prints."""

    def __init__(self, tvc):
        """Initializes _TVCPrints class

        Parameters
        ----------
        tvc: rocketpy.TVC
            Instance of the TVC class.

        Returns
        -------
        None
        """
        self.tvc = tvc

    def geometry(self):
        """Prints geometric information of the TVC system."""
        print("Geometric information of the TVC System:")
        print("----------------------------------")
        print(f"Maximum Gimbal Angle: {self.tvc.gimbal_range:.2f}°")
        print(f"Current Gimbal Angle X: {self.tvc.gimbal_angle_x:.2f}°")
        print(f"Current Gimbal Angle Y: {self.tvc.gimbal_angle_y:.2f}°")
        print(f"Angle Clamping: {'Enabled' if self.tvc.clamp else 'Disabled'}")

    def all(self):
        """Prints all information of the TVC system."""
        self.geometry()
