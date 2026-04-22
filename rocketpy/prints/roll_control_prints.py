class _RollControlPrints:
    """Class that contains all Roll Control prints."""

    def __init__(self, roll_control):
        """Initializes _RollControlPrints class

        Parameters
        ----------
        roll_control: rocketpy.RollControl
            Instance of the RollControl class.

        Returns
        -------
        None
        """
        self.roll_control = roll_control

    def basics(self):
        """Prints information of the Roll Control system."""
        print("Information of the Roll Control System:")
        print("----------------------------------")
        print(f"Maximum Roll Torque: {self.roll_control.max_roll_torque:.2f} N·m")
        print(f"Current Roll Torque: {self.roll_control.roll_torque:.2f} N·m")
        print(
            f"Torque Clamping: {'Enabled' if self.roll_control.clamp else 'Disabled'}"
        )

    def all(self):
        """Prints all information of the Roll Control system."""
        self.basics()
