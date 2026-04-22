# ActiveRocketPy

ActiveRocketPy is the enhanced version of [RocketPy](https://github.com/RocketPy-Team/RocketPy), a powerful Python package for simulating the trajectories of high-power rockets. This fork introduces a range of active control and guidance, navigation, and control (GNC) features.

## Main features (ActiveRocketPy)
1. **Thrust Vector Control (TVC)**
   - Implementation of TVC control class
   - Feed X & Y gimbal angles through a control function
   - Actuator dynamics and limits for realistic TVC simulations (WIP)

2. **Roll Control**
   - Implementation of roll control class
   - Feed ideal roll torque through a control function
   - Actuator dynamics and limits for realistic roll control simulations (WIP)

3. **Throttle Control**
   - Implementation of throttle control class
   - Feed throttle percentage through a control function
   - Actuator dynamics and limits for realistic throttle control simulations (WIP)

4. **Step simulation**
    - Step through the simulation one time step at a time
    - Update control inputs and step simulations in loop

## Main features (RocketPy)

1. **Nonlinear 6 Degrees of Freedom Simulations**
   - Rigorous treatment of mass variation effects
   - Efficiently solved using LSODA with adjustable error tolerances
   - Highly optimized for fast performance

2. **Accurate Weather Modeling**
   - Supports International Standard Atmosphere (1976)
   - Custom atmospheric profiles and Soundings (Wyoming)
   - Weather forecasts, reanalysis, and ensembles for realistic scenarios

3. **Aerodynamic Models**
   - Optional Barrowman equations for lift coefficients
   - Easy import of drag coefficients from other sources (e.g., CFD simulations)

4. **Parachutes with External Trigger Functions**
   - Test the exact code that will fly
   - Sensor data augmentation with noise for comprehensive parachute simulations

5. **Solid, Hybrid, and Liquid Motors Models**
   - Burn rate and mass variation properties from the thrust curve
   - Define custom rocket tanks based on flux data
   - Support for CSV and ENG file formats

6. **Monte Carlo Simulations**
   - Conduct dispersion analysis and global sensitivity analysis

7. **Flexible and Modular**
   - Perform straightforward engineering analysis (e.g., apogee and lift-off speed as a function of mass)
   - Handle non-standard flights (e.g., parachute drop test from a helicopter)
   - Support multi-stage rockets and custom continuous/discrete control laws
   - Easily create new classes, such as other types of motors

8. **Integration with MATLAB®**
   - Effortlessly run RocketPy from MATLAB®
   - Convert RocketPy results to MATLAB® variables for further processing

These powerful features make RocketPy an indispensable tool for high-power rocket trajectory simulation, catering to enthusiasts, researchers, and engineers in the field of rocketry.

## Validation

> WIP

# Documentation

Check out documentation details using the links below:

- [RocketPy User Guide](https://docs.rocketpy.org/en/latest/user/index.html)
- [RocketPy Code Documentation](https://docs.rocketpy.org/en/latest/reference/index.html)
- [RocketPy Development Guide](https://docs.rocketpy.org/en/latest/development/index.html)
- [RocketPy Technical Documentation](https://docs.rocketpy.org/en/latest/technical/index.html)
- [RocketPy Flight Examples](https://docs.rocketpy.org/en/latest/examples/index.html)


<br>

# Getting Started

## Quick Installation

To install ActiveRocketPy, run the following commands in your terminal:

```shell
git clone https://github.com/<Your-GitHub-Account>/ActiveRocketPy.git
cd ActiveRocketPy
pip install -e .  # install the ActiveRocketPy lib in editable mode
pip install -r requirements-optional.txt  # install optional requirements
pip install -r requirements-tests.txt  # install test/dev requirements
```

# Authors and Contributors

> RocketPy was originally created by [Giovani Ceotto](https://github.com/giovaniceotto/) as part of his work at [Projeto Jupiter](https://github.com/Projeto-Jupiter/). [Rodrigo Schmitt](https://github.com/rodrigo-schmitt/) was one of the first contributors. Later, [Guilherme Fernandes](https://github.com/Gui-FernandesBR/) and [Lucas Azevedo](https://github.com/lucasfourier/) joined the team to work on the expansion and sustainability of this project.
> Since then, the [RocketPy Team](https://github.com/orgs/RocketPy-Team/teams/rocketpy-team) has been growing fast and our contributors are what makes us special!

ActiveRocketPy is forked and maintained by [ZuoRen Chen](https://github.com/zuorenchen), along with the team from [Advanced Rocket Research Center (ARRC)](https://github.com/ARRC-Rocket).


## Citation

If you run ActiveRocketPy in your research, please consider citing:

```bibtex
@misc{ActiveRocketPy,
  author = {Zuo-Ren Chen and Advanced Rocket Research Center (ARRC)},
  title = {ActiveRocketPy: A 6-DoF Rocket GNC Simulator},
  month = {April},
  year = {2026},
  url = {https://github.com/ARRC-Rocket/ActiveRocketPy}
}
```
To cite RocketPy, please check its repository for the latest citation information: [RocketPy](https://github.com/RocketPy-Team/RocketPy)