\\\ N-body simulator ///
Author: Samuel Tickner
Last Updated: 12/12/2025 13:00:00
==================================================================================
This directory contains the python source files for running an N-body simulation. To run, it requires a version of Python 3.8.2 or later. Go to https://www.python.org/downloads/ for downloading python.
The project has the following package dependencies: math, numpy, os, csv, copy, pathlib, matplotlib, astropy, poliastro, spiceypy.
To install these packages, first get the pip installer. Refer to https://pip.pypa.io/en/stable/installation/ for guidance on this.
Then, run the following commands to install any non-default python packages:

    pip install matplotlib
    pip install astropy
    pip install jplephem spiceypy
    pip install https://github.com/poliastro/poliastro/archive/main.zip

    some may already be available on certain installs of environments on anaconda if used.

The Simulation works by using methods in the Simulation class to simulate bodies as Particle class objects. Methods then exist there for plotting. The presets of the simulation call on an instance of the Ephemeris class to gather data from JPL.

In this project directory, there are three class files: Simulation.py, Particle.py, and Ephemeris.py which are imported with:
from {class} import {class}

There are two simulation preset files: bd_earthsat.py and solarsystem.py.

There is also an error_calculation.py file containing functions for error calculation.

If you intend to run the functions yourself, instructions will follow below. If you do not, and would like to try the simple input UI, run the input.py file for guidance.

To run your own function code, I would recommend creating a new file nBody.py and entering lines in there.
All functions/methods and classes in all files have been given docstrings. Calling them with {class or function}.__doc__ will print out the docstring to tell you what the function does and what its parameters are.

A set of example code might look something like:

    from Simulation import Simulation # you must import Simulation class!!!

    nBody = Simulation( # create an instance of the class with its parameters according to Simulation.__init__.__doc__.
            sim = 'earthsat',
            integrator = '2',
            iterations = 1000,
            timestep = 100,
            saving = True
        )

    nBody.initSim() # run simulation.
    nBody.masterView() # plot master plot.
    nBody.plot_relative_energy_error() # plot relative energy error.

The full list of functions in the Simulation class file: [
    'checkData', 
    'customEnergyCalc', 
    'eccentricityEarthPlot', 
    'energyCalc', 
    'ephOrbitalPeriods', 
    'getBodyPositions', 
    'getEccentricities', 
    'groupedREE', 
    'importBodies', 
    'initSim', 
    'masterView', 
    'plot_relative_energy_error', 
    'saveAngularMomentaCSV', 
    'saveBodyPositionsCSV', 
    'saveEnergiesCSV', 
    'saveLinearMomentaCSV', 
    'seconds_to_dhms', 
    'totalAngularMomenta', 
    'totalLinearMomenta', 
    'updateIntegrator'
    ]
    printable with command:
    methods_list = [method for method in dir(Simulation) if callable(getattr(Simulation, method)) and not method.startswith("__")]
    print(f'Methods using dir():{methods_list}')

If using any of the functions in error_calculation.py, please work within that file only.

Any figures generated are saved to figures/
Any csv files are saved to data/csv while .npy files will remain in data/

General Notes:
- If you try to generate a plot without running a simulation, there are exceptions in place to run a simulation if the target file is not found.
- If you delete any of the folders by accident do not worry! The Simulation class file makes directories if they do not exist.