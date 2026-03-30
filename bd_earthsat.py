### IMPORTING PACKAGES AND CLASSES ###
# NumPy and Regex packages - array and data tools.
import numpy as np
# Particle class from the Particle.py file.
from Particle import Particle

### DEFINING CONSTANTS ###
AU = 149597870700 # Defines the length of an Atronomical Unit.
earthMass = 5.97237e24     # https://en.wikipedia.org/wiki/Earth # Sets the Earth Mass.
earthRadius = 63710 * 1e3  # https://en.wikipedia.org/wiki/Earth # Sets the Earth Radius.

### SETTING BODY PROPERTIES ###

bodies = [] # Creates an empty list to contain all of the n interacting bodies in the simulation.

# Example Body #
# Body = Particle( # Defining properties of the Earth.
    # position=np.array([0, 0, 0]), -- the position vector of the body in Cartesian coordinates.
    # velocity=np.array([0, 0, 0]), -- the velocity vector of the body in Cartesian coordinates.
    # acceleration=np.array([0, 0, 0]), -- the acceleration vector of the body in Cartesian coordinates.
    # name="Name", -- the name of the body.
    # colour = 'red' -- the colour of the body
    # mass=bodyMass -- the mass of the body.
# )

Earth = Particle( # Defining properties of the Earth.
    position=np.array([0, 0, 0]),
    velocity=np.array([0, 0, 0]),
    acceleration=np.array([0, 0, 0]),
    name="Earth",
    colour='royalblue',
    mass=earthMass
)
bodies.append(Earth) # Adds Earth to the bodies list.

satPosition = earthRadius + (35786 * 1e3)
satVelocity = np.sqrt(Earth.G * Earth.mass / satPosition)  # From centrifugal force = Gravitational force.
Satellite = Particle( # Defining properties of the Satellite.
    position=np.array([satPosition, 0, 0]),
    velocity=np.array([0, satVelocity, 0]),
    acceleration=np.array([0, 0, 0]),
    name="Satellite",
    colour='gold',
    mass=100.
)
bodies.append(Satellite) # Adds the satellite to the bodies list.

natSat = Particle(
    position=np.array([-satPosition, 800000, 67129]),
    velocity=np.array([305, 1.3*np.sqrt(Earth.G * Earth.mass / satPosition), -24]),
    acceleration=np.array([0, 0, 0]),
    name="long-range-satellite",
    colour='salmon',
    mass=345000.
)
bodies.append(natSat) # Adds this new satellite to the bodies list.

Bodies = bodies