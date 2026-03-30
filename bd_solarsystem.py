### IMPORTING PACKAGES AND CLASSES ###
# NumPy and Regex packages - array and data tools.
import numpy as np
# Particle class from the Particle.py file.
from Particle import Particle
from Ephemeris import Ephemeris
# Astropy and other conversion tools.
from astropy.time import Time

bodies = [] # Creates an empty list to contain all of the n interacting bodies in the simulation.

solarsystem = Ephemeris(
        time = Time("2025-12-12 17:00:00.0", scale="tdb"), # Get the time at 5pm on 12th Dec 2025.
        combined = {"earth-moon-barycenter": ("earth", "moon")}
    )

# Bodies #
targets = ['sun','mercury','venus','earth-moon-barycenter','mars','jupiter','saturn','uranus','neptune','pluto']

# Get the list of all files and directories
colour_map = {
    'sun': 'gold',
    'mercury': 'olive',
    'venus': 'khaki',
    'earth-moon-barycenter': 'royalblue',
    'mars': 'firebrick',
    'jupiter': 'chocolate',
    'saturn': 'sandybrown',
    'uranus': 'skyblue',
    'neptune': 'navy',
    'pluto': 'salmon'
}

dtyping = np.float32

for target_body in targets:
    body = Particle( # Defining properties of the Earth.
        position = np.array(solarsystem.getPosVel(target_body)[0], dtype=dtyping),
        velocity = np.array(solarsystem.getPosVel(target_body)[1], dtype=dtyping),
        acceleration = np.array([0, 0, 0], dtype=dtyping),
        name = target_body,
        colour = colour_map.get(target_body.lower(), "white"),
        mass = solarsystem.get_mass(target_body)
    )
    bodies.append(body)

Bodies = bodies

#t = Time("2026-04-07 10:46:40.0", scale="tdb") after (115, 17, 46, 40) d, h, m, s to the date Tuesday, 7 April 2026, 10:46:40
#print(getPosVel('earth-moon-barycenter')) # (array([-1.43350961e+11, -4.52653715e+10,  2.11454876e+07]), array([ 8.37019590e+03, -2.85557660e+04,  1.42531332e+00]))