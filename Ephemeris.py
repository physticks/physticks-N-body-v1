### IMPORTING PACKAGES AND CLASSES ###
# NumPy and Regex packages - array and data tools.
import numpy as np
# Astropy and other conversion tools.
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
from astropy.constants import G # Newton's gravitational constant
from poliastro import constants
from spiceypy import sxform, mxvg

class Ephemeris():
    """
    Ephemeris() contains two functions for retrieving positions, velocities and masses from the JPL ephemeris database.
    Check the docstring for __init__() for further information on class attributes.
    """
    def __init__(
        self,
        time = Time("2025-12-12 17:00:00.0", scale="tdb"),
        combined = {"earth-moon-barycenter": ("earth", "moon")}
    ):
        """
        Class initialisation function that runs on calling the class. It defines each of the properties of the class so they are able to be referenced in each of the following class methods.
        ///////////
        Parameters
        ----------
        self: class
            Ephemeris()
        G: float
            Newton's constant of gravitation, G.
        """
        self.time = time
        self.combined = combined
        
    def getPosVel(self, target_body: str):
        """
        Finds the position and velocity in the ephemeris database using the astropy package.
        ///////////
        Parameters
        ----------
        self: class
            Ephemeris()
        target_body: str
            The name of the body you are trying to retreive data for.
        ////////
        Returns
        -------
        position, velocity: tuple
        """
        pos, vel = get_body_barycentric_posvel(f"{target_body}", self.time, ephemeris="jpl")
        # Make a "state vector" of positions and velocities (in metres and metres/second, respectively).
        statevec = [
            pos.xyz[0].to("m").value,
            pos.xyz[1].to("m").value,
            pos.xyz[2].to("m").value,
            vel.xyz[0].to("m/s").value,
            vel.xyz[1].to("m/s").value,
            vel.xyz[2].to("m/s").value,
        ]
        # Get transformation matrix to the ecliptic (use time in Julian Days).
        trans = sxform("J2000", "ECLIPJ2000", self.time.jd)
        # Transform state vector to ecliptic.
        statevececl = mxvg(trans, statevec)
        position = np.array([statevececl[0], statevececl[1], statevececl[2]])
        velocity = np.array([statevececl[3], statevececl[4], statevececl[5]])
        return (position, velocity)

    def get_mass(self, body_name: str):
        """
        Finds the mass of a chosen body in the ephemeris database using the astropy package.
        ///////////
        Parameters
        ----------
        self: class
            Ephemeris()
        target_body: str
            The name of the body you are trying to retreive data for.
        ////////
        Returns
        -------
        mass: float
        """
        name = body_name.lower()
        # If it is a combined multiple body barycenter:
        if name in self.combined:
            total_GM = sum(
                getattr(constants, f"GM_{n}") for n in self.combined[name]
            )
            return (total_GM / G).value
        # Normal single body:
        try:
            GM = getattr(constants, f"GM_{name}")
        except AttributeError:
            raise ValueError(f"No GM constant found for '{body_name}'")
        return (GM / G).value
    
    def getCustom(self, target_body: str):
        
        pass