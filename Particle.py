import numpy as np

class Particle():
    """
    Particle() is a body object that contains its vector and constant properties. It also is the conatiner for the integrators called when the simulation is run.
    Check the docstring for __init__() for further information on class attributes.
    """
    def __init__(
        self,
        position=np.array([0.0, 0.0, 0.0], dtype=float), # position vector as ndarray
        velocity=np.array([0.0, 0.0, 0.0], dtype=float), # velocity vector as ndarray
        acceleration=np.array([0.0, -10.0, 0.0], dtype=float), # acceleration vector as ndarray
        name='Particle', # name
        colour='black', # colour
        mass=1.0, # mass
        G=6.67408E-11 # Newton's gravitational constant G
    ):
        """
        Class initialisation function that runs on calling the class. It defines each of the properties of the class so they are able to be referenced in each of the following class methods.
        ///////////
        Parameters
        ----------
        self: class
            Particle()
        position: numpy array
            The position vector of the particle.
        velocity: numpy array
            The velocity vector of the particle.
        acceleration: numpy array
            The acceleration vector of the particle.
        name: string
            The name assigned to the particle.
        colour: string
            The colour assigned to the particle.
        mass: float
            The mass of the particle.
        G: float
            Newton's constant of gravitation, G.
        """
        self.position=np.array([position[0], position[1], position[2]], dtype=float)
        self.velocity=np.array([velocity[0], velocity[1], velocity[2]], dtype=float)
        self.acceleration=np.array([acceleration[0], acceleration[1], acceleration[2]], dtype=float)
        self.name=name
        self.colour=colour
        self.mass=float(mass)
        self.G=float(G)

    def __str__(self):
        """
        Returns the Particle data attributes in a formatted string.
        ///////////
        Parameters
        ----------
        self: class
            Particle()
        ////////
        Returns
        --------
        attributes: str
            A formatted string of all the attributes of the class.
        """
        return "Position: {0}, Velocity: {1}, Acceleration: {2}, Name: {3}, Colour: {4}, Mass: {5:.3e}, Gravitational constant G: {6}".format(
            self.position, self.velocity, self.acceleration, self.name, self.colour, self.mass, self.G
        )
    
    def updateEu(self,deltaT):
        """
        Updates the positions and velocities of the particle using the Euler method.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        deltaT: float
            The change in time between each measurement of the particle.
        """
        self.position=np.add(self.position, self.velocity * deltaT)
        self.velocity=np.add(self.velocity, self.acceleration * deltaT)

    def updateEuC(self,deltaT):
        """
        Updates the positions and velocities of the particle using the Euler-Cromer method.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        deltaT: float
            The change in time between each measurement of the particle.
        """
        self.velocity=np.add(self.velocity, self.acceleration * deltaT)
        self.position=np.add(self.position, self.velocity * deltaT)

    def updateEuR(self,deltaT,bodies):
        """
        Updates the positions and velocities of the particle using the Euler-Richardson method.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        deltaT: float
            The change in time between each measurement of the particle.
        bodies: list of Particle objects
            The N bodies of the simulation.
        """
        #self.updateGravitationalAcceleration(bodies) # calculate initial acceleration (done in the simulation).
        vmid=np.add(self.velocity, 0.5 * self.acceleration * deltaT)
        self.updateGravitationalAcceleration(bodies) # update acceleration at the midpoint.
        self.velocity=np.add(self.velocity, self.acceleration * deltaT)
        self.position=np.add(self.position, vmid * deltaT)

    def updateVerlet(self,deltaT,bodies):
        """
        Updates the positions and velocities of the particle using the Verlet method.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        deltaT: float
            The change in time between each measurement of the particle.
        bodies: list of Particle objects
            The N bodies of the simulation.
        """
        #self.updateGravitationalAcceleration(bodies) # calculate initial acceleration (done in the simulation).
        self.position = np.add(self.position, self.velocity * deltaT, 0.5 * self.acceleration * (deltaT**2)) # update the position.
        self.velocity=np.add(self.velocity, 0.5 * self.acceleration * deltaT) # update velocity half way by adding 1/2 * accleration * timestep.
        self.updateGravitationalAcceleration(bodies) # calculate middle position's acceleration
        self.velocity=np.add(self.velocity, 0.5 * self.acceleration * deltaT) # get the final velocity using another half step.

    def updateRK4(self,deltaT,bodies):
        """
        Updates the positions and velocities of the particle using the Runge-Kutta-4 method.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        deltaT: float
            The change in time between each measurement of the particle.
        bodies: list of Particle objects
            The N bodies of the simulation.
        """
        pos1 = self.position
        vel1 = self.velocity
        acc1 = self.acceleration # updated by the simulation so it is != 0
        pos2 = np.add(pos1, 0.5 * vel1 * deltaT) # add 1/2 a step
        self.position = pos2
        vel2 = np.add(vel1, 0.5 * acc1 * deltaT)
        self.updateGravitationalAcceleration(bodies)
        acc2 = self.acceleration
        pos3 = np.add(pos1, 0.5 * vel2 * deltaT) # add 1/2 a step with second velocity
        self.position = pos3
        vel3 = np.add(vel1, 0.5 * acc2 * deltaT)
        self.updateGravitationalAcceleration(bodies)
        acc3 = self.acceleration
        pos4 = np.add(pos1, vel3 * deltaT) # add a full step with third velocity
        self.position = pos4
        vel4 = np.add(vel1, acc3 * deltaT)
        self.updateGravitationalAcceleration(bodies)
        acc4 = self.acceleration
        self.position = pos1 + 1/6 * deltaT * (vel1 + 2 * vel2 + 2 * vel3 + vel4)
        self.velocity = vel1 + 1/6 * deltaT * (acc1 + 2 * acc2 + 2 * acc3 + acc4)

    def updateGravitationalAcceleration(self,bodies):
        """
        Updates the total gravitational acceleration of the particle towards other bodies by calculating a vector sum of individual accelerations.

        Parameters
        ----------
        self: class
            Simulation()
        bodies: list of Particle objects
            The N bodies of the simulation.
        """
        self.acceleration = 0
        for body in bodies:
            if body == self:
                continue
            else:
                delta_r = np.subtract(self.position, body.position)
                self.acceleration += -np.divide((self.G * body.mass * delta_r) , np.abs(np.linalg.norm(delta_r))**3)

    def kineticEnergy(self):
        """
        Returns the kinetic energy of the particle.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        ////////
        Returns
        -------
        Kinetic energy: float
        """
        return 0.5 * self.mass * np.linalg.norm(self.velocity)**2
    
    def potentialEnergy(self, body):
        """
        Returns the potential energy of the particle above another body.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        body: object
            The body the particle has a gravitational potential towards.
        ////////
        Returns
        -------
        Potential energy: float
        """
        delta_r = np.subtract(self.position, body.position)
        return -(self.G * self.mass * body.mass)/(np.abs(np.linalg.norm(delta_r)))
    
    def getMomentum(self):
        """
        Returns the linear momentum of the particle.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        ////////
        Returns
        -------
        linear momentum: np.ndarray
        """
        linear_momentum = self.mass * self.velocity
        return linear_momentum
    
    def angularMomentumToOrigin(self):
        """
        Returns the angular momentum of the particle.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        ////////
        Returns
        -------
        angular momentum: np.ndarray
        """
        angular_momentum = self.mass * np.cross(self.position, self.velocity)
        return angular_momentum
    
    def eop(self, sun):
        """
        Returns the eccentricity and orbital period of the particle.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        sun: Particle object
            The sun.
        ////////
        Returns
        -------
        eccentricity: float
        orbital periof: float
        """
        r = np.subtract(self.position, sun.position)
        v = np.subtract(self.velocity, sun.velocity)
        mu = self.G * sun.mass
        eccentricity = np.linalg.norm((np.cross(v, np.cross(r, v)) / mu) - (r / np.linalg.norm(r)))
        # Specific orbital energy epsilon
        epsilon = np.linalg.norm(v)**2 / 2 - mu / np.linalg.norm(r)
        # Semi-major axis
        sma = -mu / (2 * epsilon)
        # Periapsis and apoapsis
        #r_peri = a * (1 - eccentricity)
        #r_apo = a * (1 + eccentricity)
        # Orbital period
        orbital_period = 2 * np.pi * np.sqrt(sma**3 / mu)
        return eccentricity, orbital_period