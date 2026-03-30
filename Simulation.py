##### IMPORTING PACKAGES AND CLASSES #####
# Data handling imports - numpy for arrays and math and runtime for features:
import math
import numpy as np
import time as runtime
# File handling imports - filepath-ing and copying:
import os
import csv
import copy
from pathlib import Path
# Matplotlib package imports - graphs and animations:
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.animation as anim
from matplotlib import image as mpimg
#//////////////////////////////
##### DEFINING CONSTANTS #####
G=6.67408E-11
AU = 149597870700 # Defines the length of an Atronomical Unit.
earthMass = 5.97237e24     # https://en.wikipedia.org/wiki/Earth # Sets the Earth Mass.
earthRadius = 63710000  # https://en.wikipedia.org/wiki/Earth # Sets the Earth Radius.
solMass = 1.98847e30
solRadius = 695700000
mu = G * solMass
#////////////////////////////////////
# Ensure these following folders exist:
os.makedirs('data', exist_ok=True)
os.makedirs('data/csv', exist_ok=True)
os.makedirs('figures', exist_ok=True)
#////////////////////////////////////
##### SETTING SIMULATION CLASS #####
class Simulation():
    """
    Simulation() contains functions for running an N-body simulation and plotting the data that is saved to .npy files. There are also functions that optionally save plot data to .csv files.
    Check the docstring for __init__() for further information on class attributes.
    """
    ##### CLASS FUNCTIONS #####
    def __init__(
        self,
        sim = 'earthsat', # simulation preset.
        integrator = 1,
        iterations = 2000, # number of iterations that the simulations calculates data for.
        timestep = 6.0, # the timestep of the simulation between frames and kinematic calculations.
        saving = True, # boolean to turn saving of data on or off.
        bodies = []
    ):
        """
        Class initialisation function that runs on calling the class. It defines each of the properties of the class so they are able to be referenced in each of the following class methods.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        sim: string
            The chosen simulation preset (the default is 'earthsat').
        integrator: int, (1, 2, 3, 4 or 5)
            An integer code between 1 and 5 for selecting the chosen integrator.
            1 = Euler, 2 = Euler-Cromer, 3 = Euler-Richardson, 4 = Velocity Verlet, 5 = Runge-Kutta-4.
        iterations: int, minimum value = 100
            The number of iterations that the simulations calculates data for. 
        timestep: float, any real number
            The timestep of the simulation between frames and kinematic calculations.
        saving: boolean
            A boolean to turn saving of plot data on or off. If True, it will save all of the simulation data to one .npy file. If off, it will print it out to be inspected.
        """
        self.sim = str(sim)
        self.integrator = int(integrator)
        self.itns = int(iterations)
        self.timestep = float(timestep)
        self.saving = bool(saving)
        self.bodies = bodies
    
    def __str__(self):
        """
        Returns the Simulation data attributes in a formatted string.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        """
        return "Simulation preset = {0}, Chosen integrator: {1} Number of iterations = {2}, Timestep = {3}, Saving enabled = {3}, Bodies = {4}".format(
            self.sim, self.integrator, self.itns, self.timestep, self.saving, self.bodies
        )
    #//////////////////////////////////////
    ##### IMPORTS AND DATA FUNCTIONS #####
    def importBodies(self):
        """
        Imports a list of bodies to overwrite the empty list in self.bodies from a preset file according to self.sim.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        """
        if self.sim == 'earthsat':
            from bd_earthsat import Bodies # match preset to preset file
            self.bodies = Bodies
        if self.sim == 'solarsystem':
            from bd_solarsystem import Bodies
            self.bodies = Bodies
        if self.sim == 'custom':
            from bd_custom import Bodies
            self.bodies = Bodies

    def checkData(self):
        """
        Checks the validity of the filepath specified by the following f-string:
        f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy"
        If the file is not found to exist, it will run a fresh simulation based on the entered parameters.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        /////////////////////
        Saved / Loaded Files
        --------------------
        datafile = Path(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy")
        """
        datafile = Path(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy")
        if datafile.exists(): # if there is a data file already, then load the data:
            print('Data file found, loading simulation...')
            #DataIn = np.load(f"NBodyTest_{self.sim}.npy", allow_pickle=True)
            #bodyData = DataIn.T[1:]
        else: # run a fresh simulation, then load the data from it:
            print('Data file not found, running a new simulation...')
            self.initSim()

    def getBodyPositions(self):
        """
        From a saved datafile, the positions of each body are extracted and put into a numpy array of shape (len(self.bodies), 3, self.itns / 100).
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        /////////////////////
        Saved / Loaded Files
        --------------------
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        ////////
        Returns
        --------
        body positions: np.ndarray
            An array of the positions of all the bodies at time t in the simulation.
        """
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        bodyData = DataIn.T[1:]
        bodyPositions = []
        for bodyset in bodyData:
            positions = []
            for body in bodyset:
                positions.append(body.position)
            positions = np.array(positions)
            bodyPositions.append(positions.T)
        bodyPositions = np.array(bodyPositions)
        return bodyPositions

    def energyCalc(self):
        """
        From a saved datafile, the kinetic, potential, and total energy are calculated by callign functions on the body Particle classes.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        /////////////////////
        Saved / Loaded Files
        --------------------
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        ////////
        Returns
        --------
        [[KE], [PE], [TE]]: np.ndarray
            An array of the kinetic, potential and total energy for time t in the simulation.
        """
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        bodyData = DataIn.T[1:].T
        KEs = [] # kinetic energies
        PEs = [] # potential energies
        TEs= [] # total energies
        for tick, bodyset in zip(range(len(bodyData)),bodyData):
            time = tick * self.timestep * 100
            kineticSum = 0 # kinetic sum
            potentialSum = 0 # potential sum
            for body in bodyset:
                kineticSum += body.kineticEnergy() # get body KE
            for i in range(len(bodyset)):
                for j in range(i+1, len(bodyset)):
                    potentialSum += bodyset[i].potentialEnergy(bodyset[j])
            KEs.append([time, kineticSum])
            PEs.append([time, potentialSum])
            TEs.append([time, kineticSum + potentialSum])
        return np.array([np.array(KEs).T, np.array(PEs).T, np.array(TEs).T]) # return ndarray (required for index slicing)

    def customEnergyCalc(self, integrator_code):
        """
        From a saved datafile, the positions of each body are extracted and put into a numpy array of shape (len(self.bodies), 3, self.itns / 100).
        Unlike the sibling function energyCalc() this function performs calculations for a specified integrator independent of the enetered class attributes.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        integrator_code: int, (1, 2, 3, 4 or 5)
            An integer code between 1 and 5 for selecting the chosen integrator.
        /////////////////////
        Saved / Loaded Files
        --------------------
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        ////////
        Returns
        --------
        [KE, PE, TE]: np.ndarray
            An array of the kinetic, potential and total energy for time t in the simulation.
        """
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{integrator_code}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        bodyData = DataIn.T[1:].T
        KEs = []
        PEs = []
        TEs= []
        for tick, bodyset in zip(range(len(bodyData)),bodyData):
            time = tick * self.timestep * 100
            kineticSum = 0
            potentialSum = 0
            for body in bodyset:
                kineticSum += body.kineticEnergy()
            for i in range(len(bodyset)):
                for j in range(i+1, len(bodyset)):
                    potentialSum += bodyset[i].potentialEnergy(bodyset[j])
            KEs.append([time, kineticSum])
            PEs.append([time, potentialSum])
            TEs.append([time, kineticSum + potentialSum])
        return np.array([np.array(KEs).T, np.array(PEs).T, np.array(TEs).T])
    
    def totalLinearMomenta(self):
        """
        From a saved datafile, the total linear momentum p=mv of the simulation is calculated for each sampled set of bodies using a Particle class function.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        /////////////////////
        Saved / Loaded Files
        --------------------
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        ////////
        Returns
        --------
        total linear momenta: np.ndarray
            An array of the total linear momentum for each time t.
        """
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        bodyData = DataIn.T[1:].T
        totalLinearMomenta = []
        for tick, bodyset in zip(range(len(bodyData)), bodyData):
            time = tick * self.timestep * 100
            total = np.array([0.0, 0.0, 0.0])
            for body in bodyset:
                total += body.getMomentum()
            totalLinearMomenta.append([time, np.abs(np.linalg.norm(total))])
        return np.array(totalLinearMomenta).T

    def totalAngularMomenta(self):
        """
        From a saved datafile, the total angular momentum L = m * (r {cross} v) of the simulation is calculated for each sampled set of bodies using a Particle class function.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        /////////////////////
        Saved / Loaded Files
        --------------------
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        ////////
        Returns
        --------
        total angular momenta: np.ndarray
            An array of the total angular momentum for each time t.
        """
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        bodyData = DataIn.T[1:].T
        totalAngularMomenta = []
        for tick, bodyset in zip(range(len(bodyData)), bodyData):
            time = tick * self.timestep * 100
            total = np.array([0.0, 0.0, 0.0])
            for body in bodyset:
                total += body.angularMomentumToOrigin()
            totalAngularMomenta.append([time, np.abs(np.linalg.norm(total))])
        return np.array(totalAngularMomenta).T

    def getEccentricities(self):
        """
        From a saved datafile, the eccentricities of all of the bodies at their initial position is calculated.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        /////////////////////
        Saved / Loaded Files
        --------------------
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        ////////
        Returns
        --------
        eccentricities: list
            An list of eccentricities for each body.
        """
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        bodyData = DataIn.T[1:].T
        sol = bodyData[0, 0]
        eccentricities = []
        for body in bodyData[0, 1:]:
            eccentricities.append([body.name, body.eop(sol)[0]])
        return eccentricities
    
    def seconds_to_dhms(self, seconds):
        """
        Convert a number of seconds to days, hours, minutes, and seconds.
        ///////////
        Parameters
        ----------
        seconds: float
            Number of seconds.
        ////////
        Returns
        --------
        time: tuple
            (days, hours, minutes, seconds)
        """
        seconds = int(seconds)  # ensure integer division
        days = seconds // 86400
        seconds %= 86400
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return days, hours, minutes, seconds

    def ephOrbitalPeriods(self):
        """
        From a saved datafile, the orbital period of all of the bodies at their initial position is calculated.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        /////////////////////
        Saved / Loaded Files
        --------------------
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        ////////
        Returns
        --------
        orbital periods: list
            An list of orbital periods for each body.
        """
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        bodyData = DataIn.T[1:].T
        sol = bodyData[0, 0]
        periods = []
        for body in bodyData[0, 1:]:
            periods.append([body.name, body.eop(sol)[1]])
        return periods
    #////////////////////////////////
    ##### CSV SAVING FUNCTIONS #####
    def saveBodyPositionsCSV(self):
        """
        Saves a .csv file of body positions.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
                checkData()
                getBodyPositions()
        /////////////////////
        Saved / Loaded Files
        --------------------
        csv_file = f"data/csv/{self.sim}_{self.integrator}_{self.itns}_{int(self.timestep)}_BodyPositions.csv"
        """
        self.checkData()
        positions = self.getBodyPositions() 
        csv_file = f"data/csv/{self.sim}_{self.integrator}_{self.itns}_{int(self.timestep)}_BodyPositions.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for tick_positions in positions:
                writer.writerow(tick_positions.flatten())

    def saveEnergiesCSV(self):
        """
        Saves a .csv file of body positions.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
                checkData()
                energyCalc()
        /////////////////////
        Saved / Loaded Files
        --------------------
        csv_file = f"data/csv/{self.sim}_{self.integrator}_{self.itns}_{int(self.timestep)}_Energies.csv"
        """
        self.checkData()
        energies = self.energyCalc()
        KE, PE, TE = energies
        csv_file = f"data/csv/{self.sim}_{self.integrator}_{self.itns}_{int(self.timestep)}_Energies.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'KineticEnergy', 'PotentialEnergy', 'TotalEnergy'])
            for t, ke, pe, te in zip(KE[0], KE[1], PE[1], TE[1]):
                writer.writerow([t, ke, pe, te])

    def saveLinearMomentaCSV(self):
        """
        Saves a .csv file of body positions.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
                checkData()
                totalLinearMomenta()
        /////////////////////
        Saved / Loaded Files
        --------------------
        csv_file = f"data/csv/{self.sim}_{self.integrator}_{self.itns}_{int(self.timestep)}_LinearMomenta.csv"
        """
        self.checkData()
        totalLM = self.totalLinearMomenta()
        csv_file = f"data/csv/{self.sim}_{self.integrator}_{self.itns}_{int(self.timestep)}_LinearMomenta.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'TotalLinearMomentum'])
            for t, lm in zip(totalLM[0], totalLM[1]):
                writer.writerow([t, lm])

    def saveAngularMomentaCSV(self):
        """
        Saves a .csv file of body positions.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
                checkData()
                totalAngularMomenta()
        /////////////////////
        Saved / Loaded Files
        --------------------
        csv_file = f"data/csv/{self.sim}_{self.integrator}_{self.itns}_{int(self.timestep)}_AngularMomenta.csv"
        """
        self.checkData()
        totalAM = self.totalAngularMomenta()  
        csv_file = f"data/csv/{self.sim}_{self.integrator}_{self.itns}_{int(self.timestep)}_AngularMomenta.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'TotalAngularMomentum'])
            for t, am in zip(totalAM[0], totalAM[1]):
                writer.writerow([t, am])

    ##### SIMULATION UPDATE FUNCTIONS #####
    def updateIntegrator(self, body):
        """
        Based on the value of the integer self.integrator, the inputted body will be updated with the integrator algorithm.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
        body: Particle class object
        """
        if self.integrator == 1:
            body.updateEu(self.timestep)
        if self.integrator == 2:
            body.updateEuC(self.timestep)
        if self.integrator == 3:
            body.updateEuR(self.timestep, self.bodies)
        if self.integrator == 4:
            body.updateVerlet(self.timestep, self.bodies)
        if self.integrator == 5:
            body.updateRK4(self.timestep, self.bodies)

    def initSim(self): # Function for choosing from simulation presets.
        """
        Function that initialises the simulation. It can shoose from different presets depending on the class attributes.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
                importBodies()
                seconds_to_dhms()
                updateIntegrator()
        /////////////////////
        Saved / Loaded Files
        --------------------
        np.save(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", Data, allow_pickle=True)
        """
        print('Commencing simulation:')
        start = runtime.time()
        #//////////////////////////////////////
        self.importBodies() # import bodies 
        Data = [] # create an empty data list.
        i = 0
        time = 0.0
        completion = 0
        ETAtoggle = True
        if self.itns < 100:
            print('Please increase the number of iterations beyond the minimum of 100 and run the program again.')
        else:
            while i < (self.itns):
                if (i-1)%100 == 0: # for every 100th iteration:
                    Data.append([time] + [copy.deepcopy(body) for body in self.bodies]) # record data and deepcopies of the body objects to a .npy file.
                if (i-1) % (self.itns//100) == 0:
                    completion += 1
                    print(f'Progress: {completion}%')
                if completion == 2 and ETAtoggle == True:
                    two_percent = runtime.time()
                    d, h, m, s = self.seconds_to_dhms((two_percent - start) * 98)
                    print(f"Estimated time remaining: {d} days, {h} hours, {m} minutes, {s} seconds.")
                    ETAtoggle = False
                for body in self.bodies: # for each body:
                    body.updateGravitationalAcceleration(self.bodies) # update gravitational acceleration.
                    self.updateIntegrator(body) # then update the main attributes using the selected integrator.
                time += self.timestep # adds seconds equal to the timestep to the time count to keep track of the total time elapsed.
                i += 1
            np.save(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", Data, allow_pickle=True)
            #//////////////////////////////////////
            end = runtime.time()
            print(f"Total calculation time elapsed using integrator {self.integrator}: {end - start} seconds")

    ##### PLOTTING FUNCTIONS #####
    def plot_relative_energy_error(self):
        """
        Plots the relative energy error of an N-body simulation as error (log scale) against time for a single integrator. A .svg plot is also saved if saving == True.
        ////////////
        Parameters
        ----------
        self: class
            Simulation()
                checkData()
                importBodies()
                energyCalc()
        /////////////////////
        Saved / Loaded Files
        --------------------
        plt.savefig(f'figures/{self.sim}_REE_{len(self.bodies)}_{self.integrator}_{self.itns}_{self.timestep}.svg')
        """
        self.checkData()
        self.importBodies()
        # Extract total energy array
        TE = self.energyCalc()[2]  # Total energy
        times = TE[0]         # time values
        total_energy = TE[1]  # total energy values
        # Compute initial energy
        E0 = total_energy[0]
        # Compute relative energy error
        rel_error = np.abs(total_energy - E0) / np.abs(E0)
        print(rel_error.max())
        # integrator check
        integrator = ''
        colour = ''
        if self.integrator == 1:
            integrator = 'Euler'
            colour = 'brown'
        if self.integrator == 2:
            integrator = 'Euler-Cromer'
            colour = 'darkorange'
        if self.integrator == 3:
            integrator = 'Euler-Richardson'
            colour = 'forestgreen'
        if self.integrator == 4:
            integrator = 'Verlet'
            colour = 'blue'
        if self.integrator == 5:
            integrator = 'Runge-Kutta-4'
            colour = 'navy'
        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(times, rel_error, label='Relative Energy Error', color=colour)
        plt.xlabel('t (s)', fontstyle='italic')
        plt.ylabel('Relative energy error (J)', fontstyle='italic')
        plt.yscale('log')
        ax = plt.gca()
        tickLimit = self.itns * self.timestep
        major = tickLimit / 10 # place major ticks every 1/10th of limit
        minor = major / 4 # place minor ticks every 1/4 of a major tick
        ax.xaxis.set_major_locator(MultipleLocator(major))
        ax.xaxis.set_minor_locator(MultipleLocator(minor))
        ax.minorticks_on()  # often useful to see small errors
        ax.tick_params(axis='both', which='major', direction='in')
        ax.tick_params(axis='both', which='minor', direction='in')
        plt.title(f'Relative energy error of {len(self.bodies)}-body "{self.sim}" system, using the {integrator} integrator.', fontweight='bold', fontsize=10)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        if self.saving == True:
            # Save the full figure...
            plt.savefig(f'figures/{self.sim}_REE_{len(self.bodies)}_{self.integrator}_{self.itns}_{self.timestep}.svg')
        plt.show()

    def groupedREE(self):
        """
        Plots the relative energy error of an N-body simulation as error (log scale) against time for all integrators at once. A .svg plot is also saved if saving == True.
        ////////////
        Parameters
        ----------
        self: class
            Simulation()
                checkData()
                importBodies()
                customEnergyCalc(self.integrator)[2]
        /////////////////////
        Saved / Loaded Files
        --------------------
        plt.savefig(f'figures/{self.sim}_REE_{len(self.bodies)}_1-5_{self.itns}_{self.timestep}.svg')
        """
        self.checkData()
        self.importBodies()
        plt.figure(figsize=(8, 5))
        total_energies = [self.customEnergyCalc(1)[2], self.customEnergyCalc(2)[2], self.customEnergyCalc(3)[2], self.customEnergyCalc(4)[2], self.customEnergyCalc(5)[2]]
        labels = ('Euler', 'Euler-Cromer', 'Euler-Richardson', 'Verlet', 'Runge-Kutta-4')
        colours = ('brown', 'darkorange', 'forestgreen', 'blue', 'purple')
        for total, intlabel, colour in zip(total_energies, labels, colours):
            # Extract total energy array
            TE = total
            times = TE[0]         # time values
            total_energy = TE[1]  # total energy values
            # Compute initial energy
            E0 = total_energy[0]
            # Compute relative energy error
            rel_error = np.abs(total_energy - E0) / np.abs(E0)
            print(rel_error.max())
            plt.plot(times, rel_error, label=intlabel, color=colour)
        # settings
        plt.xlabel('t (s)', fontstyle='italic')
        plt.ylabel('Relative energy error (J)', fontstyle='italic')
        plt.yscale('log')
        ax = plt.gca()
        tickLimit = self.itns * self.timestep
        major = tickLimit / 10
        minor = major / 4
        ax.xaxis.set_major_locator(MultipleLocator(major))
        ax.xaxis.set_minor_locator(MultipleLocator(minor))
        ax.minorticks_on()  # often useful to see small errors
        ax.tick_params(axis='both', which='major', direction='in')
        ax.tick_params(axis='both', which='minor', direction='in')
        plt.title(f'Relative energy error of {len(self.bodies)}-body "{self.sim}" system, using several integrators.', fontweight='bold', fontsize=10)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        if self.saving == True:
            # Save the full figure...
            plt.savefig(f'figures/{self.sim}_REE_{len(self.bodies)}_1-5_{self.itns}_{self.timestep}.svg')
        plt.show()

    def eccentricityEarthPlot(self, save_csv=False):
        """
        Plots the eccentricity of Earth at all times t in the simulation based on the collected file data.
        A .svg plot is saved if saving == True, and the data can also be saved to a .csv file if save_csv == True.
        ////////////
        Parameters
        ----------
        self: class
            Simulation()
                checkData()
                importBodies()
        save_csv: boolean
            If True, the eccentricity data will be saved to a CSV file.
        /////////////////////
        Saved / Loaded Files
        --------------------
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        """
        self.checkData()
        self.importBodies()
        plt.figure(figsize=(8, 5))
        DataIn = np.load(f"data/NBodyTest_{self.sim}_{self.integrator}_{self.itns}_{self.timestep}.npy", allow_pickle=True)
        sun_objects = DataIn.T[1:][0]
        earth_objects = DataIn.T[1:][3]
        eccentricities = []
        
        # Collect the eccentricity data for each time step
        for tick, particle, star in zip(range(len(earth_objects)), earth_objects, sun_objects):
            time = tick * self.timestep * 100
            eccentricities.append([time, particle.eop(star)[0]])
        
        eccentricities = np.array(eccentricities).T

        # Plotting the eccentricity data
        plt.plot(eccentricities[0], eccentricities[1], label='Eccentricity', color='red')
        mean = np.mean(eccentricities[1])
        plt.plot([0, 1e9], [mean, mean], label=f'Average eccentricity e = {"{:.4e}".format(mean)}', color='blue', linestyle='dashed')
        plt.plot([], [], ' ', label="Accepted value e ≈ 1.67e-02")
        plt.xlabel('t(s)', fontstyle='italic')
        plt.ylabel('Eccentricity', fontstyle='italic')
        ax = plt.gca()
        ax.minorticks_on()  # often useful to see small errors
        ax.tick_params(axis='both', which='major', direction='in')
        ax.tick_params(axis='both', which='minor', direction='in')
        title = 'Eccentricities of Earth over the course of the simulation'
        plt.title(f'{title}', fontweight='bold', fontsize=10)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figures/{title}.svg')
        plt.show()

        # If save_csv is True, save the data to a .csv file
        if save_csv:
            csv_filename = f'data/csv/{title}.csv'
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write header
                writer.writerow(['Time (s)', 'Eccentricity'])
                # Write the eccentricity data
                for t, e in zip(eccentricities[0], eccentricities[1]):
                    writer.writerow([t, e])

            print(f"Eccentricity data saved to {csv_filename}")

    def masterView(self):
        """
        Plots a figure containing a plot of energy, linear momentum, angular momentum and body position over the duration of the simulation. All are animated. A .svg plot is also saved if saving == True.
        ///////////
        Parameters
        ----------
        self: class
            Simulation()
                checkData()
                importBodies()
                energyCalc()
                getBodyPositions()
                totalLinearMomenta()
                totalAngularMomenta()
        /////////////////////
        Saved / Loaded Files
        --------------------
        fig.savefig(f'figures/{self.sim}_figure_{len(self.bodies)}_{self.integrator}_{self.itns}_{self.timestep}.svg')
        """
        self.checkData()
        self.importBodies()
        fig = plt.figure(figsize=(14, 7), dpi=100) # creates a figure to display subplots on. The size is specified (width, height) and is multiplied by dots per inch to get the number of pixel dimensions to display the window with.
        ax1 = plt.subplot2grid((3, 5), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((3, 5), (0, 2), rowspan=3, colspan=3, projection='3d', aspect='equal')
        ax3 = plt.subplot2grid((3, 5), (1, 0), colspan=2)
        ax4 = plt.subplot2grid((3, 5), (2, 0), colspan=2)
        
        ### Data Function Calls and Plot Limits ###
        # Data #
        energySets = self.energyCalc()
        bodyPositions = self.getBodyPositions() / AU
        totalLinearMomenta = self.totalLinearMomenta()
        totalAngularMomenta = self.totalAngularMomenta()
        # Limits #
        energyLims = [-1.25 * energySets[:,1].min(), 1.25 * energySets[:,1].min()]
        pLims = [-1.25 * totalLinearMomenta[1].min(), 1.25 * totalLinearMomenta[1].max()]
        apLims = [-1.25 * totalAngularMomenta[1].min(), 1.25 * totalAngularMomenta[1].max()]

        ### Axis Settings ###
        # for ax1, ax3, ax4: #
        tickLimit = self.itns * self.timestep
        major = tickLimit / 10
        minor = major / 4 # gives 3 minors between every major
        y_labels = ('Tot. Energy (J)', 'Tot. Lin. Momentum (kg·m/s)', 'Tot. Ang. Momentum (kg·m²/s)')
        for ax, y_lim, y_label in zip((ax1, ax3, ax4), (energyLims, pLims, apLims), y_labels):
            ax.set_xlim([self.timestep, tickLimit])  # scales the x-axis (time) to equal the length of the simualtion data.
            ax.xaxis.set_major_locator(MultipleLocator(major))
            ax.xaxis.set_minor_locator(MultipleLocator(minor))
            ax.set_ylim(y_lim) # limits the y-axis between the maximum and minimum energy, L-momentum or A-momentum values respectively.
            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', direction='in')
            ax.tick_params(axis='both', which='minor', direction='in')
            ax.set_xlabel('Time (s)', fontstyle='italic')
            ax.set_ylabel(f'{y_label}', fontstyle='italic')
        # ax2 #
        axisLimit = 1
        if self.sim == 'earthsat':
            axisLimit = 0.0025
        if self.sim == 'solarsystem':
            axisLimit = 20
        ax2Lims = [-axisLimit, axisLimit]
        ax2.set_xlim3d(ax2Lims)
        ax2.set_ylim3d(ax2Lims)
        ax2.set_zlim3d(ax2Lims)
        ax2.set_xlabel('x (AU)')
        ax2.set_ylabel('y (AU)')
        ax2.set_zlabel('z (AU)')
        ax2.w_xaxis.set_pane_color((0.0, 0.0, 0.0))
        ax2.w_yaxis.set_pane_color((0.0, 0.0, 0.0))
        ax2.w_zaxis.set_pane_color((0.0, 0.0, 0.0))
        ax2.ticklabel_format(style='plain')
        ax2.grid()

        ### Pre-plot for pngs ###
        lines = []
        # ax1 plot #
        energyNames = ['Kinetic Energy','Potential Energy','Total Energy']
        energyColours = ['red','blue','green']
        for energy, colour, name in zip(energySets, energyColours, energyNames):
            line1, = ax1.plot(energy[0], energy[1], f'{colour}', linestyle='solid', label=f'{name}')
            lines.append(line1)
        # ax2 plot #
        bodyNames = []
        bodyColours = []
        for body in self.bodies:
            bodyNames.append(body.name)
            bodyColours.append(body.colour)
        for body, colour, name in zip(bodyPositions, bodyColours, bodyNames):
            line2, = ax2.plot(body[0], body[1], body[2], f'{colour}', linestyle='solid', label=f'{name}')
            lines.append(line2)
        # ax2 Cosmetic Spheres #
        def addTexturedSphere(radius=1, texture_path='textures/earth.jpg', slabel='Earth'):
            u = np.linspace(0, 2*np.pi, 100)
            v = np.linspace(0, np.pi, 50)
            x = radius * np.outer(np.cos(u), np.sin(v))
            y = radius * np.outer(np.sin(u), np.sin(v))
            z = radius * np.outer(np.ones_like(u), np.cos(v))
            texture = mpimg.imread(texture_path)
            tex_u = (u / (2*np.pi)) * (texture.shape[1] - 1)
            tex_v = (v / np.pi) * (texture.shape[0] - 1)
            tex_u, tex_v = np.meshgrid(tex_u, tex_v, indexing="ij")
            rgb = texture[tex_v.astype(int), tex_u.astype(int)]
            ax2.plot_surface(x, y, z, facecolors=rgb / 255.0, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, label=f'{slabel}')
        if self.sim == 'earthsat':
            addTexturedSphere(radius=earthRadius/AU, texture_path='textures/earth.jpg', slabel='Earth')
        if self.sim == 'solarsystem':
            addTexturedSphere(radius=solRadius/AU, texture_path='textures/sun.jpg', slabel='The Sun')
        # ax3 plot #
        line3, = ax3.plot(totalLinearMomenta[0], totalLinearMomenta[1], linestyle='solid', label='Total Linear Momentum')
        lines.append(line3)
        # ax4 plot #
        line4, = ax4.plot(totalAngularMomenta[0], totalAngularMomenta[1], linestyle='solid', label='Total Angular Momentum')
        lines.append(line4)
        # legends and titles #
        ax1.legend(energyNames)
        ax2.legend(bodyNames)
        ax3.legend(['Total Linear Momentum'])
        ax4.legend(['Total Angular Momentum'])
        ax1.set_title('A. Energy values of the N-body system against time')
        ax2.set_title('B. Orbital path traces of the time-evolved N-body system')
        ax3.set_title('C. Total linear momentum of the N-body system against time')
        ax4.set_title('D. Total angular momentum of the N-body system against time')
        integrator = ''
        if self.integrator == 1:
            integrator = 'Euler'
        if self.integrator == 2:
            integrator = 'Euler-Cromer'
        if self.integrator == 3:
            integrator = 'Euler-Richardson'
        if self.integrator == 4:
            integrator = 'Verlet'
        if self.integrator == 5:
            integrator = 'Runge-Kutta-4'
        fig.suptitle(f'Simulated {len(self.bodies)}-body "{self.sim}" system, using {integrator} integrator with {self.itns} iterations, {self.timestep} second timestep.', fontweight='bold')
        plt.tight_layout()
        # .svg file saving #
        if self.saving == True:
            # Save the full figure...
            fig.savefig(f'figures/{self.sim}_figure_{len(self.bodies)}_{self.integrator}_{self.itns}_{self.timestep}.svg')
        for line in lines:
            line.remove() # clears the figure to make way for animations.

        ### Line Drawing and Animations ###
        frame_count = math.floor(self.itns / 100) # Gets the number of data points and ensures the frame count is equal to it and is an integer.
        # ax1 Lines #
        ax1Lines = []
        for i, energy, colour in zip(range(len(energySets)), energySets, energyColours):
            line, = ax1.plot(energy[0, 0:1], energy[1, 0:1], f'{colour}') # We plot the first two entries of x-y (time-energy) data for each line (KE, PE, TE).
            ax1Lines.append(line,)
        def updateAx1(frame, energySets, ax1Lines):
            for energy, ax1Line in zip(energySets, ax1Lines):
                ax1Line.set_data(energy[:2, :frame]) # set data up to (but not including) the third row and then in each up to the current frame.
        # ax2 Lines #
        ax2Lines = []
        for i, body, colour in zip(range(len(self.bodies)), bodyPositions, bodyColours): # For each body creates a plotting line of initial positions. It iterates for the range equal to the number of bodies in the simulation.
            line, = ax2.plot(body[0, 0:1], body[1, 0:1], body[2, 0:1], f'{colour}') # 
            ax2Lines.append(line,)
        def updateAx2(frame, bodyPositions, ax2Lines):
            for body, ax2Line in zip(bodyPositions, ax2Lines):
                ax2Line.set_data(body[:2, :frame]) # set data up to (but not including) the third row and then in each up to the current frame.
                ax2Line.set_3d_properties(body[2, :frame]) # because we plot here in three dimensions, we need to also set data for the z coordinate up to the current frame.
        # ax3 Lines #
        ax3Line, = ax3.plot(totalLinearMomenta[0, 0:1], totalLinearMomenta[1, 0:1])
        def updateAx3(frame, totalLinearMomenta, ax3Line):
            ax3Line.set_data(totalLinearMomenta[:2, :frame]) # set data up to (but not including) the third row and then in each up to the current frame.
        # ax4 Lines # 
        ax4Line, = ax4.plot(totalAngularMomenta[0, 0:1], totalAngularMomenta[1, 0:1])
        def updateAx4(frame, totalAngularMomenta, ax4Line):
            ax4Line.set_data(totalAngularMomenta[:2, :frame]) # set data up to (but not including) the third row and then in each up to the current frame.
        # animations #
        anim1 = anim.FuncAnimation(fig=fig, func=updateAx1, frames=frame_count, interval=1, fargs=(energySets, ax1Lines), repeat=False)
        anim2 = anim.FuncAnimation(fig=fig, func=updateAx2, frames=frame_count, interval=1, fargs=(bodyPositions, ax2Lines), repeat=False)
        anim3 = anim.FuncAnimation(fig=fig, func=updateAx3, frames=frame_count, interval=1, fargs=(totalLinearMomenta, ax3Line), repeat=False)
        anim4 = anim.FuncAnimation(fig=fig, func=updateAx4, frames=frame_count, interval=1, fargs=(totalAngularMomenta, ax4Line), repeat=False)
        plt.show()