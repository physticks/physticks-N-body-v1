import numpy as np
from astropy.time import Time
from Ephemeris import Ephemeris
from bd_solarsystem import Bodies
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

G=6.67408E-11
deltaT = 1000
solarsystem = Ephemeris(
        time = Time("2057-08-10 22:22:11.653968", scale="tdb"),
        combined = {"earth-moon-barycenter": ("earth", "moon")}
    )

test_files = [
    'NBodyTest_solarsystem_5_200_0.5.npy',
    'NBodyTest_solarsystem_5_1000_0.1.npy',
    'NBodyTest_solarsystem_5_2000_0.05.npy',
    'NBodyTest_solarsystem_5_10000_0.01.npy',
    'NBodyTest_solarsystem_5_20000_0.005.npy',
    'NBodyTest_solarsystem_5_100000_0.001.npy',
    'NBodyTest_solarsystem_5_200000_0.0005.npy',
    'NBodyTest_solarsystem_5_1000000_0.0001.npy',
    'NBodyTest_solarsystem_5_2000000_0.0005.npy',
]

time_steps = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

def getPosVel(filepath: str):
    """
    From a saved datafile, the body positions and velocities are retrieved.
    ///////////
    Parameters
    ----------
    filepath: str
        The path to the file the function tries to load.
    /////////////////////
    Saved / Loaded Files
    --------------------
    DataIn = np.load(filepath, allow_pickle=True)
    ////////
    Returns
    --------
    [[body positions], [body velocities]]: np.ndarray
        An array of the kinetic, potential and total energy for time t in the simulation.
    """
    DataIn = np.load(filepath, allow_pickle=True)
    bodyData = DataIn.T[1:]
    bodyPositions = []
    bodyVelocities = []
    for bodyset in bodyData:
        positions = []
        velocities = []
        for body in bodyset:
            positions.append(body.position)
            velocities.append(body.velocity)
        positions = np.array(positions)
        velocities = np.array(velocities)
        bodyPositions.append(positions.T)
        bodyVelocities.append(velocities.T)
    return np.array([np.array(bodyPositions), np.array(bodyVelocities)])

def getAcceleration(position, bodies):
    """
    Updates the total gravitational acceleration of the particle towards other bodies by calculating a vector sum of individual accelerations.
    ///////////
    Parameters
    ----------
    position: np.ndarray vector
        inputted position
    bodies: list of objects
        A list of all the body objects that the particle gravitates towards.
    Returns
    --------
    acceleration: np.ndarray vector
    """
    acceleration = 0
    sbodies = bodies[:3] + bodies[4:]
    for body in sbodies:
        delta_r = np.subtract(position, body.position)
        acceleration += -np.divide((G * body.mass * delta_r) , np.abs(np.linalg.norm(delta_r))**3)
    return acceleration

def getErrors(file_path):
    """
    Updates the total gravitational acceleration of the particle towards other bodies by calculating a vector sum of individual accelerations.
    ///////////
    Parameters
    ----------
    file_path: str
        The path to the file the function tries to load.
    ////////
    Returns
    --------
    errors: list
        [numerical_error, [positional_errors]]
    """
    # positions and velocities
    earth_positions = getPosVel(file_path)[0, 3].T
    earth_velocities = getPosVel(file_path)[1, 3].T
    r64 = getPosVel('data/NBodyTest_solarsystem_5_5000_1000.0_dt64.npy')[0, 3].T[-1]
    r32 = getPosVel('data/NBodyTest_solarsystem_5_5000_1000.0_dt32.npy')[0, 3].T[-1]
    pos_JPL = solarsystem.getPosVel('earth-moon-barycenter')
    # final pos, vel
    pos_final = earth_positions[-1]
    vel_final = earth_velocities[-1]
    new_earth_pos_final = getPosVel('data/NBodyTest_solarsystem_5_500000_2000.0.npy')[0, 3].T[-1]
    # penultimate pos, vel
    pos_pen = earth_positions[-2]
    vel_pen = earth_velocities[-2]
    # final, penultimate accelerations
    acc_final = getAcceleration(pos_final, Bodies)
    acc_pen = getAcceleration(pos_pen, Bodies)
    # series vectors
    r_p_n1 = 1/3 * (4 * pos_final - pos_pen) + 1/6 * deltaT * (3 * vel_final + vel_pen) + 1/36 * deltaT**2 * (31 * acc_final - acc_pen)
    v_p_n1 = 1/3 * (4 * vel_final - vel_pen) + 2/3 * deltaT * (2 * acc_final - acc_pen)
    a_p_n1 = getAcceleration(r_p_n1, Bodies)
    r_n1 = 1/3 * (4 * pos_final - pos_pen) + 1/24 * deltaT * (v_p_n1 + 14 * vel_final + vel_pen) + 1/72 * deltaT**2 * (10 * a_p_n1 + 51 * acc_final - acc_pen)
    # errors
    trunc_error = np.abs(np.linalg.norm(r_n1 - r_p_n1)) / np.abs(np.linalg.norm(r_n1))
    #print(f'Estimated truncation error: {trunc_error}')
    rel_rounding_error = np.linalg.norm(np.subtract(r32, r64)) / np.linalg.norm(r64)
    #print(f'Rounding error: {rounding_error}')
    rel_rich_error = 16/15 * np.abs(np.linalg.norm(np.subtract(new_earth_pos_final, pos_final)))
    #print(f'Richardson error: {rich_error}')
    JPL_ratio = np.linalg.norm(np.subtract(new_earth_pos_final, pos_JPL)) / max(1, np.linalg.norm(pos_JPL))
    #print(f'Error to JPL data: {JPL_ratio}')
    numerical_error = trunc_error + rel_rounding_error
    positional_errors = [rel_rich_error, JPL_ratio]
    return [numerical_error, positional_errors]

def plotNumericalError(test_files, time_steps):
    """
    Plots the relative energy error of an N-body simulation as error (log scale) against time for a single integrator. A .svg plot is also saved if saving == True.
    ////////////
    Parameters
    ----------
    test_files: list of strings
        A list of the files, not the filepath!! files should be in the data folder.
    time_steps: list of floats
        The timesteps that match those in the file names in test_files.
    /////////////////////
    Saved / Loaded Files
    --------------------
    plt.savefig(f'figures/{title}.svg')
    """
    error_data = []
    for file, time_step in zip(test_files, time_steps):
        numerical_error = getErrors(f'data/{file}')[0]
        error_data.append([time_step, numerical_error])
    error_data = np.array(error_data).T
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(error_data[0], error_data[1], label='numerical error', color='red')
    #plt.plot(np.log10(error_data[0]), np.log10(error_data[1]), label='numerical error', color='red')
    plt.xlabel('Δt(s)', fontstyle='italic')
    plt.ylabel('numerical_error', fontstyle='italic')
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    ax.minorticks_on()  # often useful to see small errors
    ax.tick_params(axis='both', which='major', direction='in')
    ax.tick_params(axis='both', which='minor', direction='in')
    title = 'Total Numerical Error against simulation time step'
    plt.title(f'{title}', fontweight='bold', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/{title}.svg')
    plt.show()

#plotNumericalError(test_files, time_steps)