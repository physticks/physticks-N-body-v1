from Simulation import Simulation
print("")
print("N-body Simulator Ready. Please see the docstring of the Simulation class below:")
print(Simulation.__init__.__doc__)
print("Avaialble methods to call from the class:")
methods_list = [method for method in dir(Simulation) if callable(getattr(Simulation, method)) and not method.startswith("__")]
print(f'Methods using dir():{methods_list}')
print("")
nBody = Simulation(
        sim = str(input("Please enter a preset ('earthsat' or 'solarsystem'): ")),
        integrator = int(input("Choose an integrator using an integer from 1 to 5 (1 = Euler, 2 = Euler-Cromer etc.): ")),
        iterations = int(input("Enter the number of iterations: ")),
        timestep = float(input("Enter the size of the timestep: ")),
        saving = bool(input("Would you like plots to be saved to the figures/ directory? True/False: "))
    )

if input("Would you like to run a simulation with those entered parameters? Y/N:") == "Yes" or "Y":
    nBody.initSim()

print("What would you like to plot?")
if nBody.sim == 'earthsat':
    print("Select from: nBody.plot_relative_energy_error() (Enter 'REE'), nBody.groupedREE() (Enter 'G-REE'), nBody.masterView() (Enter 'MV')")
    if str(input("Plot: ")) == 'REE':
        nBody.plot_relative_energy_error()
        if input("Would you like the data outputted to a .csv? Y/N: ") == "Yes" or "Y":
            nBody.saveEnergiesCSV()
            print(".csv files can be found in data/csv.")
    elif str(input("Plot: ")) == 'G-REE':
        nBody.groupedREE()
        if input("Would you like the data outputted to a .csv? Y/N: ") == "Yes" or "Y":
            nBody.saveEnergiesCSV()
            print(".csv files can be found in data/csv.")
    elif str(input("Plot: ")) == 'MV':
        nBody.masterView()
        if input("Would you like the data outputted to a .csv? Y/N: ") == "Yes" or "Y":
            nBody.saveEnergiesCSV()
            nBody.saveAngularMomentaCSV()
            nBody.saveBodyPositionsCSV()
            nBody.saveLinearMomentaCSV()
            print(".csv files can be found in data/csv.")

if nBody.sim == 'solarsystem':
    print("Select from: nBody.plot_relative_energy_error() (Enter 'REE'), nBody.groupedREE() (Enter 'G-REE'), nBody.masterView() (Enter 'MV'), nBody.eccentricityEarthPlot() (Enter 'EEP')")
    if str(input("Plot: ")) == 'REE':
        nBody.plot_relative_energy_error()
        if input("Would you like the data outputted to a .csv? Y/N: ") == "Yes" or "Y":
            nBody.saveEnergiesCSV()
            print(".csv files can be found in data/csv.")
    elif str(input("Plot: ")) == 'G-REE':
        nBody.groupedREE()
        if input("Would you like the data outputted to a .csv? Y/N: ") == "Yes" or "Y":
            nBody.saveEnergiesCSV()
            print(".csv files can be found in data/csv.")
    elif str(input("Plot: ")) == 'MV':
        nBody.masterView()
        if input("Would you like the data outputted to a .csv? Y/N: ") == "Yes" or "Y":
            nBody.saveEnergiesCSV()
            nBody.saveAngularMomentaCSV()
            nBody.saveBodyPositionsCSV()
            nBody.saveLinearMomentaCSV()
            print(".csv files can be found in data/csv.")
    elif str(input("Plot: ")) == 'EEP':
        if input("Save a .csv? Y/N: ") == "Yes" or "Y":
            nBody.eccentricityEarthPlot(save_csv=True)
            print(".csv files can be found in data/csv.")
        else:
            nBody.eccentricityEarthPlot(save_csv=False)