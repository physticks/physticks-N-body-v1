from Simulation import Simulation

simulation = Simulation(
    sim = 'solarsystem',
    integrator = 5,
    iterations = 10000,
    timestep = 1000,
    saving = True
)

simulation.masterView()
#simulation.plot_relative_energy_error()