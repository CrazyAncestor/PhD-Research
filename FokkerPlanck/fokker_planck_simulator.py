import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class FokkerPlanckSimulator:
    def __init__(self, t_start, t_end, dt, gamma, nu, n_th, eps2, x, y, x0, y0, output_dir, ProbDensMap, solver, init_cond):
        # Simulation time settings
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.nsteps = int((t_end - t_start) / dt)
        self.t_vals = np.linspace(t_start, t_end, self.nsteps)

        # Parameters for the system
        self.gamma = gamma
        self.nu = nu
        self.n_th = n_th
        self.eps2 = eps2

        # Grid for 2D probability distribution
        self.x = x
        self.y = y
        self.x0 = x0
        self.y0 = y0

        # Initial conditions and solution storage
        self.init_cond = init_cond
        sol0 = init_cond(x0, y0, eps2)

        self.sol0 = sol0
        self.solution = np.zeros((self.nsteps, 4))
        self.solution[0] = sol0

        # Functions for simulation
        self.ProbDensMap = ProbDensMap
        self.solver = solver

        # Prepare output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize list to store centroid coordinates
        self.centroid_x = []
        self.centroid_y = []

        # Precompute min and max values for consistent color scaling in plots
        self.u_init = ProbDensMap(x, y, sol0)
        self.vmin, self.vmax = np.min(self.u_init), np.max(self.u_init)

    def run_simulation(self):
        # Time integration using RK4 with progress bar
        for t in tqdm(range(1, self.nsteps), desc="Simulating", unit="step"):
            self.solution[t] = self.solver(self.t_vals[t-1], self.solution[t-1], self.dt, self.gamma, self.nu, self.n_th)

            # Compute the analytical solution at the current time step
            u_analytic = self.ProbDensMap(self.x, self.y, self.solution[t])

            # Calculate the center of the probability distribution
            weighted_sum_x = np.sum(self.x[:, None] * u_analytic)  # Sum over x for each y
            weighted_sum_y = np.sum(self.y[None, :] * u_analytic)  # Sum over y for each x
            total_weight = np.sum(u_analytic)  # Total sum (normalization factor)

            # Centroid coordinates
            center_x = weighted_sum_x / total_weight
            center_y = weighted_sum_y / total_weight

            # Store the centroid coordinates
            self.centroid_x.append(center_x)
            self.centroid_y.append(center_y)

            # Save a snapshot every 10 steps
            if t % 10 == 0:
                self.save_snapshot(t, u_analytic)

        # Plot the path of the center of the distribution at the end of the simulation
        self.plot_center_path()

        # Plot the evolution of a(t), b(t), c(t), d(t) over time
        self.plot_parameter_evolution()

    def save_snapshot(self, t, u_analytic):
        # Create a plot for the analytical solution
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(u_analytic.T, extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], origin='lower', aspect='auto', cmap='hot', vmin=self.vmin, vmax=self.vmax)
        ax.set_title(f"Analytical Solution at Time = {t * self.dt:.2f}")
        ax.set_xlabel(r'Re{$\alpha$} (x)')
        ax.set_ylabel(r'Im{$\alpha$} (y)')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probability Density')

        # Plot the path of the center
        ax.plot(self.centroid_x, self.centroid_y, 'w-', label='Center Path', linewidth=2)
        ax.legend(loc="upper right")

        # Save snapshot
        plt.savefig(f'{self.output_dir}/snapshot_{t:04d}.png')
        plt.close(fig)

    def plot_center_path(self):
        # Plot the path of the center of the distribution at the end of the simulation
        plt.figure(figsize=(8, 6))
        plt.plot(self.centroid_x, self.centroid_y, 'k-', label='Path of the Center', linewidth=2)
        plt.xlabel(r"Re{$\alpha$} (x)")
        plt.ylabel(r"Im{$\alpha$} (y)")
        plt.title("Path of the Center of the Probability Distribution")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_parameter_evolution(self):
        # Plot the evolution of a(t), b(t), c(t), d(t) versus time in subplots
        plt.figure(figsize=(10, 8))

        # Subplot for a(t)
        plt.subplot(2, 2, 1)
        plt.plot(self.t_vals, self.solution[:, 0], label="a(t)", color='b')
        plt.xlabel("Time (t)")
        plt.ylabel("a(t)")
        plt.title("Evolution of a(t) over Time")
        plt.legend()
        plt.grid(True)

        # Subplot for b(t)
        plt.subplot(2, 2, 2)
        plt.plot(self.t_vals, self.solution[:, 1], label="b(t)", color='g')
        plt.xlabel("Time (t)")
        plt.ylabel("b(t)")
        plt.title("Evolution of b(t) over Time")
        plt.legend()
        plt.grid(True)

        # Subplot for c(t)
        plt.subplot(2, 2, 3)
        plt.plot(self.t_vals, self.solution[:, 2], label="c(t)", color='r')
        plt.xlabel("Time (t)")
        plt.ylabel("c(t)")
        plt.title("Evolution of c(t) over Time")
        plt.legend()
        plt.grid(True)

        # Subplot for d(t)
        plt.subplot(2, 2, 4)
        plt.plot(self.t_vals, self.solution[:, 3], label="d(t)", color='c')
        plt.xlabel("Time (t)")
        plt.ylabel("d(t)")
        plt.title("Evolution of d(t) over Time")
        plt.legend()
        plt.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plots
        plt.show()

