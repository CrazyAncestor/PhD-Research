import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class FokkerPlanckSimulator:
    def __init__(self, t_start, t_end, dt, x, y, phys_parameter, init_cond, output_dir, ProbDensMap, solver):
        # Simulation time settings
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.nsteps = int((t_end - t_start) / dt)
        self.t_vals = np.linspace(t_start, t_end, self.nsteps)

        # Grid for 2D probability distribution
        self.x = x
        self.y = y

        # Initial conditions and solution storage
        self.phys_parameter = phys_parameter
        self.init_cond = init_cond

        self.solution = np.zeros((self.nsteps, len(init_cond)))
        self.solution[0] = init_cond

        # Functions for simulation
        self.ProbDensMap = ProbDensMap
        self.solver = solver

        # Prepare output directory
        self.output_dir = output_dir

        # Initialize list to store centroid coordinates
        self.centroid_x = []
        self.centroid_y = []


    def run_simulation(self, pure_parameter = False):

        os.makedirs(self.output_dir, exist_ok=True)

        # Open a log file to write simulation messages
        log_filename = os.path.join(self.output_dir, "simulation_log.txt")
        with open(log_filename, 'w') as log_file:
            log_file.write(f"Simulation started at {self.t_start} with time steps from {self.t_start} to {self.t_end}.\n")

            if pure_parameter:
                log_file.write("Running with pure_parameter mode enabled.\n")
                for t in tqdm(range(1, self.nsteps), desc="Simulating", unit="step"):
                    self.solution[t] = self.solver(self.t_vals[t-1], self.solution[t-1], self.dt, self.phys_parameter)
                # Plot the evolution of a(t), b(t), c(t), d(t) over time
                self.plot_parameter_evolution()
                log_file.write("Simulation completed.\n")
            else:
                # Precompute min and max values for consistent color scaling in plots
                self.u_init = self.ProbDensMap(self.x, self.y, self.init_cond)
                self.vmin, self.vmax = np.min(self.u_init), np.max(self.u_init)

                log_file.write(f"Initial probability density range: vmin={self.vmin}, vmax={self.vmax}.\n")
                
                # Time integration using RK4 with progress bar
                for t in tqdm(range(1, self.nsteps), desc="Simulating", unit="step"):
                    self.solution[t] = self.solver(self.t_vals[t-1], self.solution[t-1], self.dt, self.phys_parameter)

                    # Compute the Probability Distribution at the current time step
                    ProbDens = self.ProbDensMap(self.x, self.y, self.solution[t])

                    # Calculate the center of the probability distribution
                    weighted_sum_x = np.sum(self.x[:, None] * ProbDens)  # Sum over x for each y
                    weighted_sum_y = np.sum(self.y[None, :] * ProbDens)  # Sum over y for each x
                    total_weight = np.sum(ProbDens)  # Total sum (normalization factor)

                    # Centroid coordinates
                    center_x = weighted_sum_x / total_weight
                    center_y = weighted_sum_y / total_weight

                    # Store the centroid coordinates
                    self.centroid_x.append(center_x)
                    self.centroid_y.append(center_y)

                    # Inside your run_simulation method (where the centroid and ProbDensSum are calculated)
                    ProbDensSum = self.volume_integration(self.x, self.y, ProbDens)

                    # Log the information and check for deviation from 1.0
                    self.log_warning_message(t, ProbDensSum, center_x, center_y, log_file)

                    # Save a snapshot every 10 steps
                    if t % 10 == 0:
                        self.save_snapshot(t, ProbDens)
                        log_file.write(f"Snapshot saved for time step {t}.\n")

                # Plot the path of the center of the distribution at the end of the simulation
                self.plot_center_path()
                log_file.write("Path of the center plot generated.\n")

                # Plot the evolution of a(t), b(t), c(t), d(t) over time
                self.plot_parameter_evolution()
                log_file.write("Parameter evolution plots generated.\n")
            
            log_file.write(f"Simulation ended at {self.t_end}.\n")
            log_file.write("Simulation completed successfully.\n")

    def volume_integration(self, x, y, Dens):
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        Sum = np.sum(Dens) * np.abs(dx) * np.abs(dy)
        return Sum
    
    def log_warning_message(self, t, ProbDensSum, center_x, center_y, log_file):
        # Calculate the difference from 1.0
        diff_from_one = abs(ProbDensSum - 1.0)

        # Check if the difference is more than 1% away from 1.0
        if diff_from_one > 0.01:
            log_file.write(f"WARNING: Step {t}, Time = {self.t_vals[t]:.2f}, ProbDensSum = {ProbDensSum:.4f}, Center = ({center_x:.4f}, {center_y:.4f})\nProbability summation on the plane has deviation larger than 1.0%! Simulation Box may be too small.\n")
        else:
            log_file.write(f"Step {t}, Time = {self.t_vals[t]:.2f}, ProbDensSum = {ProbDensSum:.4f}, Center = ({center_x:.4f}, {center_y:.4f})\n")

    
    def save_snapshot(self, t, ProbDens):
        # Create a plot for the Probability Distribution
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(ProbDens.T, extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], origin='lower', aspect='auto', cmap='hot', vmin=self.vmin, vmax=self.vmax)
        ax.set_title(f"Probability Distribution on Coherent State Plane at Time = {t * self.dt:.2f}")
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
        # Plot the evolution of b(t), c(t), d(t), e(t) versus time in subplots
        plt.figure(figsize=(10, 8))

        # Subplot for b(t)
        plt.subplot(2, 2, 1)
        plt.plot(self.t_vals, self.solution[:, 1], label="b(t)", color='g')
        plt.xlabel("Time (t)")
        plt.ylabel("b(t)")
        plt.title("Evolution of b(t) over Time")
        plt.legend()
        plt.grid(True)

        # Subplot for c(t)
        plt.subplot(2, 2, 2)
        plt.plot(self.t_vals, self.solution[:, 2], label="c(t)", color='r')
        plt.xlabel("Time (t)")
        plt.ylabel("c(t)")
        plt.title("Evolution of c(t) over Time")
        plt.legend()
        plt.grid(True)

        # Subplot for d(t)
        plt.subplot(2, 2, 3)
        plt.plot(self.t_vals, self.solution[:, 3], label="d(t)", color='c')
        plt.xlabel("Time (t)")
        plt.ylabel("d(t)")
        plt.title("Evolution of d(t) over Time")
        plt.legend()
        plt.grid(True)

        # Subplot for e(t)
        plt.subplot(2, 2, 4)
        plt.plot(self.t_vals, self.solution[:, 4], label="e(t)", color='m')  # e(t) is solution[:, 4]
        plt.xlabel("Time (t)")
        plt.ylabel("e(t)")
        plt.title("Evolution of e(t) over Time")
        plt.legend()
        plt.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plots
        plt.show()

            # Plot the evolution of f(t), h(t), k(t), l(t) in a new figure
        plt.figure(figsize=(10, 8))

        # Subplot for f(t)
        plt.subplot(2, 2, 1)
        plt.plot(self.t_vals, self.solution[:, 5], label="f(t)", color='b')
        plt.xlabel("Time (t)")
        plt.ylabel("f(t)")
        plt.title("Evolution of f(t) over Time")
        plt.legend()
        plt.grid(True)

        # Subplot for h(t)
        plt.subplot(2, 2, 2)
        plt.plot(self.t_vals, self.solution[:, 6], label="h(t)", color='orange')
        plt.xlabel("Time (t)")
        plt.ylabel("h(t)")
        plt.title("Evolution of h(t) over Time")
        plt.legend()
        plt.grid(True)

        # Subplot for k(t)
        plt.subplot(2, 2, 3)
        plt.plot(self.t_vals, self.solution[:, 7], label="k(t)", color='purple')
        plt.xlabel("Time (t)")
        plt.ylabel("k(t)")
        plt.title("Evolution of k(t) over Time")
        plt.legend()
        plt.grid(True)

        # Subplot for l(t)
        plt.subplot(2, 2, 4)
        plt.plot(self.t_vals, self.solution[:, 8], label="l(t)", color='brown')
        plt.xlabel("Time (t)")
        plt.ylabel("l(t)")
        plt.title("Evolution of l(t) over Time")
        plt.legend()
        plt.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plots for f, h, k, l
        plt.show()