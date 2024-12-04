import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import os  # For handling directories

import numpy as np
import matplotlib.pyplot as plt

# Define the system of differential equations
def system(t, y, gamma, nu, n_th):
    a, b, c, d = y
    a_prime = gamma * (1 + d + (b**2 + c**2)/4)
    b_prime = gamma * b/2 + gamma * n_th * b * d + nu * c
    c_prime = gamma * c/2 + gamma * n_th * c * d - nu * b
    d_prime = gamma * (d + n_th * d**2)
    
    return np.array([a_prime, b_prime, c_prime, d_prime])

# RK4 Solver
def rk4_step(t, y, h, gamma, nu, n_th):
    # Get the k1, k2, k3, k4 slopes
    k1 = h * system(t, y, gamma, nu, n_th)
    k2 = h * system(t + h/2, y + k1/2, gamma, nu, n_th)
    k3 = h * system(t + h/2, y + k2/2, gamma, nu, n_th)
    k4 = h * system(t + h, y + k3, gamma, nu, n_th)
    
    # Update the solution using the weighted average
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# Initial conditions
# Constants
eps2 = 0.1
gamma = 1.0  # Example value for gamma
nu = 3     # Example value for nu
n_th = 0.01      # Example value for n
x0 = 2
y0 = 2


a0 = - (x0 * x0 + y0 * y0)/eps2   # Initial condition for a
b0 = 2 * x0 /eps2  # Initial condition for b
c0 = 2 * y0 /eps2  # Initial condition for c
d0 = - 1/eps2  # Initial condition for d
y0 = np.array([a0, b0, c0, d0])


# Time settings
t_start = 0   # Start time
t_end = 10    # End time
h = 0.01       # Step size
nsteps = int((t_end - t_start) / h)
t_points = int((t_end - t_start) / h)  # Number of steps

Lx, Ly = 100.0, 100.0   # Domain size in x and y directions
Nx, Ny = 1000, 1000     # Grid points in x and y directions
dx, dy = Lx/(Nx-1), Ly/(Ny-1)  # Grid spacing

# Initial condition (Gaussian distribution)
x = np.linspace(-Lx, Lx, Nx)
y = np.linspace(-Lx, Ly, Ny)
X, Y = np.meshgrid(x, y)

def analytical_solution(x, y, solution):
    """
    Generate the analytical solution for the 2D diffusion equation at time t.
    
    Parameters:
    x (numpy array): Grid points in x direction
    y (numpy array): Grid points in y direction
    t (float): Time at which to evaluate the solution
    D (float): Diffusion coefficient
    Lx (float): Length of the domain in x direction
    Ly (float): Length of the domain in y direction
    
    Returns:
    u_analytic (numpy array): Analytical solution at time t
    """
    a = solution[0]
    b = solution[1]
    c = solution[2]
    d = solution[3]

    X, Y = np.meshgrid(x, y)
    u_analytic = np.exp(a + b * X + c * Y + d * (X**2 + Y**2))
    return u_analytic


# Prepare array to store solutions
t_vals = np.linspace(t_start, t_end, t_points)
solution = y0  # Initial state
u_init = analytical_solution(x, y, solution)

# Create directory for saving snapshots
output_dir = "finite_ele_result"
os.makedirs(output_dir, exist_ok=True)  # Create directory for simulation result snapshots

# Create the plot for each time step with a progress bar
for t in tqdm(range(nsteps), desc="Simulating", unit="step"):

    solution = rk4_step(t*h + t_start, solution, h, gamma, nu, n_th)
    
    # Save a snapshot of the solution every 10 steps
    if t % 10 == 0:  
        # Compute the analytical solution at this time step
        
        u_analytic = analytical_solution(x, y, solution)
        
        # Find the global min and max values for the color limits to ensure consistency across plots
        vmin = min(np.min(u_init), np.min(u_init))
        vmax = max(np.max(u_init), np.max(u_init))

        # Create a combined figure with two subplots
        fig, (ax2) = plt.subplots(1, 1, figsize=(8, 6))

        # Plot the analytical solution
        im2 = ax2.imshow(u_analytic.T, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='auto', cmap='hot', vmin=vmin, vmax=vmax)
        ax2.set_title(f"Analytical Solution at Time = {t * h:.2f}")
        ax2.set_xlabel(r'Re{$\alpha$} (x)')
        ax2.set_ylabel(r'Im{$\alpha$} (y)')
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Probability Density')

        # Save the snapshot of both the simulation and analytical solutions as PNG
        plt.savefig(f'{output_dir}/snapshot_{t:04d}.png')
        plt.close(fig)

"""# Time integration using RK4
t = t_start
for i in range(1, t_points):
    solution[i] = rk4_step(t, solution[i-1], h, gamma, nu, n_th)
    t += h

# Plot the results
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(t_vals, solution[:, 0], label="a(t)")
plt.xlabel("Time (t)")
plt.ylabel("a(t)")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t_vals, solution[:, 1], label="b(t)")
plt.xlabel("Time (t)")
plt.ylabel("b(t)")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t_vals, solution[:, 2], label="c(t)")
plt.xlabel("Time (t)")
plt.ylabel("c(t)")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t_vals, solution[:, 3], label="d(t)")
plt.xlabel("Time (t)")
plt.ylabel("d(t)")
plt.legend()

plt.tight_layout()
plt.show()
"""

"""# Parameters

X0, Y0 = 2.0, 2.0
T = 10.0               # Total time
D = 1               # Diffusion coefficient
gamma = 1

u_init = np.exp(-0.5 * ((X - Lx/2 - X0)**2 + (Y - Ly/2 - Y0)**2) / eps**2)  # Gaussian distribution
u_t = np.copy(u_init)


# Create the plot for each time step with a progress bar
for t in tqdm(range(nsteps), desc="Simulating", unit="step"):
    u_t = BTCS_2d(u_t, D, gamma, dx, dy, dt)
    
    # Save a snapshot of the solution every 10 steps
    if t % 10 == 0:  
        # Compute the analytical solution at this time step
        u_analytic = analytical_solution(x, y, t * dt, D, Lx, Ly, eps)
        
        # Find the global min and max values for the color limits to ensure consistency across plots
        vmin = min(np.min(u_init), np.min(u_init))
        vmax = max(np.max(u_init), np.max(u_init))

        # Create a combined figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot the simulation (BTCS) solution
        im1 = ax1.imshow(u_t.T, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='auto', cmap='hot', vmin=vmin, vmax=vmax)
        ax1.set_title(f"BTCS Solution at Time = {t * dt:.2f}")
        ax1.set_xlabel(r'Re{$\alpha$} (x)')
        ax1.set_ylabel(r'Im{$\alpha$} (y)')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Probability Density')

        # Plot the analytical solution
        im2 = ax2.imshow(u_analytic.T, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='auto', cmap='hot', vmin=vmin, vmax=vmax)
        ax2.set_title(f"Analytical Solution at Time = {t * dt:.2f}")
        ax2.set_xlabel(r'Re{$\alpha$} (x)')
        ax2.set_ylabel(r'Im{$\alpha$} (y)')
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Probability Density')

        # Save the snapshot of both the simulation and analytical solutions as PNG
        plt.savefig(f'{output_dir}/snapshot_{t:04d}.png')
        plt.close(fig)"""