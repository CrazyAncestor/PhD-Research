import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import os  # For handling directories

# Define the system of differential equations
def system(t, y, gamma, nu, n_th):
    a, b, c, d = y
    a_prime = gamma * (1 + n_th * (d + (b**2 + c**2)/4))
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
eps2 = 0.25
gamma = 1.0  # Example value for gamma
nu = 3     # Example value for nu
n_th = 1      # Example value for n
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
t_vals = np.linspace(t_start, t_end, nsteps)

# Initialize solution array
solution = np.zeros((nsteps, 4))  # 4 variables for each time step
solution[0] = y0  # Initial condition at t_start

# Precompute grid values (they don't change during simulation)
Lx, Ly = 10.0, 10.0   # Domain size in x and y directions
Nx, Ny = 100, 100     # Grid points in x and y directions
dx, dy = Lx/(Nx-1), Ly/(Ny-1)  # Grid spacing
x = np.linspace(-Lx, Lx, Nx)
y = np.linspace(-Lx, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Analytical solution function
def analytical_solution(x, y, solution):
    a, b, c, d = solution
    u_analytic = np.exp(a + b * X + c * Y + d * (X**2 + Y**2))
    return u_analytic

# Prepare array to store solutions
u_init = analytical_solution(x, y, y0)

# Precompute the min and max values for consistent color scaling
vmin, vmax = np.min(u_init), np.max(u_init)

# Create directory for saving snapshots
output_dir = "finite_ele_result"
os.makedirs(output_dir, exist_ok=True)  # Create directory for simulation result snapshots

# Initialize a list to store the center coordinates
centroid_x = []
centroid_y = []

# Time integration using RK4 with progress bar
for t in tqdm(range(1, nsteps), desc="Simulating", unit="step"):
    solution[t] = rk4_step(t_vals[t-1], solution[t-1], h, gamma, nu, n_th)

    # Compute the analytical solution at the current time step
    u_analytic = analytical_solution(x, y, solution[t])

    # Calculate the center of the probability distribution
    weighted_sum_x = np.sum(x[:, None] * u_analytic)  # Sum over x for each y
    weighted_sum_y = np.sum(y[None, :] * u_analytic)  # Sum over y for each x
    total_weight = np.sum(u_analytic)  # Total sum (normalization factor)

    # Centroid coordinates
    center_x = weighted_sum_x / total_weight
    center_y = weighted_sum_y / total_weight

    # Store the centroid coordinates
    centroid_x.append(center_x)
    centroid_y.append(center_y)

    # Save a snapshot every 10 steps
    if t % 10 == 0:
        # Create a plot for the analytical solution
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(u_analytic.T, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='auto', cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_title(f"Analytical Solution at Time = {t * h:.2f}")
        ax.set_xlabel(r'Re{$\alpha$} (x)')
        ax.set_ylabel(r'Im{$\alpha$} (y)')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probability Density')

        # Plot the path of the center
        ax.plot(centroid_x, centroid_y, 'w-', label='Center Path', linewidth=2)
        ax.legend(loc="upper right")

        # Save snapshot
        plt.savefig(f'{output_dir}/snapshot_{t:04d}.png')
        plt.close(fig)

# Plot the path of the center of the distribution at the end of the simulation
plt.figure(figsize=(8, 6))
plt.plot(centroid_x, centroid_y, 'k-', label='Path of the Center', linewidth=2)
plt.xlabel(r"Re{$\alpha$} (x)")
plt.ylabel(r"Im{$\alpha$} (y)")
plt.title("Path of the Center of the Probability Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot the evolution of a(t), b(t), c(t), d(t) versus time in subplots
plt.figure(figsize=(10, 8))

# Subplot for a(t)
plt.subplot(2, 2, 1)
plt.plot(t_vals, solution[:, 0], label="a(t)", color='b')
plt.xlabel("Time (t)")
plt.ylabel("a(t)")
plt.title("Evolution of a(t) over Time")
plt.legend()
plt.grid(True)

# Subplot for b(t)
plt.subplot(2, 2, 2)
plt.plot(t_vals, solution[:, 1], label="b(t)", color='g')
plt.xlabel("Time (t)")
plt.ylabel("b(t)")
plt.title("Evolution of b(t) over Time")
plt.legend()
plt.grid(True)

# Subplot for c(t)
plt.subplot(2, 2, 3)
plt.plot(t_vals, solution[:, 2], label="c(t)", color='r')
plt.xlabel("Time (t)")
plt.ylabel("c(t)")
plt.title("Evolution of c(t) over Time")
plt.legend()
plt.grid(True)

# Subplot for d(t)
plt.subplot(2, 2, 4)
plt.plot(t_vals, solution[:, 3], label="d(t)", color='c')
plt.xlabel("Time (t)")
plt.ylabel("d(t)")
plt.title("Evolution of d(t) over Time")
plt.legend()
plt.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()