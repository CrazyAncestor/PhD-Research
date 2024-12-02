import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import os  # For handling directories

def FTCS_2d(U, D, dx, dy, dt):
    """
    Solve the 2D diffusion equation using the FTCS method (explicit).
    
    Parameters:
    U (numpy array): Initial concentration profile (2D array)
    D (float): Diffusion coefficient
    dx (float): Grid spacing in x direction
    dy (float): Grid spacing in y direction
    dt (float): Time step
    
    Returns:
    U_new_new (numpy array): Updated concentration profile after nsteps
    """
    Nx, Ny = U.shape  # Number of grid points in x and y directions
    alpha_x = D * dt / 2 / dx**2
    alpha_y = D * dt / 2 / dy**2

    U_new_new = np.copy(U)
    for i in range(1, (Nx - 1)):
        for j in range(1, (Ny - 1)):
            U_new_new[i, j] = U[i, j] + alpha_x * (U[i - 1, j] - 2 * U[i, j] + U[i + 1, j]) + alpha_y * (U[i, j - 1] - 2 * U[i, j] + U[i, j + 1])

    U_new_new[0, :] = U_new_new[1, :]  # No change at the left boundary
    U_new_new[-1, :] = U_new_new[-2, :]  # No change at the right boundary
    U_new_new[:, 0] = U_new_new[:, 1]  # No change at the bottom boundary
    U_new_new[:, -1] = U_new_new[:, -2]  # No change at the top boundary

    return U_new_new  # Yield the updated concentration field at each time step
def BTCS_2d(U, D, dx, dy, dt):
    """
    Solve the 2D diffusion equation using the BTCS method (implicit).
    
    Parameters:
    U (numpy array): Initial concentration profile (2D array)
    D (float): Diffusion coefficient
    dx (float): Grid spacing in x direction
    dy (float): Grid spacing in y direction
    dt (float): Time step
    
    Returns:
    U_new_new (numpy array): Updated concentration profile after nsteps
    """
    Nx, Ny = U.shape  # Number of grid points in x and y directions
    alpha_x = D * dt /2. / dx**2
    alpha_y = D * dt /2. / dy**2

    # Build the coefficient matrix for the system of equations for x-direction
    Ax = (1 + 2 * alpha_x) * np.eye(Nx-2)  # Diagonal
    Ax += -alpha_x * np.diag(np.ones(Nx-3), 1)  # Upper diagonal
    Ax += -alpha_x * np.diag(np.ones(Nx-3), -1)  # Lower diagonal

    # Build the coefficient matrix for the system of equations for y-direction
    Ay = (1 + 2 * alpha_y) * np.eye(Ny-2)  # Diagonal
    Ay += -alpha_y * np.diag(np.ones(Ny-3), 1)  # Upper diagonal
    Ay += -alpha_y * np.diag(np.ones(Ny-3), -1)  # Lower diagonal

    # Solve the linear system for each row (x-direction)
    U0 = np.copy(U[1:-1, 1:-1])
    U1 = []
    for i in range(Nx-2):
        us = np.copy(U0[i, :])
        for j in range(Ny-2):
            us[j] = alpha_y * U[i+2,j+1] + (1 - 2 * alpha_y) * U[i+1,j+1] + alpha_y * U[i,j+1]
        u_new = np.linalg.solve(Ax, us)
        U1.append(u_new)
    U1 = np.array(U1)

    # Solve the linear system for each column (y-direction)
    U1_ext = np.copy(U)
    U1_ext[1:-1, 1:-1] = U1
    U2 = []
    for j in range(Ny-2):
        us = np.copy(U1[:, j])
        for i in range(Nx-2):
            us[i] = alpha_x * U1_ext[i+1,j+2] + (1 - 2 * alpha_x) * U1_ext[i+1,j+1] + alpha_y * U1_ext[i+1,j]
        u_new = np.linalg.solve(Ay, us)
        U2.append(u_new)
    U2 = np.array(U2).T  # Transpose back to 2D array
    
    U_final = np.copy(U)
    U_final[1:-1, 1:-1] = U2
    """U_final[0, :] = U_final[1, :]  # No change at the left boundary
    U_final[-1, :] = U_final[-2, :]  # No change at the right boundary
    U_final[:, 0] = U_final[:, 1]  # No change at the bottom boundary
    U_final[:, -1] = U_final[:, -2]  # No change at the top boundary"""

    return U_final

def analytical_solution(x, y, t, D, Lx, Ly, eps):
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
    t0 = eps**2 / (4 * D)
    X, Y = np.meshgrid(x, y)
    u_analytic = np.exp(-0.5 * ((X - Lx / 2)**2 + (Y - Ly / 2)**2) / (4 * D * (t0 + t))) * (t0 / (t0 + t))**0.5
    return u_analytic

# Parameters
Lx, Ly = 10.0, 10.0   # Domain size in x and y directions
T = 1.0               # Total time
D = 1               # Diffusion coefficient
Nx, Ny = 100, 100     # Grid points in x and y directions
dx, dy = Lx/(Nx-1), Ly/(Ny-1)  # Grid spacing
dt = 0.01             # Time step
nsteps = int(T / dt)  # Number of time steps
eps = 0.5             # Size of initial distribution

# Initial condition (Gaussian distribution)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
u_init = np.exp(-0.5 * ((X - Lx/2)**2 + (Y - Ly/2)**2) / eps**2)  # Gaussian distribution
u_t = np.copy(u_init)

# Create directory for saving snapshots
output_dir = "simulation_result"
os.makedirs(output_dir, exist_ok=True)  # Create directory for simulation result snapshots

# Create the plot for each time step with a progress bar
for t in tqdm(range(nsteps), desc="Simulating", unit="step"):
    u_t = BTCS_2d(u_t, D, dx, dy, dt)
    
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
        plt.close(fig)
