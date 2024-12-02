import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar

def btcs_2d(U, D, dx, dy, dt):
    """
    Solve the 2D diffusion equation using the BTCS method (implicit).
    
    Parameters:
    U (numpy array): Initial concentration profile (2D array)
    D (float): Diffusion coefficient
    dx (float): Grid spacing in x direction
    dy (float): Grid spacing in y direction
    dt (float): Time step
    nsteps (int): Number of time steps to run
    
    Returns:
    U_new_new (numpy array): Updated concentration profile after nsteps
    """
    Nx, Ny = U.shape  # Number of grid points in x and y directions
    alpha_x = D * dt / 2 / dx**2
    alpha_y = D * dt / 2 / dy**2

    # Build the coefficient matrix for the system of equations for x-direction
    Ax = (1 + 2 * alpha_x) * np.eye(Nx)  # Diagonal
    Ax += -alpha_x * np.diag(np.ones(Nx-1), 1)  # Upper diagonal
    Ax += -alpha_x * np.diag(np.ones(Nx-1), -1)  # Lower diagonal

    # Build the coefficient matrix for the system of equations for y-direction
    Ay = (1 + 2 * alpha_y) * np.eye(Ny)  # Diagonal
    Ay += -alpha_y * np.diag(np.ones(Ny-1), 1)  # Upper diagonal
    Ay += -alpha_y * np.diag(np.ones(Ny-1), -1)  # Lower diagonal


    # Solve the linear system for each row (x-direction)
    U_new = []
    for i in range(Nx):
        u = U[i, :]
        u_new = np.linalg.solve(Ax, u)
        U_new.append(u_new)
    U_new = np.array(U_new)

    # Solve the linear system for each column (y-direction)
    U_new_new = []
    for j in range(Ny):
        u = U_new[:, j]
        u_new = np.linalg.solve(Ay, u)
        U_new_new.append(u_new)
    U_new_new = np.array(U_new_new).T  # Transpose back to 2D array

    return U_new_new  # Yield the updated concentration field at each time step

# Parameters
Lx, Ly = 10.0, 10.0   # Domain size in x and y directions
T = 2.0               # Total time
D = 1                 # Diffusion coefficient
Nx, Ny = 100, 100     # Grid points in x and y directions
dx, dy = Lx/(Nx-1), Ly/(Ny-1)  # Grid spacing
dt = 0.01             # Time step
nsteps =  int(T / dt)  # Number of time steps

# Initial condition (Gaussian distribution)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
u_init = np.exp(-0.5 * ((X - Lx/2)**2 + (Y - Ly/2)**2) / 0.5**2)  # Gaussian distribution
u_t = np.copy(u_init)

# Create the plot for each time step with a progress bar
for t in tqdm(range(nsteps), desc="Simulating", unit="step"):
    u_t = btcs_2d(u_t, D, dx, dy, dt)
    
    if t % 10 == 0:  # Save a snapshot every 10 steps
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(u_t.T, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='auto', cmap='hot')

        # Create the colorbar
        cbar = plt.colorbar(im, ax=ax, label='Probability Density')

        # Set fixed limits for the color bar
        vmin = np.min(u_init)  # Minimum value in the initial distribution
        vmax = np.max(u_init)  # Maximum value in the initial distribution
        im.set_clim(vmin, vmax)

        # Add labels
        ax.set_xlabel(r'Re{$\alpha$} (x)')
        ax.set_ylabel(r'Im{$\alpha$} (y)')
        ax.set_title(f'Time = {t * dt:.2f}')

        # Save the snapshot as PNG
        plt.savefig(f'snapshot_{t:04d}.png')
        plt.close(fig)
