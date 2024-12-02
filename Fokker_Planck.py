import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import imageio
import os

# Parameters
gamma = 1.0  # Damping coefficient (friction)
D = 0.5  # Diffusion coefficient
x_max = 5.0  # Maximum Re{alpha}
y_max = 5.0  # Maximum Im{alpha}
N_x = 100  # Number of grid points in Re{alpha} space
N_y = 100  # Number of grid points in Im{alpha} space
T_max = 2.0  # Maximum simulation time
dt = 0.001  # Time step
dx = 2 * x_max / N_x  # Re{alpha} grid spacing
dy = 2 * y_max / N_y  # Im{alpha} grid spacing
time_steps = int(T_max / dt)  # Number of time steps
x0 = 1  # Initial value of x
y0 = 1  # Initial value of y
eps = 0.5 # Size of the initial probability distribution

# Discretize the Re{alpha} and Im{alpha} space
x_vals = np.linspace(-x_max, x_max, N_x)
y_vals = np.linspace(-y_max, y_max, N_y)

# Initialize the probability distribution (initial condition)
# Assume an initial Gaussian distribution in Re{alpha} and Im{alpha}
p = np.zeros((N_x, N_y))
for i in range(N_x):
    for j in range(N_y):
        p[i, j] = np.exp(-0.5 * ((x_vals[i]-x0)**2 + (y_vals[j]-y0)**2)/eps**2)

# Function to evolve the Fokker-Planck equation
def evolve_fokker_planck(p, dt, dx, dy, gamma, D):
    # Temporary array for the next time step
    p_new = np.copy(p)
    mid = N_x // 2
    # Loop over Re{alpha} and Im{alpha} grids
    for i in range(1, N_x-1):
        for j in range(1, N_y-1):
            v1 = (i - mid) * (p[i+1, j] - p[i-1, j]) / (2) + p[i, j]
            v2 = (j - mid) * (p[i, j+1] - p[i, j-1]) / (2) + p[i, j]

            d2p_dx2 = (p[i+1, j+1] + p[i-1, j+1] + p[i+1, j-1] + p[i-1, j-1] - 4 * p[i, j]) / dx / dy
            
            # Fokker-Planck equation update
            p_new[i, j] = p[i, j] + dt * (
                gamma * (v1 + v2) + D * d2p_dx2
            )
    
    # Apply boundary conditions (no-flux boundaries)
    p_new[0, :] = p_new[1, :]  # No change at the left boundary
    p_new[-1, :] = p_new[-2, :]  # No change at the right boundary
    p_new[:, 0] = p_new[:, 1]  # No change at the bottom boundary
    p_new[:, -1] = p_new[:, -2]  # No change at the top boundary
    
    return p_new

# Time evolution of the Fokker-Planck equation and storing the data
p_t = np.copy(p)

# Create the plot for each time step with a progress bar
for t in tqdm(range(time_steps), desc="Simulating", unit="step"):
    p_t = evolve_fokker_planck(p_t, dt, dx, dy, gamma, D)
    
    if t % 10 == 0:  # Save a snapshot every 10 steps
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(p_t.T, extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]], origin='lower', aspect='auto', cmap='hot')

        # Create the colorbar
        cbar = plt.colorbar(im, ax=ax, label='Probability Density')

        # Set fixed limits for the color bar
        vmin = np.min(p)  # Minimum value in the initial distribution
        vmax = np.max(p)  # Maximum value in the initial distribution
        im.set_clim(vmin, vmax)

        # Add labels
        ax.set_xlabel(r'Re{$\alpha$} (x)')
        ax.set_ylabel(r'Im{$\alpha$} (y)')
        ax.set_title(f'Time = {t * dt:.2f}')

        # Save the snapshot as PNG
        plt.savefig(f'snapshot_{t:04d}.png')
        plt.close(fig)

print("Fokker-Planck evolution completed!")

def create_animation(png_folder='.',output_video='animation.mp4',fps=1):
    # Get a sorted list of all PNG files in the folder
    png_files = sorted([f for f in os.listdir(png_folder) if f.endswith('.png')])

    # Create a list of image paths
    image_paths = [os.path.join(png_folder, f) for f in png_files]

    # Create the video using imageio
    with imageio.get_writer(output_video, fps=fps) as writer:
        for image_path in image_paths:
            image = imageio.imread(image_path)
            writer.append_data(image)

    print(f"Animation saved as {output_video}")

create_animation()
