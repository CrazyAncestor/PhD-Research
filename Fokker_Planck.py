import numpy as np
import matplotlib.pyplot as plt

# Parameters
gamma = 1.0  # Damping coefficient (friction)
D = 1.0  # Diffusion coefficient
x_max = 5.0  # Maximum position
v_max = 5.0  # Maximum velocity
N_x = 100  # Number of grid points in position space
N_v = 100  # Number of grid points in velocity space
T_max = 2.0  # Maximum simulation time
dt = 0.001  # Time step
dx = 2 * x_max / N_x  # Position grid spacing
dv = 2 * v_max / N_v  # Velocity grid spacing
time_steps = int(T_max / dt)  # Number of time steps
x0 = 1
v0 = 1


# Discretize the position and velocity space
x_vals = np.linspace(-x_max, x_max, N_x)
v_vals = np.linspace(-v_max, v_max, N_v)

# Initialize the probability distribution (initial condition)
# Assume an initial Gaussian distribution in position and velocity
p = np.zeros((N_x, N_v))
for i in range(N_x):
    for j in range(N_v):
        p[i, j] = np.exp(-0.5 * ((x_vals[i]-x0)**2 + (v_vals[j]-v0)**2))

# Function to evolve the Fokker-Planck equation
def evolve_fokker_planck(p, dt, dx, dv, gamma, D):
    # Temporary array for the next time step
    p_new = np.copy(p)
    mid = N_x//2
    # Loop over position and velocity grids
    for i in range(1, N_x-1):
        for j in range(1, N_v-1):
            v1 = (i - mid) * (p[i+1,j]-p[i-1,j]) / (2 ) + p[i,j]
            v2 = (j - mid) * (p[i,j+1]-p[i,j-1]) / (2 ) + p[i,j]

            dp_dv2 = (p[i+1,j+1] + p[i-1,j+1] + p[i+1,j-1] + p[i-1,j-1] - 4 * p[i,j])/dx/dv
            
            # Fokker-Planck equation update
            p_new[i, j] = p[i, j] + dt * (
                gamma * (v1 + v2) + D * dp_dv2
            )
    
    # Apply boundary conditions (no-flux boundaries)
    p_new[0, :] = p_new[1, :]  # No change at the left boundary
    p_new[-1, :] = p_new[-2, :]  # No change at the right boundary
    p_new[:, 0] = p_new[:, 1]  # No change at the bottom boundary
    p_new[:, -1] = p_new[:, -2]  # No change at the top boundary
    
    return p_new

# Time evolution of the Fokker-Planck equation
p_t = np.copy(p)

# Set up the plot for animation
plt.figure(figsize=(8, 6))

# Loop over time steps
for t in range(time_steps):
    p_t = evolve_fokker_planck(p_t, dt, dx, dv, gamma, D)
    
    if t % 10 == 0:  # Plot every 100 steps
        plt.clf()
        plt.imshow(p_t.T, extent=[x_vals[0], x_vals[-1], v_vals[0], v_vals[-1]], origin='lower', aspect='auto', cmap='hot')
        plt.colorbar(label='Probability Density')
        plt.xlabel('Position (x)')
        plt.ylabel('Velocity (v)')
        plt.title(f'Time = {t * dt:.2f}')
        plt.pause(0.1)

plt.show()
