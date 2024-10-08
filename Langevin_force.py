import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 1.0        # mass
k = 1.0        # spring constant
gamma = 0.5    # damping coefficient
T = 1.0        # temperature
dt = 0.01      # time step
total_time = 100.0  # total simulation time

# Derived parameters
beta = 1.0 / (T)  # Inverse temperature
sigma = np.sqrt(2 * gamma * k * T / dt)  # Strength of Langevin force

# Initialize arrays
num_steps = int(total_time / dt)
time = np.linspace(0, total_time, num_steps)
position = np.zeros(num_steps)
velocity = np.zeros(num_steps)

# Initial conditions
position[0] = 1.0  # initial position
velocity[0] = 0.0  # initial velocity

# Simulation loop
for i in range(1, num_steps):
    # Langevin force
    force_noise = np.random.normal(0, sigma)
    
    # Update velocity and position using the Euler method
    acceleration = -(k / m) * position[i-1] - (gamma / m) * velocity[i-1] + force_noise
    velocity[i] = velocity[i-1] + acceleration * dt
    position[i] = position[i-1] + velocity[i] * dt

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(time, position, label='Position', color='blue')
plt.plot(time, velocity, label='Velocity', color='orange')
plt.title('Harmonic Oscillator with Langevin Force')
plt.xlabel('Time')
plt.ylabel('Position / Velocity')
plt.legend()
plt.grid()
plt.show()
