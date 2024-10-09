import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 1.0        # mass
k = 100.0      # spring constant
gamma = 0.5    # damping coefficient
T = 1.0        # temperature
dt = 0.01      # time step
total_time = 1000.0  # total simulation time

# Derived parameters
beta = 1.0 / (T)  # Inverse temperature
sigma = np.sqrt(2 * gamma * k * T / dt)  # Strength of Langevin force

# Values of omega_mod to evaluate
omega_mod_values = [10, 20, 50]

# Create a figure for subplots
for omega_mod in omega_mod_values:
    # Initialize arrays for each omega_mod
    num_steps = int(total_time / dt)
    time = np.linspace(0, total_time, num_steps)
    position = np.zeros(num_steps)
    velocity = np.zeros(num_steps)

    # Initial conditions
    position[0] = 1.0  # initial position
    velocity[0] = 0.0  # initial velocity

    # Simulation loop
    for i in range(1, num_steps):
        # Langevin force with time-dependent spring constant
        force_noise = np.random.normal(0, sigma)
        acceleration = -(k * (1 + 0.1 * np.cos(i * dt * omega_mod)) / m) * position[i-1] - (gamma / m) * velocity[i-1] + force_noise
        velocity[i] = velocity[i-1] + acceleration * dt
        position[i] = position[i-1] + velocity[i] * dt

    # Compute the Fourier Transform of the position
    frequency = np.fft.fftfreq(num_steps, dt)
    position_spectrum = np.fft.fft(position)

    # Focus on the peak region
    peak_freq_range = (0.0, 2.0)  # Define the frequency range around the peak
    indices = np.where((frequency >= peak_freq_range[0]) & (frequency <= peak_freq_range[1]))

    # Create a new figure for this omega_mod
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'Harmonic Oscillator with Langevin Force - $\\omega_{{mod}}$ = {omega_mod}', fontsize=16)

    # Subplot for time-domain information (only position)
    axs[0].plot(time, position, label='Position', color='blue')
    axs[0].set_title('Time-Domain Information (Position Only)')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Position')
    axs[0].legend()
    axs[0].grid()

    # Subplot for spectrum around the peak
    axs[1].plot(frequency[indices]*2*np.pi, np.abs(position_spectrum[indices]), color='green')
    axs[1].set_title('Position Spectrum Near the Peak')
    axs[1].set_xlabel(r'Angular Frequency $\omega$')
    axs[1].set_ylabel('Magnitude')
    axs[1].grid()

    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the main title
    plt.savefig(f'harmonic_oscillator_omega_mod_{omega_mod}.png')  # Save the figure
    plt.close()  # Close the figure to free memory
