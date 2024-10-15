import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parameters for oscillator 1
Gamma_a = 0.01   # damping coefficient of oscillator 1
wa = 1.0  # spring constant of oscillator 1 (imaginary)

# Parameters for oscillator 2
Gamma_b = 0.01   # damping coefficient of oscillator 2
wb = 1.0  # spring constant of oscillator 2 (imaginary)

# Coupling constants (real for interaction)
g = 0.2  # coupling constant(Rabi Oscillation frequency)

# Parameters for the input probe pulse
A = 100.0    # Amplitude of the probe pulse
wp = 3      # Central frequency of the probe pulse
tp = 1   # Time duration of the probe pulse

# Simulation time parameter
T_tot = 1000    # total time: unit ps
N_tot = 10000   # number of timesteps

def input_probe_pulse_force(t):
    return A * np.exp(1j * wp * t) * np.exp(-(t / tp) ** 2 / 2)

# Calculate the pulse
t_pulse = np.linspace(0, tp * 5, N_tot)
pulse = input_probe_pulse_force(t_pulse)

# Plotting the real part of the pulse
plt.figure(figsize=(10, 6))
plt.plot(t_pulse, np.real(pulse), label='Real Part of Pulse', color='blue')
plt.title('Input Probe Pulse Electric Field', fontsize=14)  # Increase title font size
plt.xlabel('Time (ps)', fontsize=12)  # Increase x-label font size
plt.ylabel('Pulse Amplitude', fontsize=12)  # Increase y-label font size
plt.grid()
plt.legend()
plt.show()

# Define the differential equations in a, b
def coupled_oscillators_a(y, t):
    a, b = y
    fa = input_probe_pulse_force(t)
    dadt = -Gamma_a * a - wa * 1j * a - g * 1j * b + fa
    dbdt = -Gamma_b * b - wb * 1j * b - g * 1j * a
    return np.array([dadt, dbdt])

def rk4(deriv, y0, t):
    n = len(t)  # number of time steps
    y = np.zeros((n, len(y0))) * (0.0 + 0.0j)  # initialize solution array
    y[0] = y0  # set initial conditions

    for i in range(1, n):
        dt = t[i] - t[i-1]  # time step
        k1 = dt * deriv(y[i-1], t[i-1])
        k2 = dt * deriv(y[i-1] + 0.5 * k1, t[i-1] + 0.5 * dt)
        k3 = dt * deriv(y[i-1] + 0.5 * k2, t[i-1] + 0.5 * dt)
        k4 = dt * deriv(y[i-1] + k3, t[i-1] + dt)
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6  # update solution

    return y

# Initial conditions: [a, v1, b, v2]
initial_conditions = [0.0 + 0.0j, 0.0 + 0.0j]

# Time span for the simulation
t_eval = np.linspace(0, T_tot, N_tot)  # time points to evaluate the solution

# Solve the ODE
solution = rk4(coupled_oscillators_a, initial_conditions, t_eval)

# Extract the results
a = solution[:, 0]
b = solution[:, 1]

# Plotting
plt.figure(figsize=(10, 5))

# Position vs Time
plt.plot(t_eval[0:N_tot//3], np.real(a[0:N_tot//3]), label='<a>')
plt.plot(t_eval[0:N_tot//3], np.real(b[0:N_tot//3]), label='<b>', linestyle='--')
plt.title('Amplitude vs Time', fontsize=14)  # Increase title font size
plt.xlabel('Time (ps)', fontsize=12)  # Increase x-label font size
plt.ylabel('Amplitude', fontsize=12)  # Increase y-label font size
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

def plot_peaks(t, x, labels=None, max_distance=0.5, xlabel='Time', ylabel='Values', title='Selective Plot Around Peaks'):
    plt.figure(figsize=(10, 5))

    # Loop through each row in x
    for i, row in enumerate(x):
        peaks, properties = find_peaks(row)
        peak_heights = row[peaks]

        if len(peak_heights) > 0:
            largest_peaks_indices = np.argsort(peak_heights)[-2:]
            largest_peaks = peaks[largest_peaks_indices]
            valid_indices = set()

            for peak in largest_peaks:
                peak_time = t[peak]
                within_distance = np.abs(t - peak_time) <= max_distance
                valid_indices.update(np.where(within_distance)[0])

            valid_indices = sorted(valid_indices)
            t_plot = t[valid_indices]
            row_plot = row[valid_indices]

            label = labels[i] if labels is not None and i < len(labels) else None
            
            t_plot, row_plot = zip(*sorted(zip(t_plot, row_plot)))
            plt.plot(t_plot, row_plot, marker='o', linestyle='-', markersize=3, label=label)

    plt.title(title, fontsize=14)  # Increase title font size
    plt.xlabel(xlabel, fontsize=12)  # Increase x-label font size
    plt.ylabel(ylabel, fontsize=12)  # Increase y-label font size
    plt.grid()
    plt.legend()
    plt.show()

# Fourier Transform
A = np.abs(np.fft.fft(a))**2/(T_tot)**2
B = np.abs(np.fft.fft(b))**2/(T_tot)**2

# Compute the frequencies
dt = t_eval[1] - t_eval[0]  # Sampling interval
n = len(t_eval)  # Length of the signal
freqs = -np.fft.fftfreq(n, d=dt)  # frequency definition of numpy fft.fftfreq is opposite in sign with our calculation

plot_peaks(freqs * 2 * np.pi, np.array([A, B]), labels=[r'$S_{<a^{\dagger}a>}(\omega)$', r'$S_{<b^{\dagger}b>}(\omega)$'], 
           max_distance=0.2, xlabel=r'angular frequency $\omega$ (THz)', 
           ylabel=r'FFT Spectrum Intensity S$(\omega)$', title='FFT Spectrum')
