import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parameters for oscillator 1
Gamma_a = 0.01   # damping coefficient of oscillator 1
omega_a = 1.0  # spring constant of oscillator 1 (imaginary)

# Parameters for oscillator 2
Gamma_b = 0.01   # damping coefficient of oscillator 2
omega_b = 1.0  # spring constant of oscillator 2 (imaginary)

# Coupling constants (real for interaction)
g = 0.2  # coupling constant

# Parameters for the temperature and Langevin force
kB = 1.0
Temp = 1.0        # temperature
beta = 1.0 / Temp  # Inverse temperature
sigma_a = np.sqrt(2 * Gamma_a * kB * Temp)  # Strength of Langevin force
sigma_b = np.sqrt(2 * Gamma_b * kB * Temp)  # Strength of Langevin force

# Parameters for simulation time
T_tot = 1000
Step_tot = 10000

def Langevin_force(sigma, size):
    ax = np.random.normal(0, sigma, size) * 0.5
    ay = np.random.normal(0, sigma, size) * 0.5
    r = (ax**2 + ay**2) * 0.5
    theta = 2 * np.pi * np.random.uniform(0, 1, size)

    return r * np.exp(-1j * theta)

def coupled_oscillators_a(y, t, dt, fa, fb):
    a, b = y
    dadt = -Gamma_a * a - omega_a * 1j * a - g * 1j * b + fa
    dbdt = -Gamma_b * b - omega_b * 1j * b - g * 1j * a + fb
    return np.array([dadt, dbdt])

def rk4(deriv, y0, t):
    n = len(t)  # number of time steps
    y = np.zeros((n, len(y0)), dtype=np.complex128)  # initialize solution array
    y[0] = y0  # set initial conditions

    dt = np.diff(t)  # time steps
    fa = Langevin_force(sigma_a / np.sqrt(dt), n - 1)
    fb = Langevin_force(sigma_b / np.sqrt(dt), n - 1)

    for i in range(1, n):
        k1 = dt[i - 1] * deriv(y[i - 1], t[i - 1], dt[i - 1], fa[i - 1], fb[i - 1])
        k2 = dt[i - 1] * deriv(y[i - 1] + 0.5 * k1, t[i - 1] + 0.5 * dt[i - 1], dt[i - 1], fa[i - 1], fb[i - 1])
        k3 = dt[i - 1] * deriv(y[i - 1] + 0.5 * k2, t[i - 1] + 0.5 * dt[i - 1], dt[i - 1], fa[i - 1], fb[i - 1])
        k4 = dt[i - 1] * deriv(y[i - 1] + k3, t[i - 1] + dt[i - 1], dt[i - 1], fa[i - 1], fb[i - 1])
        y[i] = y[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6  # update solution

    return y

# Initial conditions
initial_conditions = [10.0 + 0.0j, 0.0 + 0.0j]

# Time span for the simulation
t_eval = np.linspace(0, T_tot, Step_tot)

# Solve the ODE
solution = rk4(coupled_oscillators_a, initial_conditions, t_eval)

# Extract the results
a = solution[:, 0]
b = solution[:, 1]

# Plotting full time-domain data
plt.figure(figsize=(12, 6))
plt.plot(t_eval, np.real(a), label='a')
plt.plot(t_eval, np.real(b), label='b', linestyle='--')
plt.title(r'Time Evolution from Coherence to Thermal Noise', fontsize=16)
plt.xlabel('Time (ps)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.legend(fontsize=12)
plt.grid()
plt.axvspan(0, 400, color='lightblue', alpha=0.5, label='Coherent Regime')
plt.axvspan(600, 1000, color='lightcoral', alpha=0.5, label='Thermal Regime')
plt.legend()
plt.text(0.05, 0.9, '(a)', transform=plt.gca().transAxes, fontsize=16)
plt.tight_layout()
plt.show()

# Plotting data from 600 to 1000 ps
plt.figure(figsize=(12, 6))
time_range = t_eval[(t_eval >= 600) & (t_eval <= 1000)]
a_range = np.real(a[(t_eval >= 600) & (t_eval <= 1000)])
b_range = np.real(b[(t_eval >= 600) & (t_eval <= 1000)])

plt.plot(time_range, a_range, label='a')
plt.plot(time_range, b_range, label='b', linestyle='--')
plt.title(r'Time Evolution in the Thermal Regime', fontsize=16)
plt.xlabel('Time (ps)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.legend(fontsize=12)
plt.grid()
plt.text(0.05, 0.9, '(b)', transform=plt.gca().transAxes, fontsize=16)
plt.tight_layout()
plt.show()

# Initial conditions
initial_conditions = [0.0 + 0.0j, 0.0 + 0.0j]

# Time span for the simulation
t_eval = np.linspace(0, T_tot*10, Step_tot*10)

# Solve the ODE
solution = rk4(coupled_oscillators_a, initial_conditions, t_eval)

# Extract the results
a = solution[:, 0]
b = solution[:, 1]

# Fourier Transform
A = np.abs(np.fft.fft(a))**2/(T_tot)**2
B = np.abs(np.fft.fft(b))**2/(T_tot)**2

# Compute the frequencies
dt = t_eval[1] - t_eval[0]  # Sampling interval
n = len(t_eval)  # Length of the signal
freqs = -np.fft.fftfreq(n, d=dt)  # frequency definition of numpy fft.fftfreq is opposite in sign with our calculation

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
    plt.xlim(0,2)
    plt.grid()
    plt.legend()
    plt.show()

plot_peaks(freqs * 2 * np.pi, np.array([A, B]), labels=[r'$S_{<a^{\dagger}a>}(\omega)$', r'$S_{<b^{\dagger}b>}(\omega)$'], 
           max_distance=1, xlabel=r'angular frequency $\omega$ (THz)', 
           ylabel=r'FFT Spectrum Intensity S$(\omega)$', title='Long-time averaged FFT Spectrum')