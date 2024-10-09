import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parameters for oscillator 1
Gamma_a = 0.03   # damping coefficient of oscillator 1
omega_a = 1.0  # spring constant of oscillator 1 (imaginary)

# Parameters for oscillator 2
Gamma_b = 0.02   # damping coefficient of oscillator 2
omega_b = 1.0  # spring constant of oscillator 2 (imaginary)

# Coupling constants (real for interaction)
g = 0.  # coupling constant

# Parameters for the temperature and Langevin force
kB = 1.0
Temp = 1.0        # temperature
beta = 1.0 / (Temp)  # Inverse temperature
sigma_a = np.sqrt(2 * Gamma_a * kB * Temp)  # Strength of Langevin force
sigma_b = np.sqrt(2 * Gamma_b * kB * Temp)  # Strength of Langevin force

# Parameters for simulation time
Time_tot = 10000
Step_tot = 100000

# Modulation frequencies to test
omega_mod_values = [0.5, 1.0, 2.0, 5.0, 10.0]
T_long = 100
amp_mod = 0.5

def mod_function(t, amp_mod, omega_mod):
    #return 1 + amp_mod * np.cos(omega_mod * t)
    return 1 + amp_mod * np.cos(omega_mod * t)

def Langevin_force(t, sigma):
    ax = np.random.normal(-sigma, sigma)*0.5
    ay = np.random.normal(-sigma, sigma)*0.5
    r = (ax**2+ay**2)*0.5
    theta = np.pi*2* np.random.uniform(low=0.0, high=1.0, size=None)
    return r* np.exp(-1j*theta)

def coupled_oscillators_a(y, t, dt, omega_mod):
    a, b = y
    fa = Langevin_force(t, sigma_a / np.sqrt(dt))
    fb = Langevin_force(t, sigma_b / np.sqrt(dt))
    dadt = -Gamma_a * a - omega_a * 1j * a - g * 1j * b + fa
    dbdt = -Gamma_b * b - omega_b * mod_function(t, amp_mod, omega_mod) * 1j * b - g * 1j * a + fb
    return np.array([dadt, dbdt])

def rk4(deriv, y0, t, omega_mod):
    n = len(t)  # number of time steps
    y = np.zeros((n, len(y0))) * (0.0 + 0.0j)  # initialize solution array
    y[0] = y0  # set initial conditions

    for i in range(1, n):
        dt = t[i] - t[i-1]  # time step
        k1 = dt * deriv(y[i-1], t[i-1], dt, omega_mod)
        k2 = dt * deriv(y[i-1] + 0.5 * k1, t[i-1] + 0.5 * dt, dt, omega_mod)
        k3 = dt * deriv(y[i-1] + 0.5 * k2, t[i-1] + 0.5 * dt, dt, omega_mod)
        k4 = dt * deriv(y[i-1] + k3, t[i-1] + dt, dt, omega_mod)
        y[i] = y[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6  # update solution

    return y

# Initial conditions
initial_conditions = [0.0 + 0.0j, 0.0 + 0.0j]

# Time span for the simulation
t_eval = np.linspace(0, Time_tot, Step_tot) 

# Loop over modulation frequencies
for omega_mod in omega_mod_values:
    # Solve the ODE
    solution = rk4(coupled_oscillators_a, initial_conditions, t_eval, omega_mod)

    # Extract the results
    a = solution[:, 0]
    b = solution[:, 1]

    # Fourier Transform
    A = np.abs(np.fft.fft(a))**2
    B = np.abs(np.fft.fft(b))**2

    # Compute the frequencies
    dt = t_eval[1] - t_eval[0]  # Sampling interval
    n = len(t_eval)  # Length of the signal
    freqs = np.fft.fftfreq(n, d=dt)

    # Find peaks in the FFT spectrum
    peaks_a, _ = find_peaks(A)
    peaks_b, _ = find_peaks(B)

    # Get the five maximum peaks
    top_n = 5
    peak_indices_a = peaks_a[np.argsort(A[peaks_a])[-top_n:]]
    peak_indices_b = peaks_b[np.argsort(B[peaks_b])[-top_n:]]

    # Define a range around the peaks to plot
    peak_range = 0.1  # Adjust this value as needed

    # Collect valid indices around each peak
    filtered_indices_a = np.zeros_like(freqs, dtype=bool)
    filtered_indices_b = np.zeros_like(freqs, dtype=bool)

    for peak in peak_indices_a:
        filtered_indices_a |= (np.abs(freqs - freqs[peak]) <= peak_range)

    for peak in peak_indices_b:
        filtered_indices_b |= (np.abs(freqs - freqs[peak]) <= peak_range)

    # Prepare data for plotting only around the peaks
    freqs_a_filtered = freqs[filtered_indices_a]
    A_filtered = A[filtered_indices_a]

    freqs_b_filtered = freqs[filtered_indices_b]
    B_filtered = B[filtered_indices_b]

    # Convert frequencies to angular frequencies
    angular_freqs_a_filtered = -2 * np.pi * freqs_a_filtered
    angular_freqs_b_filtered = -2 * np.pi * freqs_b_filtered

    # Plotting both time-domain data and frequency data together
    plt.figure(figsize=(12, 6))

    # Time-domain data
    plt.subplot(2, 1, 1)
    plt.plot(t_eval, np.real(a), label='a')
    plt.plot(t_eval, np.real(b), label='b', linestyle='--')
    plt.title(r'Time Domain Data ($\omega_{mod}$'+f' = {omega_mod}'+r'$\omega_{0}$)')
    plt.xlabel('Time (ps)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    # Frequency data (FFT Spectrum)
    plt.subplot(2, 1, 2)
    plt.plot(angular_freqs_a_filtered, A_filtered, label=r'$a^{\dagger}a$')
    plt.plot(angular_freqs_b_filtered, B_filtered, label=r'$b^{\dagger}b$', linestyle='--')
    plt.title(r'FFT Spectrum ($\omega_{mod}$'+f' = {omega_mod}'+r'$\omega_{0})$')
    plt.xlabel('Angular Frequency (THz)')
    plt.ylabel('FFT Spectrum')
    plt.legend()
    plt.grid()

    plt.tight_layout()

    # Save the figure
    plt.savefig(f'oscillator_plots_omega_mod_{omega_mod}.pdf')
    plt.close()  # Close the figure after saving
