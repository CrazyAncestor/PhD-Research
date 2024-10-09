import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parameters for oscillator 1
Gamma_a = 0.01   # damping coefficient of oscillator 1
wk = 1.0  # spring constant of oscillator 1 (imaginary)

# Parameters for oscillator 2
Gamma_b = 0.01   # damping coefficient of oscillator 2
wc = 1.0  # spring constant of oscillator 2 (imaginary)

# Coupling constants (real for interaction)
g = -0.2  # coupling constant from oscillator 2 to 1
gp = 0.2  # coupling constant from oscillator 1 to 2

# Parameters for the input
A = 10.0
wp = 10
tp = 2*np.pi/wp*2

# Parameters for the modulation
T_start_mod = 50
w_mod = 0.5
amp_mod = 0.

def mod_function(t,amp_mod,w_mod,start_time):
    return 1+ amp_mod*np.cos(w_mod*t)*(np.heaviside(t-start_time, 1))

def input_langevin_force(t):
    return A * np.exp(1j*wp*t) * np.exp(-(t/tp)**2/2) * np.heaviside(t, 1)

# Define the differential equations in a, b
def coupled_oscillators_a(y, t):
    a, b = y
    fa = input_langevin_force(t)
    dadt = -Gamma_a * a - wk * 1j * a + g  * b + fa
    dbdt = -Gamma_b * b - wc * 1j * mod_function(t,amp_mod,w_mod,T_start_mod) * b + gp  * a
    return np.array([dadt, dbdt])

def rk4(deriv, y0, t):
    """
    Runge-Kutta 4th Order Method to solve ODEs.

    Parameters:
    - deriv: function that returns dy/dt
    - y0: initial conditions (list or array)
    - t: time array

    Returns:
    - y: array of solutions over time
    """
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
t_span = (0, 1000)  # from t=0 to t=20 seconds
t_eval = np.linspace(*t_span, 10000)  # time points to evaluate the solution

# Solve the ODE
solution = rk4(coupled_oscillators_a, initial_conditions, t_eval)

# Extract the results
a = solution[:, 0]
b = solution[:, 1]

# Plotting
plt.figure(figsize=(10, 5))

# Position vs Time
plt.plot(t_eval[0:3000], np.real(a[0:3000]), label='<a>')
plt.plot(t_eval[0:3000], np.real(b[0:3000]), label='<b>', linestyle='--')
plt.title('Amplitude vs Time')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

def plot_peaks(t, x, labels=None, max_distance=0.5, xlabel='Time', ylabel='Values', title='Selective Plot Around Peaks'):
    """
    Plots the largest 2 peaks of each row in x against t, focusing on the significant peaks of each row.

    Parameters:
    - t: Array-like, the x-axis values (time).
    - x: 2D Array-like, where each row represents different data to plot against t.
    - labels: List of strings, labels for each row in x (default is None).
    - max_distance: Float, maximum distance from peaks to consider for plotting.
    - xlabel: String, label for the x-axis.
    - ylabel: String, label for the y-axis.
    - title: String, title of the plot.
    """
    plt.figure(figsize=(10, 5))

    # Loop through each row in x
    for i, row in enumerate(x):
        # Filter out data where t < 0
        valid_mask = t >= 0
        t_filtered = t[valid_mask]
        row_filtered = row[valid_mask]

        # Find peaks in filtered row
        peaks, properties = find_peaks(row_filtered)

        # Get the heights of the peaks
        peak_heights = row_filtered[peaks]

        # Get indices of the largest 2 peaks
        if len(peak_heights) > 0:
            largest_peaks_indices = np.argsort(peak_heights)[-2:]  # Indices of the largest 2 peaks
            largest_peaks = peaks[largest_peaks_indices]  # Corresponding peak indices

            # Create a mask for valid indices based on the max distance from largest peaks
            valid_indices = set()

            for peak in largest_peaks:
                peak_time = t_filtered[peak]
                # Find indices of t within max_distance from the peak
                within_distance = np.abs(t_filtered - peak_time) <= max_distance
                valid_indices.update(np.where(within_distance)[0])

            # Convert set to sorted list
            valid_indices = sorted(valid_indices)

            # Create subsets of t and row for plotting
            t_plot = t_filtered[valid_indices]
            row_plot = row_filtered[valid_indices]

            # Plot each row with a label if provided
            label = labels[i] if labels is not None and i < len(labels) else None
            
            # Assuming t_plot and row_plot are your arrays
            t_plot, row_plot = zip(*sorted(zip(t_plot, row_plot)))

            # Now plot the sorted arrays
            plt.plot(t_plot, row_plot, marker='o', linestyle='-', markersize=3, label=label)

    # Finalize the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()  # Show legend if labels are provided
    plt.show()

# Fourier Transform
def clean_time_series(t, x, time_range):
    """
    Removes elements from t and x that fall within a specified time range.

    Parameters:
    - t: Array-like, the x-axis values (time).
    - x: Array-like, the y-axis values (data).
    - time_range: Tuple (start, end), the time range to remove from t and x.

    Returns:
    - t_clean: Array-like, cleaned time values with specified range removed.
    - x_clean: Array-like, cleaned data values corresponding to t_clean.
    """
    # Create a mask to filter out elements within the time range
    mask = (t > time_range[0]) & (t < time_range[1])

    # Apply the mask to t and x
    t_clean = t[mask]
    x_clean = x[mask]

    return t_clean, x_clean

t_clean, a_clean = clean_time_series(t_eval, a, [T_start_mod,1000])
t_clean, b_clean = clean_time_series(t_eval, b, [T_start_mod,1000])

A = np.abs(np.fft.fft(a_clean))**2
B = np.abs(np.fft.fft(b_clean))**2

# Compute the frequencies
dt = t_clean[1] - t_clean[0]  # Sampling interval
n = len(t_clean)  # Length of the signal
freqs = -(np.fft.fftfreq(n, d=dt))

plot_peaks(freqs, np.array([A,B]), labels=[r'$<a^{\dagger}a>$',r'$<b^{\dagger}b>$'], max_distance=0.2, xlabel='freq (THz)', ylabel='FFT Spectrum', title='FFT Spectrum')