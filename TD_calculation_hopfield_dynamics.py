import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

# Parameters for oscillator 1
Gamma_a = 0.01   # damping coefficient of oscillator 1
wk = 1.0j  # spring constant of oscillator 1 (imaginary)

# Parameters for oscillator 2
Gamma_b = 0.01   # damping coefficient of oscillator 2
wc = 1.0j  # spring constant of oscillator 2 (imaginary)

# Coupling constants (real for interaction)
g = 0.2  # coupling constant from oscillator 2 to 1
gp = -0.2  # coupling constant from oscillator 1 to 2

# Parameters for the input
A = 10.0
wp = 10
tp = 2*np.pi/wp*2
print(tp)

def input_langevin_force(t):
    return A * np.exp(1j*wp*t) * np.exp(-(t/tp)**2/2) * np.heaviside(t, 1)

# Define the differential equations in a, a^+
def coupled_oscillators_a(y, t):
    a, b = y
    fa = input_langevin_force(t)
    dadt = -Gamma_a * a + wk * a + g * (1+ 0.5*np.cos(0.*t)*(np.heaviside(t-50, 1))) * b + fa
    dbdt = -Gamma_b * b + wc * b + gp * (1+ 0.5*np.cos(0.*t)*(np.heaviside(t-50, 1)))* a
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
#solution = solve_ivp(coupled_oscillators_a, t_span, initial_conditions, t_eval=t_eval)
solution = rk4(coupled_oscillators_a, initial_conditions, t_eval)

# Extract the results
a = solution[:, 0]
b = solution[:, 1]

# Plotting
plt.figure(figsize=(12, 10))

# Position vs Time
#plt.subplot(2, 2, 1)
plt.plot(t_eval, np.real(a), label='Oscillator 1 (a)')
plt.plot(t_eval, np.real(b), label='Oscillator 2 (a)', linestyle='--')
plt.title('Position vs Time in a Basis')
plt.xlabel('Time (s)')
plt.ylabel('Position (a)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()



# Plotting
plt.figure(figsize=(12, 10))

A = np.fft.fft(a)
B = np.fft.fft(b)

# Compute the frequencies
dt = t_eval[1] - t_eval[0]  # Sampling interval
n = len(a)  # Length of the signal
freqs = np.fft.fftfreq(n, d=dt)

# Position vs Time
#plt.subplot(2, 2, 1)
#plt.plot(freqs, A, label='Oscillator 1 (a)')
f = freqs[0:500]
bf =  B[0:500]

plt.plot(f, bf, label='Oscillator 2 (B)')

plt.title('Position vs Time in a Basis')
plt.xlabel('Time (s)')
plt.ylabel('Position (a)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
