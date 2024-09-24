import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the parameters
g = 10
l = 1
alpha0 = 2
a0 = (2*g/l/(0.5+alpha0/6))**0.5

# Define the differential equation
def dtheta_dt(t, theta):
    return a0 * np.sin(theta)

def d2theta_dt2(t, theta):
    return a0 * np.cos(theta) * dtheta_dt(t, theta)

def ax(t, theta):
    return -l*np.sin(theta)*dtheta_dt(t, theta)**2 + l*np.cos(theta)*d2theta_dt2(t, theta)

def ay(t, theta):
    return -l*np.cos(theta)*dtheta_dt(t, theta)**2 - l*np.sin(theta)*d2theta_dt2(t, theta)

# Time span for the solution
T0 = (g/l)**0.5
t_span = (0, 0.5*T0)  # From t=0 to t=10
t_eval = np.linspace(t_span[0], t_span[1], 100)  # Points to evaluate

# Initial condition
theta0 = 0.01  # Initial angle

# Solve the differential equation
solution = solve_ivp(dtheta_dt, t_span, [theta0], t_eval=t_eval)

# Extract time and theta values from the solution
t_values = solution.t
theta_values = solution.y[0]
dtheta_dt_values = []
d2theta_dt2_values = []
axs = []
ays = []
for i in range(len(t_values)):
    dtheta_dt_values.append(dtheta_dt(t_values[i], theta_values[i]))
    d2theta_dt2_values.append(d2theta_dt2(t_values[i], theta_values[i]))
    axs.append(ax(t_values[i], theta_values[i])/g/np.cos(theta_values[i]*2)/np.sin(theta_values[i]))
    ays.append(ay(t_values[i], theta_values[i])/g+1)

# Plot the results
plt.figure
plt.plot(t_values, theta_values/np.pi)
plt.title("Solution of dθ/dt = a sin(θ)")
plt.xlabel("Time (t)")
plt.ylabel("Theta (θ)")
plt.grid()
#plt.show()

"""plt.figure
plt.plot(t_values, dtheta_dt_values)
plt.title("Solution of dθ/dt = a sin(θ)")
plt.xlabel("Time (t)")
plt.ylabel("Theta (θ)")
plt.grid()
plt.show()

plt.figure
plt.plot(t_values, d2theta_dt2_values)
plt.title("Solution of dθ/dt = a sin(θ)")
plt.xlabel("Time (t)")
plt.ylabel("Theta (θ)")
plt.grid()
plt.show()
"""
plt.figure
plt.plot(t_values, axs)
plt.title("Solution of ax")
plt.xlabel("Time (t)")
plt.ylabel("ax (m/s^2)")
plt.grid()
#plt.show()

plt.figure
plt.plot(t_values, ays)
plt.title("Solution of ay")
plt.xlabel("Time (t)")
plt.ylabel("ay (m/s^2)")
plt.grid()
plt.show()