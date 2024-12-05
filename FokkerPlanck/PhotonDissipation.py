import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import os  # For handling directories
from fokker_planck_simulator import FokkerPlanckSimulator

# RK4 Solver
def rk4_step(t, y, dt, gamma, nu, n_th):
    # Get the k1, k2, k3, k4 slopes
    k1 = dt* system(t, y, gamma, nu, n_th)
    k2 = dt* system(t + dt/2, y + k1/2, gamma, nu, n_th)
    k3 = dt* system(t + dt/2, y + k2/2, gamma, nu, n_th)
    k4 = dt* system(t + dt, y + k3, gamma, nu, n_th)
    
    # Update the solution using the weighted average
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6


# Define the system of differential equations
def system(t, y, gamma, nu, n_th):
    a, b, c, d = y
    a_prime = gamma * (1 + n_th * (d + (b**2 + c**2)/4))
    b_prime = gamma * b/2 + gamma * n_th * b * d + nu * c
    c_prime = gamma * c/2 + gamma * n_th * c * d - nu * b
    d_prime = gamma * (d + n_th * d**2)
    
    return np.array([a_prime, b_prime, c_prime, d_prime])

# Initial condition
def init_cond(x0, y0, eps2):
    a0 = - (x0 * x0 + y0 * y0)/eps2   # Initial condition for a
    b0 = 2 * x0 /eps2  # Initial condition for b
    c0 = 2 * y0 /eps2  # Initial condition for c
    d0 = - 1/eps2  # Initial condition for d
    sol0 = np.array([a0, b0, c0, d0])
    return sol0

# Analytical solution function
def ProbDensMap(x, y, solution):
    X, Y = np.meshgrid(x, y)
    a, b, c, d = solution
    u_analytic = np.exp(a + b * X + c * Y + d * (X**2 + Y**2))
    return u_analytic

# Initialize simulation parameters
#   Physical Parameter
gamma = 1.0
nu = 3
n_th = 1
eps2 = 0.25

#   Time parameter
t_start = 0
t_end = 10
dt= 0.01

#   Map Parameter
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

x0 = 2
y0 = 2

output_dir = "fokker-planck-sim-result"

# Instantiate and run the simulation
simulator = FokkerPlanckSimulator(t_start, t_end, dt, gamma, nu, n_th, eps2, x, y, x0, y0, output_dir, ProbDensMap, rk4_step, init_cond)
simulator.run_simulation()
