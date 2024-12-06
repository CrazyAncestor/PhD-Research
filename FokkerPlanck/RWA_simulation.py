import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import os  # For handling directories
from fokker_planck_simulator import FokkerPlanckSimulator

# RK4 Solver
def rk4_step(t, y, dt, phys_parameter):
    # Get the k1, k2, k3, k4 slopes
    k1 = dt* system(t, y, phys_parameter)
    k2 = dt* system(t + dt/2, y + k1/2, phys_parameter)
    k3 = dt* system(t + dt/2, y + k2/2, phys_parameter)
    k4 = dt* system(t + dt, y + k3, phys_parameter)
    
    # Update the solution using the weighted average
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6


# Define the system of differential equations
def system(t, y, phys_parameter):
    gamma, nu, n_th =  phys_parameter
    a, b, c, d, e, f, h, k, l, o, p, q, r, s, w = y
    a_prime = (n_th*gamma*b**2)/4 + (n_th*gamma*c**2)/4 + (eta*m_th*d**2)/4 + (eta*m_th*e**2)/4 + eta + gamma + (f*n_th*gamma)/2 + (h*n_th*gamma)/2 + (eta*k*m_th)/2 + (eta*l*m_th)/2

    b_prime = (b*gamma)/2 + e*g + c*nu + (c*gamma*n_th*o)/2 + (d*eta*m_th*q)/2 + (e*eta*m_th*s)/2 + b*f*gamma*n_th
    c_prime = (c*gamma)/2 - d*g - b*nu + c*gamma*h*n_th + (b*gamma*n_th*o)/2 + (d*eta*m_th*r)/2 + (e*eta*m_th*w)/2
    d_prime = (d*eta)/2 + c*g + e*nu + d*eta*k*m_th + (e*eta*m_th*p)/2 + (b*gamma*n_th*q)/2 + (c*gamma*n_th*r)/2
    e_prime = (e*eta)/2 - b*g - d*nu + e*eta*l*m_th + (d*eta*m_th*p)/2 + (b*gamma*n_th*s)/2 + (c*gamma*n_th*w)/2

    f_prime = f*gamma + g*s + nu*o + (gamma*n_th*(8*f**2 + 2*o**2))/8 + (eta*m_th*(2*q**2 + 2*s**2))/8
    h_prime = gamma*h - g*r - nu*o + (gamma*n_th*(8*h**2 + 2*o**2))/8 + (eta*m_th*(2*r**2 + 2*w**2))/8
    k_prime = eta*k + g*r + nu*p + (eta*m_th*(8*k**2 + 2*p**2))/8 + (gamma*n_th*(2*q**2 + 2*r**2))/8
    l_prime = eta*l - g*s - nu*p + (eta*m_th*(8*l**2 + 2*p**2))/8 + (gamma*n_th*(2*s**2 + 2*w**2))/8

    o_prime = gamma*o - 2*f*nu + 2*h*nu - g*q + g*w + (eta*m_th*(2*q*r + 2*s*w))/4 + (gamma*n_th*(4*f*o + 4*h*o))/4
    p_prime = eta*p - g*q - 2*k*nu + 2*l*nu + g*w + (eta*m_th*(4*k*p + 4*l*p))/4 + (gamma*n_th*(2*q*s + 2*r*w))/4
    q_prime = g*o + (eta*q)/2 + g*p + (gamma*q)/2 + nu*r + nu*s + (gamma*n_th*(4*f*q + 2*o*r))/4 + (eta*m_th*(4*k*q + 2*p*s))/4
    r_prime = 2*g*h - 2*g*k + (eta*r)/2 + (gamma*r)/2 - nu*q + nu*w + (gamma*n_th*(4*h*r + 2*o*q))/4 + (eta*m_th*(4*k*r + 2*p*w))/4
    s_prime = 2*g*l - 2*f*g + (eta*s)/2 + (gamma*s)/2 - nu*q + nu*w + (eta*m_th*(4*l*s + 2*p*q))/4 + (gamma*n_th*(4*f*s + 2*o*w))/4
    w_prime = (eta*w)/2 - g*p - g*o + (gamma*w)/2 - nu*r - nu*s + (gamma*n_th*(4*h*w + 2*o*s))/4 + (eta*m_th*(2*p*r + 4*l*w))/4
    
    
    return np.array([a_prime, b_prime, c_prime, d_prime,e_prime, f_prime, h_prime, k_prime,l_prime,o_prime, p_prime, q_prime, r_prime,s_prime,w_prime])

# Analytical solution function
def ProbDensMap(x, y, solution):
    X, Y = np.meshgrid(x, y)
    a, b, c, d = solution
    ProbDens = np.exp(a + b * X + c * Y + d * (X**2 + Y**2))
    return ProbDens

# Initialize simulation parameters
#   Physical Parameter
gamma = 0.1
eta = 0.1
nu = 3
omega = 3
g = 1/3
n_th = 1
m_th = 1

eps2 = 0.25
x0 = 2
y0 = 2

phys_parameter = gamma, nu, n_th

# Initial condition
a0 = - (x0 * x0 + y0 * y0)/eps2

b0 = 2 * x0 /eps2
c0 = 2 * y0 /eps2
d0 = 0 * x0 /eps2
e0 = 0 * y0 /eps2

f0 = - 1/eps2
h0 = - 1/eps2
k0 = - 1/eps2
l0 = - 1/eps2

o0 = 0
p0 = 0
q0 = 0
r0 = 0
s0 = 0
w0 = 0


init_cond = np.array([a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0])

#   Time parameter
t_start = 0
t_end = 10
dt= 0.01

#   Map Parameter
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

output_dir = "fokker-planck-sim-result"

# Instantiate and run the simulation
simulator = FokkerPlanckSimulator(t_start, t_end, dt, x, y, phys_parameter, init_cond, output_dir, ProbDensMap, rk4_step)
simulator.run_simulation(pure_parameter = True)
