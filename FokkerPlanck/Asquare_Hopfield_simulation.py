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
    Dia = g**2/omega
    a_prime = (n_th*gamma*b**2)/4 - Dia*b*c - (g*b*e)/2 + (n_th*gamma*c**2)/4 - (g*c*d)/2 + (eta*m_th*d**2)/4 + (eta*m_th*e**2)/4 + eta + gamma - Dia*o - (g*r)/2 - (g*s)/2 + (f*n_th*gamma)/2 + (h*n_th*gamma)/2 + (eta*k*m_th)/2 + (eta*l*m_th)/2
 
    b_prime = 4*Dia*c + (b*gamma)/2 + 2*e*g + c*nu - 2*Dia*c*f - Dia*b*o - e*f*g - (d*g*o)/2 - (c*g*q)/2 - (b*g*s)/2 + (c*gamma*n_th*o)/2 + (d*eta*m_th*q)/2 + (e*eta*m_th*s)/2 + b*f*gamma*n_th
    c_prime = (c*gamma)/2 - b*nu - 2*Dia*b*h - Dia*c*o - d*g*h - (e*g*o)/2 - (c*g*r)/2 - (b*g*w)/2 + c*gamma*h*n_th + (b*gamma*n_th*o)/2 + (d*eta*m_th*r)/2 + (e*eta*m_th*w)/2
    d_prime = 2*Dia*e + (d*eta)/2 + 2*c*g + e*nu - Dia*b*r - Dia*c*q - c*g*k - (b*g*p)/2 - (d*g*r)/2 - (e*g*q)/2 + d*eta*k*m_th + (e*eta*m_th*p)/2 + (b*gamma*n_th*q)/2 + (c*gamma*n_th*r)/2
    e_prime = (e*eta)/2 - 2*Dia*d - d*nu - Dia*c*s - Dia*b*w - b*g*l - (c*g*p)/2 - (e*g*s)/2 - (d*g*w)/2 + e*eta*l*m_th + (d*eta*m_th*p)/2 + (b*gamma*n_th*s)/2 + (c*gamma*n_th*w)/2

    f_prime = f*gamma + 2*g*s + o*(4*Dia + nu) - 2*Dia*f*o - f*g*s - (g*o*q)/2 + (gamma*n_th*(8*f**2 + 2*o**2))/8 + (eta*m_th*(2*q**2 + 2*s**2))/8
    h_prime = gamma*h - nu*o - 2*Dia*h*o - g*h*r - (g*o*w)/2 + (gamma*n_th*(8*h**2 + 2*o**2))/8 + (eta*m_th*(2*r**2 + 2*w**2))/8
    k_prime = eta*k + 2*g*r + p*(2*Dia + nu) - Dia*q*r - g*k*r - (g*p*q)/2 + (eta*m_th*(8*k**2 + 2*p**2))/8 + (gamma*n_th*(2*q**2 + 2*r**2))/8
    l_prime = eta*l - p*(2*Dia + nu) - Dia*s*w - g*l*s - (g*p*w)/2 + (eta*m_th*(8*l**2 + 2*p**2))/8 + (gamma*n_th*(2*s**2 + 2*w**2))/8

    o_prime = gamma*o - 2*f*nu + 2*h*(4*Dia + nu) + 2*g*w - (g*(2*h*q + o*r))/2 - (g*(2*f*w + o*s))/2 - Dia*(o**2 + 4*f*h) + (eta*m_th*(2*q*r + 2*s*w))/4 + (gamma*n_th*(4*f*o + 4*h*o))/4
    p_prime = eta*p - 2*k*(2*Dia + nu) + 2*l*(2*Dia + nu) + 2*g*w - Dia*(r*s + q*w) - (g*(2*l*q + p*s))/2 - (g*(2*k*w + p*r))/2 + (eta*m_th*(4*k*p + 4*l*p))/4 + (gamma*n_th*(2*q*s + 2*r*w))/4
    q_prime = 2*g*o + (eta*q)/2 + 2*g*p + (gamma*q)/2 + r*(4*Dia + nu) + s*(2*Dia + nu) - Dia*(2*f*r + o*q) - (g*(2*f*p + q*s))/2 - (g*(2*k*o + q*r))/2 + (gamma*n_th*(4*f*q + 2*o*r))/4 + (eta*m_th*(4*k*q + 2*p*s))/4
    r_prime = 4*g*h - (g*(r**2 + 4*h*k))/2 + (eta*r)/2 + (gamma*r)/2 - nu*q + w*(2*Dia + nu) - Dia*(2*h*q + o*r) - (g*(o*p + q*w))/2 + (gamma*n_th*(2*h*r + o*q))/2 + (eta*m_th*(2*k*r + p*w))/2
    s_prime = 4*g*l - (g*(s**2 + 4*f*l))/2 + (eta*s)/2 + (gamma*s)/2 - q*(2*Dia + nu) + w*(4*Dia + nu) - Dia*(2*f*w + o*s) - (g*(o*p + q*w))/2 + (eta*m_th*(2*l*s + p*q))/2 + (gamma*n_th*(2*f*s + o*w))/2
    w_prime = (eta*w)/2 + (gamma*w)/2 - nu*s - r*(2*Dia + nu) - Dia*(2*h*s + o*w) - (g*(2*h*p + r*w))/2 - (g*(2*l*o + s*w))/2 + (gamma*n_th*(4*h*w + 2*o*s))/4 + (eta*m_th*(2*p*r + 4*l*w))/4
     
    return np.array([a_prime, b_prime, c_prime, d_prime,e_prime, f_prime, h_prime, k_prime,l_prime,o_prime, p_prime, q_prime, r_prime,s_prime,w_prime])

# Analytical solution function
def ProbDensMap(x, y, solution):
    X, Y = np.meshgrid(x, y)
    a, b, c, d, e, f, h, k, l, o, p, q, r, s, w = solution
    A = d + q * X + r * Y
    B = e + s * X + w * Y
    H = l - p**2/4/k
    U = B - A*p/2/k
    Prob0 = np.exp(f*X**2 + o*X*Y + b*X + h*Y**2 + c*Y + a)
    Prob_var = np.pi/ (np.abs(k)*np.abs(l-p**2/4/k))**0.5 *np.exp(-U**2/4/H)*np.exp(-A**2/4/k)
    ProbDens = Prob0 * Prob_var
    return ProbDens

# Initialize simulation parameters
#   Physical Parameter
gamma = 0.5
eta = 0.1
nu = 2 * np.pi
omega = 2 * np.pi
g =  2 * np.pi * 0.1
n_th = 1
m_th = 1

eps2 = 0.5
x0 = 2
y0 = 2

phys_parameter = gamma, nu, n_th

# Initial condition
Dim = 4
a0 = - (x0 * x0 + y0 * y0)/eps2 - Dim * np.log( np.sqrt(np.pi * eps2))

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
dt= 0.001

#   Map Parameter
x = np.linspace(-5, 5, 150)
y = np.linspace(-5, 5, 150)

output_dir = "fokker-planck-sim-result"

# Instantiate and run the simulation
simulator = FokkerPlanckSimulator(t_start, t_end, dt, x, y, phys_parameter, init_cond, output_dir, ProbDensMap, rk4_step)
simulator.run_simulation(pure_parameter = False)
