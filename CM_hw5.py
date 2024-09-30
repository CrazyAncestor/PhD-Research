import numpy as np
import matplotlib.pyplot as plt
A= 0.9
beta = 0.1
omega0 =20
tau = 2*np.pi/omega0
w = (omega0**2 - beta**2)**0.5

def G(t,t0):
    if t>t0:
        return np.exp(-(t-t0))*np.sin(w*(t-t0))
    else:
        return 0

def all_G(t):
    sum = 0
    for n in range(10):
        sum += G(t,n/2*tau) * A**n
    return sum

t = np.linspace(0,10,1000)
x = []
for i in range(len(t)):
    #x.append(G(t[i],2.5))
    x.append(all_G(t[i]))

print(beta * tau)
print(A**2*(1-A))
plt.plot(t,x)
plt.show()