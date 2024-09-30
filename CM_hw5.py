import numpy as np
import matplotlib.pyplot as plt

global A_crit

def displacement(A,make_plot=False):
    beta = 1
    omega0 =20
    w = (omega0**2 - beta**2)**0.5
    tau = 2*np.pi/w


    def G(t,t0):
        if t>t0:
            return np.exp(-beta*(t-t0))*np.sin(w*(t-t0))
        else:
            return 0

    def all_G(t):
        sum = 0
        for n in range(100):
            sum += G(t,n/2*tau) * A**n
        return sum

    t = np.linspace(0,10,1000)
    x = []
    y = []

    global A_crit
    A_crit = np.exp(-beta*tau/2)
    for i in range(len(t)):
        x.append(all_G(t[i]))
        y.append( np.exp(-beta*(t[i])))
    
    if make_plot:
        plt.plot(t,x,label = 'x(t) for A='+str(A))
        plt.plot(t,y,label = r'y(t)=$e^{-\beta t}$')
        plt.xlabel('time')
        plt.ylabel('displacement')
        plt.legend()
        plt.show()

    return t,x,y


t,x1,y = displacement(0.8,True)
t,x2,y = displacement(0.85,True)
t,x3,y = displacement(0.95,True)
