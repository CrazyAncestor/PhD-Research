import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Assuming DGF and w_Bmod are defined globally or passed as arguments
DGF = None  # Placeholder, set this according to your program
w_Bmod = None  # Placeholder, set this according to your program

def fft_analysis(yt):
    yf = np.fft.fft(yt)
    N = len(yf)
    a = yf[:N//2]
    b = yf[N//2:]
    yf = np.concatenate((b, a))
    return yf

def magnetic_time_domain(t, Bi, DB, wp, Dt):
    #
    if t>0:
        Bt = Bi + DB
    else:
        Bt = Bi
    Bt = DB * np.cos(wp * t) * np.exp(-t**2 / (2 * Dt**2)) + Bi
    return Bt

def Hopfield_Matrix(w, wk, wc, g, Ga, Gb, Gc):
    D = g**2 / wc
    G = np.array([[wk - 1j * (Ga + Gc) - w + 2 * D, 1j * g, 2 * D, -1j * g],
                  [-1j * g, wc - 1j * Gb - w, -1j * g, 0],
                  [-2 * D, -1j * g, -wk - 1j * (Ga + Gc) - w - 2 * D, 1j * g],
                  [-1j * g, 0, -1j * g, -wc - 1j * Gb - w]])
    return G

def wc_fft(w_dim, wc, Bi, DB, wp, Dt):
    N = w_dim
    dw = wp * 6 / N
    w = dw * np.arange(-N/2, N/2)
    dt = 2 * np.pi / (dw * N)
    t = dt * np.arange(-N/2, N/2)
    
    wct = []
    for i in range(len(t)):
        Bt = magnetic_time_domain(t[i], Bi, DB, wp, Dt)
        wct.append(wc * (Bt / Bi) - wc)
    
    wct = np.array(wct)
    DWCF = fft_analysis(wct) #* dt
    return w, DWCF

def g_fft(w_dim, wc, Bi, DB, wp, Dt, g):
    N = w_dim
    dw = wp * 6 / N
    w = dw * np.arange(-N/2, N/2)
    dt = 2 * np.pi / (dw * N)
    t = dt * np.arange(-N/2, N/2)
    
    gt = []
    for i in range(len(t)):
        Bt = magnetic_time_domain(t[i], Bi, DB, wp, Dt)
        gt.append(g * np.sqrt(np.abs(Bt / Bi)) - g)
    
    gt = np.array(gt)
    DGF = fft_analysis(gt)# * dt
    return w, DGF

# Physical parameters
# Angular frequency unit
THz = 1e12 * 2 * np.pi
GHz = 1e9 * 2 * np.pi

# Coupling strength and decaying rates
Gamma_A = 5 * GHz
Gamma_B = 2 * GHz
Gamma_C = 1 * GHz
wc = 0.4 * THz
wk = 0.4 * THz
g = 0.2 * wc

# Modulating external magnetic field
e = 1.6e-19
me = 9.11e-31
me_GaAs = 0.067 * me
Bi = me_GaAs * wc / e
DB = Bi * 1e-5
wp = wc * np.array([2,1,0.5])
Dt = 10. / wp
w_Bmod_dim = 1000

# Perform FFT analyses
w_Bmod, DWCF = wc_fft(w_Bmod_dim, wc, Bi, DB, wp[0], Dt[0])
w_Bmod, DGF = g_fft(w_Bmod_dim, wc, Bi, DB, wp[0], Dt[0], g)

# Plotting
plt.figure()
plt.plot(w_Bmod / 1e12, np.real(DWCF))
plt.xlabel('Angular frequency (THz)')
plt.ylabel('Coupling strength in frequency domain')
plt.show()

def g_mod_freq(wi, wj):
    global DGF, w_Bmod
    w = wi - wj
    interp_func = interp1d(w_Bmod, DGF, bounds_error=False, fill_value=0)
    dg = interp_func(w)
    return dg

def wc_mod_freq(wi, wj):
    global DWCF, w_Bmod
    w = wi - wj
    interp_func = interp1d(w_Bmod, DWCF, bounds_error=False, fill_value=0)
    dwc = interp_func(w)
    return dwc

def init_G(N, w, wk, wc, g, Gamma_A, Gamma_B, Gamma_C):
    G = np.zeros((4 * N, 4 * N), dtype=complex)  # Initialize a complex zero matrix
    for i in range(N):
        Gu = Hopfield_Matrix(w[i], wk, wc, g, Gamma_A, Gamma_B, Gamma_C)
        m = 4 * i
        G[m:m + 4, m:m + 4] = Gu
    return G

def Modulated_Hopfield_Matrix(wi, wj, dw):
    dg = g_mod_freq(wi, wj)
    dwc = wc_mod_freq(wi, wj)
    
    DG = np.array([[0, 1j * dg, 0, -1j * dg],
                   [-1j * dg, dwc, -1j * dg, 0],
                   [0, -1j * dg, 0, 1j * dg],
                   [-1j * dg, 0, -1j * dg, -dwc]], dtype=complex)
    return DG * dw

def Mod_G(N, w, dw):
    G_mod_middle = []
    for j in range(N):
        DG = Modulated_Hopfield_Matrix(w[N // 2], w[j], dw)
        G_mod_middle.append(DG)
    
    G_mod_middle = np.array(G_mod_middle)  # Convert list of matrices to a 3D NumPy array
    G_mod = np.zeros((4 * N, 4 * N), dtype=complex)

    for i in range(N):
        for j in range(N):
            m = 4 * i
            n = 4 * j
            s = (j - i +  (N // 2))
            if 0 < s <  N:
                DGU = G_mod_middle[s, :, :]  # Select the appropriate slice
                G_mod[m:m + 4, n:n + 4] = DGU
                
    return G_mod

def final_M(N, H, au, bu):
    M = np.zeros((3 * N, 3 * N), dtype=complex)

    for i in range(N):
        for j in range(N):
            m = 3 * i
            n = 3 * j
            mu = np.matmul(au, H[4 * i:4 * (i + 1), 4 * j:4 * (j + 1)])
            mu  = np.matmul(mu,bu)
            M[m:m + 3, n:n + 3] = mu
            
    return M

def Time_Mod_Matrix(w, wk, wc, g, Gamma_A, Gamma_B, Gamma_C, turn_on_ext_B):
    # Calculate parameters
    ga = np.sqrt(Gamma_A / np.pi)
    gb = np.sqrt(Gamma_B / np.pi)
    gc = np.sqrt(Gamma_C / np.pi)

    dw = w[1] - w[0]
    N = len(w)

    au = np.array([[ga, 0, ga, 0],
                   [0, gb, 0, gb],
                   [gc, 0, gc, 0]])
    
    bu = np.array([[2j * Gamma_A / ga, 0, 2j * Gamma_C / gc],
                   [0, 2j * Gamma_B / gb, 0],
                   [2j * Gamma_A / ga, 0, 2j * Gamma_C / gc],
                   [0, 2j * Gamma_B / gb, 0]])

    G = init_G(N, w, wk, wc, g, Gamma_A, Gamma_B, Gamma_C)

    if turn_on_ext_B:
        G_mod = Mod_G(N, w, dw)

        G += G_mod  # Element-wise addition
    

    H = np.linalg.inv(G)
    
    M = final_M(N, H, au, bu)
    #np.savetxt('SPEC_MATRIX.csv', M, delimiter=',')  # Save the matrix as CSV
    return M

def MATRIX_tra_spectra(w, wk, wc, g, Gamma_A, Gamma_B, Gamma_C, turn_on_ext_B, w_Bmod_dim, Bi, DB, wp, Dt):
    global DWCF, DGF, w_Bmod
    w_Bmod, DWCF = wc_fft(w_Bmod_dim, wc, Bi, DB, wp, Dt)
    w_Bmod, DGF = g_fft(w_Bmod_dim, wc, Bi, DB, wp, Dt, g)
    
    tran = []
    reflec = []
    absp = []

    M = Time_Mod_Matrix(w, wk, wc, g, Gamma_A, Gamma_B, Gamma_C, turn_on_ext_B)

    for i in range(len(w)):
        idx = i * 3
        tr = M[idx + 2, idx]  # Adjusted for zero-based indexing
        ref = 1 + M[idx, idx]
        ab = M[idx + 1, idx]
        tran.append(abs(tr) ** 2)
        reflec.append(abs(ref) ** 2)
        absp.append(abs(ab) ** 2)

    spectra = np.array([w, tran, reflec, absp])
    return spectra

# Parameters

w_dim = 500
w = wc * np.linspace(0, 2, w_dim)

trans_spectra = []

# Plotting
plt.figure()
for i in range(len(wp)):
    spectra_B_on = MATRIX_tra_spectra(w, wk, wc, g, Gamma_A, Gamma_B, Gamma_C, 
                                        turn_on_ext_B=1, 
                                        w_Bmod_dim=w_Bmod_dim, 
                                        Bi=Bi, 
                                        DB=DB, 
                                        wp=wp[i], 
                                        Dt=Dt[i])

    print('hi')
    plt.plot(w / THz, spectra_B_on[1,:],label='wp='+str(wp[i]/wc)+'wc')
    #plt.plot(w / THz, spectra_B_off[1,:],label='wp='+str(wp[i]/wc)+'wc')

plt.legend()
plt.xlabel('Frequency (THz)')
plt.ylabel('Transmission Spectra')
plt.title('Transmission Spectra vs Frequency')
plt.grid()
plt.show()

