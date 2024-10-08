import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.sparse.linalg import inv

# Assuming DGF and w_Bmod are defined globally or passed as arguments
DGF = None  # Placeholder, set this according to your program
w_Bmod = None  # Placeholder, set this according to your program

def fft_analysis(yt):
    yf = np.fft.fft(np.fft.fftshift(yt))
    N = len(yf)
    a = yf[:N//2]
    b = yf[N//2:]
    yf = np.concatenate((b, a))
    return yf

def magnetic_time_domain(t, Bi, DB, wp, Dt):
    #   Heaviside function
    if t>0:
        Bt = Bi + DB
    else:
        Bt = Bi
    
    #   Pulse
    #Bt =  DB * np.cos(wp * t) * np.exp(-t**2 / (2 * Dt**2) ) + Bi
    #   Continuous wave
    Bt =  DB * np.cos(wp * t)  + Bi
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
    dw = wp * 20 /N
    T_tot = 20 /dw
    t = T_tot/N * np.arange(-N/2, N/2)
    
    wct = []
    for i in range(len(t)):
        Bt = magnetic_time_domain(t[i], Bi, DB, wp, Dt)
        wct.append( wc * (Bt / Bi) - wc)
    
    wct = np.array(wct)
    DWCF = fft(wct) * T_tot/N
    w = fftfreq(N, T_tot/N) * 2 * np.pi

    return w, DWCF

def g_fft(w_dim, wc, Bi, DB, wp, Dt, g):
    N = w_dim
    dw = wp * 20 /N
    T_tot = 20 /dw
    t = T_tot/N * np.arange(-N/2, N/2)
    
    gt = []
    for i in range(len(t)):
        Bt = magnetic_time_domain(t[i], Bi, DB, wp, Dt)
        gt.append(g * np.sqrt(np.abs(Bt / Bi)) - g)
    
    gt = np.array(gt)
    DGF = fft(gt) * T_tot/N
    w = fftfreq(N, T_tot/N) * 2 * np.pi

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
DB = Bi * 0.5
wp = wc * np.array([2])
Dt = 10. / wp
w_Bmod_dim = 10000

# Perform FFT analyses
w_Bmod, DWCF = wc_fft(w_Bmod_dim, wc, Bi, DB, wp[0], Dt[0])
w_Bmod, DGF = g_fft(w_Bmod_dim, wc, Bi, DB, wp[0], Dt[0], g)

ft = []
for i in range(len(w_Bmod)):
    ft.append((2 * np.pi * Dt[0]**2)**0.5 * np.exp(- (w_Bmod[i] * Dt[0])**2 /2) *DB/Bi *wc )
    

# Plotting
plt.figure()
plt.plot(w_Bmod / 1e12, np.abs(DGF))
plt.xlabel(r'$\omega$ (THz)')
plt.ylabel(r'$\tilde{\Omega}_{Rabi, mod}(\omega)$')
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

def Heaviside_Mod_G(w, g, wc):
    Bi = 1
    DB = 0.5

    dg = g * np.sqrt(np.abs( (Bi + DB) / Bi)) - g
    dwc = wc *  ((Bi + DB) / Bi) - wc

    dg = dg * 0.5
    dwc = dwc * 0.5
    
    DG = np.array([[0, 1j * dg, 0, -1j * dg],
                   [-1j * dg, dwc, -1j * dg, 0],
                   [0, -1j * dg, 0, 1j * dg],
                   [-1j * dg, 0, -1j * dg, -dwc]], dtype=complex)
    
    N = len(w)
    G_mod = np.zeros((4 * N, 4 * N), dtype=complex)

    for i in range(N):
        for j in range(N):
            m = 4 * i
            n = 4 * j
            G_mod[m:m + 4, n:n + 4] = DG
                
    return G_mod

def inverse_reduced_matrix(G):
    N = len(G)//4
    H = np.zeros((4 * N, 4 * N), dtype=complex)
    for i in range(N):
        m = 4 * i
        GU = G[m:m + 4, m:m + 4]
        HU = np.linalg.inv(GU)
        H[m:m + 4, m:m + 4] = HU

    return H



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
        #G_mod = Heaviside_Mod_G(w, g, wc)

        G += G_mod  # Element-wise addition

    H = np.linalg.inv(G)
    #H = inverse_reduced_matrix(G)


    plt.imshow(np.abs(G), cmap='viridis', aspect='auto')  # You can choose other colormaps
    plt.colorbar()  # Show color scale
    plt.title('Matrix Heatmap')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()

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

def THz_spec(w, wk, wc, g, Gamma_A, Gamma_B, Gamma_C, turn_on_ext_B, w_Bmod_dim, Bi, DB, wp, Dt):
    global DWCF, DGF, w_Bmod
    w_Bmod, DWCF = wc_fft(w_Bmod_dim, wc, Bi, DB, wp, Dt)
    w_Bmod, DGF = g_fft(w_Bmod_dim, wc, Bi, DB, wp, Dt, g)
    
    inc_spec = []
    
    for i in range(len(w)):
        u = w[i]/(0.4*THz)
        inc_spec.append(np.exp(-(u-1)**2))

    inc_spec = np.array(inc_spec)
    tr_spec = []

    M = Time_Mod_Matrix(w, wk, wc, g, Gamma_A, Gamma_B, Gamma_C, turn_on_ext_B)

    for i in range(len(w)):
        idx = i * 3
        tr = M[idx + 2, ::3]  # Adjusted for zero-based indexing
        tr_spec.append(np.abs(np.dot(tr,inc_spec))**2)


    return tr_spec

# Parameters

w_dim = 1000
w = wc * np.linspace(0, 2, w_dim)

trans_spectra = []

# Plotting
plt.figure()
for i in range(len(wp)):
    """spectra_B_on = MATRIX_tra_spectra(w, wk, wc, g, Gamma_A, Gamma_B, Gamma_C, 
                                        turn_on_ext_B=1, 
                                        w_Bmod_dim=w_Bmod_dim, 
                                        Bi=Bi, 
                                        DB=DB, 
                                        wp=wp[i], 
                                        Dt=Dt[i])

    print('hi')
    spectra_B_off = MATRIX_tra_spectra(w, wk, wc, g, Gamma_A, Gamma_B, Gamma_C, 
                                        turn_on_ext_B=0, 
                                        w_Bmod_dim=w_Bmod_dim, 
                                        Bi=Bi, 
                                        DB=DB, 
                                        wp=wp[i], 
                                        Dt=Dt[i])"""
    spectra_B_on = THz_spec(w, wk, wc, g, Gamma_A, Gamma_B, Gamma_C, 
                                        turn_on_ext_B=1, 
                                        w_Bmod_dim=w_Bmod_dim, 
                                        Bi=Bi, 
                                        DB=DB, 
                                        wp=wp[i], 
                                        Dt=Dt[i])

    print('hi')
    spectra_B_off = THz_spec(w, wk, wc, g, Gamma_A, Gamma_B, Gamma_C, 
                                        turn_on_ext_B=0, 
                                        w_Bmod_dim=w_Bmod_dim, 
                                        Bi=Bi, 
                                        DB=DB, 
                                        wp=wp[i], 
                                        Dt=Dt[i])
    print('hi')
    plt.plot(w / THz, spectra_B_on,label=r'$\omega_p$='+str(wp[i]/wc)+r'$\omega_c$')
    plt.plot(w / THz, spectra_B_off,label='No modulation')

plt.legend()
plt.xlabel('Frequency (THz)')
plt.ylabel('Transmission Spectra')
plt.title('Transmission Spectra vs Frequency')
plt.grid()
plt.show()
