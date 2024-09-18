    %   Physical parameter
%  Angular frequency unit
THz = 1e12*2*pi;
GHz = 1e9*2*pi;

%   Coupling strength and decaying rates
Gamma_A = 5 * GHz;
Gamma_B = 2 * GHz;
Gamma_C = 1 * GHz;
wc = 0.4 * THz;
wk = 0.4 * THz;
g = 0.2 * wc;

%   Modulating external magnetic field
e = 1.6e-19;
me = 9.11e-31;
me_GaAs = 0.067 * me;
Bi = me_GaAs * wc / e; 
DB = Bi * 0.5;
wp = wc * [2,1,0.5,0.25];
Dt = 10./wp;
w_Bmod_dim = 1000;

[w_Bmod,DWCF] = wc_fft(w_Bmod_dim,wc,Bi,DB,wp(1),Dt(1));
[w_Bmod,DGF] = g_fft(w_Bmod_dim,wc,Bi,DB,wp(1),Dt(1),g);
figure
plot(w_Bmod/1e12,real(DGF));
xlabel('Angular frequency (THz)')
ylabel('Coupling strength in frequency domain')


%   Calculate the transmission spectra
w_dim = 100;
w = wc * linspace(0,2,w_dim);

trans_spectra = [];

for i = 1:length(wp)
spectra_B_on = MATRIX_tra_spectra(w,wk,wc,g,Gamma_A,Gamma_B,Gamma_C,1,w_Bmod_dim,Bi,DB,wp(i),Dt(i));
trans_spectra = [trans_spectra ; spectra_B_on(2,:)];
end
spectra_B_off = MATRIX_tra_spectra(w,wk,wc,g,Gamma_A,Gamma_B,Gamma_C,0,w_Bmod_dim,Bi,DB,wp(1),Dt(1));
trans_spectra = [trans_spectra ; spectra_B_off(2,:)];

figure
hold on

for i = 1: (length(wp)+1)
    plot(w/THz,trans_spectra(i,:));
end

legend('wp = 2 wc','wp = 1 wc','wp = 0.5 wc','wp = 0.25 wc','Ext B turn off')
xlabel('Frequency(THz)')
ylabel('Transmission Spectra')
hold off

    %   Functions
function yf = fft_analysis(yt)
    yf = fft(yt);
    N = length(yf);
    a = yf(1:N/2);
    b = yf(N/2+1:N);
    yf = [b,a];
end

function Bt = magnetic_time_domain(t,Bi,DB,wp,Dt)
    Bt = DB * cos(wp*t) * exp(-t^2/2/Dt^2) + Bi;
end

function G = Hopfield_Matrix(w,wk,wc,g,Ga,Gb,Gc)
    D = g^2/wc;
    G = [wk-1i*(Ga+Gc)-w+2*D,1i*g,2*D,-1i*g;
        -1i*g,wc-1i*Gb-w,-1i*g,0;
        -2*D,-1i*g,-wk-1i*(Ga+Gc)-w-2*D,1i*g;
        -1i*g,0,-1i*g,-wc-1i*Gb-w];
end

function [w,DWCF] = wc_fft(w_dim,wc,Bi,DB,wp,Dt)
    N = w_dim;
    dw = wp * 6/N;
    w = dw * (-N/2:N/2-1);
    dt = 2 * pi/(dw * N);
    t = dt *(-N/2:N/2-1);
    wct = [];
    for i =1:length(t)
        Bt = magnetic_time_domain(t(i),Bi,DB,wp,Dt);
        wct = [wct, wc*(Bt/Bi) - wc];
    end
    DWCF = fft_analysis(wct)*dt;
end

function [w,DGF] = g_fft(w_dim,wc,Bi,DB,wp,Dt,g)
    N = w_dim;
    dw = wp * 6/N;
    w = dw * (-N/2:N/2-1);
    dt = 2 * pi/(dw * N);
    t = dt *(-N/2:N/2-1);
    gt = [];
    for i =1:length(t)
        Bt = magnetic_time_domain(t(i),Bi,DB,wp,Dt);
        gt = [gt, g*sqrt(Bt/Bi) - g];
    end
    DGF = fft_analysis(gt)*dt;   
end

function dg = g_mod_freq(wi,wj)
    global DGF
    global w_Bmod
    w = wi - wj;
    dg = interp1(w_Bmod,DGF,w);
    if isnan(dg)==1
        dg = 0;
    end
end

function dwc = wc_mod_freq(wi,wj)
    global DWCF
    global w_Bmod
    w = wi - wj;
    dwc = interp1(w_Bmod,DWCF,w);
    if isnan(dwc)==1
        dwc = 0;
    end
end

function G = init_G(N,w,wk,wc,g,Gamma_A,Gamma_B,Gamma_C)
    G = zeros(4*N);
    for i = 1:N
        Gu = Hopfield_Matrix(w(i),wk,wc,g,Gamma_A,Gamma_B,Gamma_C);
        m = 4*(i-1);
        for p=1:4
            for q = 1:4
                G(m+p,m+q) = Gu(p,q);
            end
        end
    end
end

function DG = Modulated_Hopfield_Matrix(wi,wj,dw)
    dg = g_mod_freq(wi,wj);
    dwc = wc_mod_freq(wi,wj);
    
    DG = [0,1i*dg,0,-1i*dg;
         -1i*dg,dwc,-1i*dg,0;
         0,-1i*dg,0,1i*dg;
         -1i*dg,0,-1i*dg,-dwc];
    DG = DG * dw;
end

function G_mod = Mod_G(N,w,dw)
    G_mod_middle = [];
    for j  = 1:N
        DG = Modulated_Hopfield_Matrix(w(N/2),w(j),dw);
        G_mod_middle = [G_mod_middle,DG];
    end
    
    G_mod = zeros(4*N);
    for i =1:N
        for j = 1:N

            m = 4*(i-1);
            n = 4*(j-1);
            s = 4*(j-1 - (i-N/2));
            if s>0 & s<4*N
                DGU = G_mod_middle(1:4, s+1:s+4 );
                G_mod(m+1:m+4,n+1:n+4) = DGU;
            end

        end
    end
end

function M = final_M(N,H,au,bu)
    M = zeros(3*N);

    for i = 1:N %9sec
        for j = 1:N
            m = 3*(i-1);
            n = 3*(j-1);
            M(m+1:m+3,n+1:n+3) = au*H(4*(i-1)+1:4*(i-1)+4,4*(j-1)+1:4*(j-1)+4)*bu;
        end
    end
end

function M = Time_Mod_Matrix(w,wk,wc,g,Gamma_A,Gamma_B,Gamma_C,turn_on_ext_B)

    T = [];
    UA = [];
    UB = [];

    ga = sqrt(Gamma_A/pi);
    gb = sqrt(Gamma_B/pi);
    gc = sqrt(Gamma_C/pi);
   
    dw = w(2)-w(1);
    N = length(w);

    au = [ga,0,ga,0;0,gb,0,gb;gc,0,gc,0];
    bu = [2i*Gamma_A/ga,0,2i*Gamma_C/gc;0,2i*Gamma_B/gb,0;2i*Gamma_A/ga,0,2i*Gamma_C/gc;0,2i*Gamma_B/gb,0];

    
    G = init_G(N,w,wk,wc,g,Gamma_A,Gamma_B,Gamma_C);
    G_mod = Mod_G(N,w,dw);
    
    G = G + G_mod;

    H  = inv(G);
    
    M = final_M(N,H,au,bu);
    writematrix(M,'SPEC_MATRIX.csv') 
end

function spectra = MATRIX_tra_spectra(w,wk,wc,g,Gamma_A,Gamma_B,Gamma_C,turn_on_ext_B,w_Bmod_dim,Bi,DB,wp,Dt)
    %   External magnetic field
    global DWCF
    global DGF
    global w_Bmod
    [w_Bmod,DWCF] = wc_fft(w_Bmod_dim,wc,Bi,DB,wp,Dt);
    [w_Bmod,DGF] = g_fft(w_Bmod_dim,wc,Bi,DB,wp,Dt,g);
    
    tran = [];
    reflec = [];
    absp = [];

    M = Time_Mod_Matrix(w,wk,wc,g,Gamma_A,Gamma_B,Gamma_C,turn_on_ext_B);

    for i=1:length(w)
        idx = (i-1)*3;
        tr = M(idx+3,idx+1);
        ref = 1 + M(idx+1,idx+1);
        ab = M(idx+2,idx+1);
        tran = [tran,abs(tr)^2];
        reflec = [reflec,abs(ref)^2];
        absp = [absp,abs(ab)^2];
    end

    spectra = [w;tran;reflec;absp];
end