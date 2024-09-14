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

main = 1;
if main ==1
%   Calculate the transmission spectra
w_dim = 1000;
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
end
    %   Functions
function yf = fft_analysis(yt)
    yf = fft(yt);
    N = length(yf);
    a = yf(1:N/2);
    b = yf(N/2+1:N);
    yf = [b,a];
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


function Bt = magnetic_time_domain(t,Bi,DB,wp,Dt)
    Bt = DB * cos(wp*t) * exp(-t^2/2/Dt^2) + Bi;
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

function M = Time_Mod_Matrix(w,wk,wc,g,Gamma_A,Gamma_B,Gamma_C,turn_on_ext_B)

    T = [];
    UA = [];
    UB = [];

    ga = sqrt(Gamma_A/pi);
    gb = sqrt(Gamma_B/pi);
    gc = sqrt(Gamma_C/pi);
    A = [ga,0,ga,0;0,gb,0,gb;gc,0,gc,0];
    B = [2i*Gamma_A/ga,0,2i*Gamma_C/gc;0,2i*Gamma_B/gb,0;2i*Gamma_A/ga,0,2i*Gamma_C/gc;0,2i*Gamma_B/gb,0];
    
    dw = w(2)-w(1);
    N = length(w);
    main = 0;
    if main==1
    for i = 1:N
        T_temp = [];
        ua = [];
        ub = [];
        
        for j = 1:N
            if turn_on_ext_B == 1
                DG = Modulated_Hopfield_Matrix(w(i),w(j),dw);
            else
                DG = zeros(4);
            end
            if i==j
                G = Hopfield_Matrix(w(j),wk,wc,g,Gamma_A,Gamma_B,Gamma_C);
                ua = [ua,A];
                T_temp = [T_temp,G + DG];
                ub = [ub,B];
            else
                ua = [ua,zeros(3,4)];
                T_temp = [T_temp,DG];
                ub = [ub,zeros(4,3)];
            end
            
        end
        UA = [UA;ua];
        T = [T;T_temp];
        UB = [UB;ub];
    end
    end
    
    gu = [];
    au = [ga,0,ga,0;0,gb,0,gb;gc,0,gc,0];
    bu = [2i*Gamma_A/ga,0,2i*Gamma_C/gc;0,2i*Gamma_B/gb,0;2i*Gamma_A/ga,0,2i*Gamma_C/gc;0,2i*Gamma_B/gb,0];
    
    for i = 1:N
        gu = [gu,Hopfield_Matrix(w(i),wk,wc,g,Gamma_A,Gamma_B,Gamma_C)];
    end

    G = zeros(4*N);
    A = kron(eye(N),au);
    B = kron(eye(N),bu);
    for i = 1:4
        a = ones(N,1);
        b = [0;0;0;0];
        b(i) = 1;
        c = kron(a,b);
        d = gu(i,:).* c;
        G = G + d .* kron(eye(N),ones(4));
    end

    M = A*inv(G)*B;
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

function G = Hopfield_Matrix(w,wk,wc,g,Ga,Gb,Gc)
    D = g^2/wc;
    G = [wk-1i*(Ga+Gc)-w+2*D,1i*g,2*D,-1i*g;
        -1i*g,wc-1i*Gb-w,-1i*g,0;
        -2*D,-1i*g,-wk-1i*(Ga+Gc)-w-2*D,1i*g;
        -1i*g,0,-1i*g,-wc-1i*Gb-w];
end

