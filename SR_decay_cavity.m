    %   Physical parameter
%   Coupling strength and decaying rates
THz = 1e12*2*pi;
GHz = 1e9*2*pi;

Gamma_A = 5 * GHz;
Gamma_B = 2 * GHz;
Gamma_C = 1 * GHz;
wc = 0.4 * THz;
wk = 0.4 * THz;
g = 0.2 * wc;


Bi = 1;
DB = 0.5;
wp = wc;
Dt = 10/wp;
dB_freq(w_dim,wc,Bi,DB,wp,Dt);

main = 0;

if main==1
global model_usage
model_usage = 0;
w_dim = 250;

spectra = MATRIX_tra_spectra(w_dim,wk,wc,g,Gamma_A,Gamma_B,Gamma_C);

figure
hold on
for k = 2:4
    w = spectra(1,:);
    plot(w/THz,spectra(k,:));
end
legend('Transmission','Reflection','Absorption')
xlabel('Frequency(THz)')
ylabel('Transmission/Reflection/Absorption Spectra')
hold off

%   Coupling strength and decaying rates
THz = 1e12*2*pi;
GHz = 1e9*2*pi;

Gamma_A = 5 * GHz;
Gamma_B = 2 * GHz;
Gamma_C = 1 * GHz;
wc = 0.4 * THz;
wk = 0.4 * THz;
g = 0. * wc;


spectra = tra_spectra(wk,wc,g,Gamma_A,Gamma_B,Gamma_C);

figure
hold on
for k = 2:4
    w = spectra(1,:);
    plot(w/THz,spectra(k,:));
end
legend('Transmission','Reflection','Absorption')
xlabel('Frequency(THz)')
ylabel('Transmission/Reflection/Absorption Spectra')
hold off

%   Free space SR decaying rate
ne = 1.9e11 / (1e-2)^2;
m = 9.11e-31*0.07;
epsilon = 8.85e-12;
c = 3e8;
e = 1.6e-19;
n_GaAs = 3.8;
SR_decay_free_space = e^2 * ne / m / epsilon / (1+n_GaAs)/ c /GHz

%   Transmission width of polariton modes
Gamma_A = 2 * GHz;
Gamma_B = 1 * GHz * linspace(1,5,100);
wc = 0.4 * THz;
wk = 0.4 * THz;
g = 0.1 * wc;

width = [];


for i=1:length(Gamma_B)
    width_LP = transmission_pol_width(wk,wc,g,Gamma_A,Gamma_B(i));
    width = [width,width_LP/pi/1e9];
end
figure
plot(Gamma_B/GHz,width);
xlabel('Electron decaying rate(GHz)')
ylabel('Lower polariton transmission')

%   Transmission width of polariton modes
Gamma_A = 1 * GHz * linspace(0.01,5,100);
Gamma_B = 5 * GHz ;
wc = 0.4 * THz;
wk = 0.4 * THz;
g = 0.1 * wc ;

width = [];

for i=1:length(Gamma_A)
    width_LP = transmission_pol_width(wk,wc,g,Gamma_A(i),Gamma_B);
    width = [width,width_LP];
end
figure
plot(Gamma_A/GHz,width/pi/1e9);
xlabel('Photon decaying rate(GHz)')
ylabel('Lower polariton transmission')
end
    %   Functions
%       Physics calculation functions
function dB_freq(w_dim,wc,Bi,DB,wp,Dt)
    w = wc*linspace(-2*pi,2*pi,w_dim);
    dw = w(2) - w(1);
    t = 1/dw *linspace(-1,1,w_dim);
    Bt = [];
    for i =1:length(t)
        Bt = [Bt,magnetic_time_domain(t(i),Bi,DB,wp,Dt)];
    end
    Bf = fft(Bt);
    plot(w,Bf);
end

function Bt = magnetic_time_domain(t,Bi,DB,wp,Dt)
    Bt = DB * cos(wp*t) * exp(-t^2/2/Dt^2);
end

function spectra = MATRIX_tra_spectra(w_dim,wk,wc,g,Gamma_A,Gamma_B,Gamma_C)
    tran = [];
    reflec = [];
    absp = [];

    w = wc*linspace(0,2,w_dim);
    M = Time_Mod_Matrix(w_dim,wk,wc,g,Gamma_A,Gamma_B,Gamma_C);

    for i=1:w_dim
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

function M = Time_Mod_Matrix(w_dim,wk,wc,g,Gamma_A,Gamma_B,Gamma_C)
    w = wc*linspace(0,2,w_dim);
    %dw = w(2)-w(1);
    T = [];
    UA = [];
    UB = [];

    ga = sqrt(Gamma_A/pi);
    gb = sqrt(Gamma_B/pi);
    gc = sqrt(Gamma_C/pi);
    A = [ga,0,ga,0;0,gb,0,gb;gc,0,gc,0];
    B = [2i*Gamma_A/ga,0,2i*Gamma_C/gc;0,2i*Gamma_B/gb,0;2i*Gamma_A/ga,0,2i*Gamma_C/gc;0,2i*Gamma_B/gb,0];

    for i = 1:w_dim
        T_temp = [];
        ua = [];
        ub = [];
        for j = 1:w_dim
            if i==j
                G = Hopfield_Matrix(w(j),wk,wc,g,Gamma_A,Gamma_B,Gamma_C);
                ua = [ua,A];
                T_temp = [T_temp,G];
                ub = [ub,B];
            else
                ua = [ua,zeros(3,4)];
                T_temp = [T_temp,zeros(4)];
                ub = [ub,zeros(4,3)];
            end
            
        end
        UA = [UA;ua];
        T = [T;T_temp];
        UB = [UB;ub];
    end
    M = UA * inv(T) * UB;
end

function spectra = tra_spectra(wk,wc,g,Gamma_A,Gamma_B,Gamma_C)
    tran = [];
    reflec = [];
    absp = [];

    w =  linspace(0,2*wc,1001); % Angular frequencies
    for i=1:length(w)
        tr = Transmission_SR(w(i),wk,wc,g,Gamma_A,Gamma_B,Gamma_C);
        ref = Reflection_SR(w(i),wk,wc,g,Gamma_A,Gamma_B,Gamma_C);
        ab = Absorption_SR(w(i),wk,wc,g,Gamma_A,Gamma_B,Gamma_C);
        tran = [tran,abs(tr)^2];
        reflec = [reflec,abs(ref)^2];
        absp = [absp,abs(ab)^2];
    end

    spectra = [w;tran;reflec;absp];
end

function G = RWA_Matrix(w,wk,wc,g,Ga,Gb,Gc)
    U = [wk-1i*(Ga+Gc)-w, 1i*g;
        -1i*g, wc-1i*Gb-w];
    G = U;
end

function G = Hopfield_Matrix(w,wk,wc,g,Ga,Gb,Gc)
    D = g^2/wc;
    U = [wk-1i*(Ga+Gc)-w+2*D,1i*g,2*D,-1i*g;
        -1i*g,wc-1i*Gb-w,-1i*g,0;
        -2*D,-1i*g,-wk-1i*(Ga+Gc)-w-2*D,1i*g;
        -1i*g,0,-1i*g,-wc-1i*Gb-w];
    G = U;
end

function U = TRA_Matrix(w,wk,wc,g,Ga,Gb,Gc)

    ga = sqrt(Ga/pi);
    gb = sqrt(Gb/pi);
    gc = sqrt(Gc/pi);

    global model_usage
    if model_usage==0
        G = RWA_Matrix(w,wk,wc,g,Ga,Gb,Gc);
        G = inv(G);
        A = [ga,0;0,gb;gc,0];
        B = [2i*Ga/ga,0,2i*Gc/gc;0,2i*Gb/gb,0];
        U = A*G*B;
    elseif model_usage==1
        G = Hopfield_Matrix(w,wk,wc,g,Ga,Gb,Gc);
        G = inv(G);
        A = [ga,0,ga,0;0,gb,0,gb;gc,0,gc,0];
        B = [2i*Ga/ga,0,2i*Gc/gc;0,2i*Gb/gb,0;2i*Ga/ga,0,2i*Gc/gc;0,2i*Gb/gb,0];
        U = A*G*B;
    end
    
    
end

function trans_SR = Transmission_SR(w,wk,wc,g,Ga,Gb,Gc)
    U = TRA_Matrix(w,wk,wc,g,Ga,Gb,Gc);
    trans_SR = U(3,1);
end
function ref_SR = Reflection_SR(w,wk,wc,g,Ga,Gb,Gc)
    U = TRA_Matrix(w,wk,wc,g,Ga,Gb,Gc);
    ref_SR = 1 + U(1,1);
end
function abs_SR = Absorption_SR(w,wk,wc,g,Ga,Gb,Gc)
    U = TRA_Matrix(w,wk,wc,g,Ga,Gb,Gc);
    abs_SR = U(2,1);
end

function SP_rate = spontaneous_emission_rate(g,wk,wc,Ga,Gb)
    w =  linspace(0.0,0.8,1001) *1e12*2*pi; % Angular frequencies
    absp = [];
    for i=1:length(w)
        absp = [absp,abs(Absorption_SR(w(i),wk,wc,g,Ga,Gb))^2];
    end
    dw = w(2)-w(1);
    SP_rate = sum(absp)*dw;
end

function width_LP = transmission_pol_width(wk,wc,g,Gamma_A,Gamma_B)

    tran = [];
    reflec = [];
    absp = [];

    w =  linspace(0,2*wc,1001); % Angular frequencies

    for i=1:length(w)
        tran = [tran,abs(Transmission_SR(w(i),wk,wc,g,Gamma_A,Gamma_B))^2];
        reflec = [reflec,abs(Reflection_SR(w(i),wk,wc,g,Gamma_A,Gamma_B))^2];
        absp = [absp,abs(Absorption_SR(w(i),wk,wc,g,Gamma_A,Gamma_B))^2];
        
    end

    [pks,locs,widths,proms] = findpeaks(tran,w);
    width_LP = widths(1);
end