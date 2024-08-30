    %   Physical parameter
%   Coupling strength and decaying rates
THz = 1e12*2*pi;
GHz = 1e9*2*pi;

Gamma_A = 2 * GHz;
Gamma_B = 5 * GHz;
wc = 0.4 * THz;
wk = 0.4 * THz;
g = 0.1 * wc;

spectra = tra_spectra(wk,wc,g,Gamma_A,Gamma_B);

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
g = 0.1 * wc;

width = [];

for i=1:length(Gamma_A)
    width_LP = transmission_pol_width(wk,wc,g,Gamma_A(i),Gamma_B);
    width = [width,width_LP];
end
figure
plot(Gamma_A/GHz,width/pi/1e9);
xlabel('Photon decaying rate(GHz)')
ylabel('Lower polariton transmission')

    %   Functions
%       Physics calculation functions
function spectra = tra_spectra(wk,wc,g,Gamma_A,Gamma_B)
    tran = [];
    reflec = [];
    absp = [];

    w =  linspace(0,2*wc,1001); % Angular frequencies
    for i=1:length(w)
        tran = [tran,abs(Transmission_SR(w(i),wk,wc,g,Gamma_A,Gamma_B))^2];
        reflec = [reflec,abs(Reflection_SR(w(i),wk,wc,g,Gamma_A,Gamma_B))^2];
        absp = [absp,abs(Absorption_SR(w(i),wk,wc,g,Gamma_A,Gamma_B))^2];
    end

    spectra = [w;tran;reflec;absp];
end

function G = System_Matrix(w,wk,wc,g,Ga,Gb)
    U11 = wk-1i*Ga-w;
    U12 = 1i*g;
    U13 = 0;
    U21 = -1i*g;
    U22 = wc-1i*Gb-w;
    U23 = 1i*g;
    U31 = 0;
    U32 = -1i*g;
    U33 = wk-1i*Ga-w;
    U = [U11,U12,U13;U21,U22,U23;U31,U32,U33];
    G = inv(U);
end

function trans_SR = Transmission_SR(w,wk,wc,g,Ga,Gb)
    G = System_Matrix(w,wk,wc,g,Ga,Gb);
    ga = sqrt(Ga/pi);
    gb = sqrt(Gb/pi);

    trans_SR = 2i*Ga*G(1,3);
end
function ref_SR = Reflection_SR(w,wk,wc,g,Ga,Gb)
    G = System_Matrix(w,wk,wc,g,Ga,Gb);
    ga = sqrt(Ga/pi);
    gb = sqrt(Gb/pi);

    ref_SR = 1+2i*Ga*G(1,1);
end
function abs_SR = Absorption_SR(w,wk,wc,g,Ga,Gb)
    G = System_Matrix(w,wk,wc,g,Ga,Gb);
    ga = sqrt(Ga/pi);
    gb = sqrt(Gb/pi);

    abs_SR = 2i*gb/ga*Ga*G(2,1);
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