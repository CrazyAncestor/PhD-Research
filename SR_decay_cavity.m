    %   Physical parameters
%       Refraction index
n_Si = 3.4;
n_Vac = 1.;
n_defect = n_Si;

%       Central frequency wavelength
lambda0 = 300;  %   unit: u

%       Thickness
d_Si = 0.75*lambda0/n_Si; %   unit: um
d_Vac = 0.25*lambda0/n_Vac;
d_defect = 2*d_Si;

%       DBR total layer number
DBR_layer = 4;

%   Build DBR & Tamm cavity parameters
DBR = make_DBR(d_Si,n_Si,d_Vac,n_Vac,DBR_layer);
Tamm = make_die_Tamm(DBR, d_defect, n_defect);

%   Give frequencies and unit cell length.
w = 2*pi* linspace(0.6,1.4,10001)/lambda0/n_Vac; % Angular frequencies
One_THz = 2*pi* 1/lambda0/n_Vac;

%   Calculate and plot the field of 1D Tamm cavity at 1THz
[r_1THz, field_1THz] = TMM_analysis(One_THz,Tamm,100);
%plot_field('Tamm field enhancement',abs(field_1THz),Tamm)

%   Calculate DBR and Tamm cavity reflection coefficients for different
%   wavenumbers
reflections_Tamm = [];

for i=1:length(w)
    [r_Tamm, field_Tamm] = TMM_analysis(w(i),Tamm,0);
    reflections_Tamm = [reflections_Tamm , r_Tamm];
end

%   Plot reflection coefficients vs. frequency
%figure
%title('Reflection coefficient vs. frequency')
%plot(w*lambda0/2/pi,abs(reflections_Tamm).^2)

%legend('Tamm reflection')

%xlabel('Frequency(THz)')
%ylabel('Reflection')

%   Quality factor
Q_ref = Quality_factor_reflection(w,[One_THz*0.8,One_THz*1.2],abs(reflections_Tamm).^2)
Q_eng = Quality_factor_energy(field_1THz,One_THz)

%   Coupling strength
h = 6.63e-34/2/pi;
w = 1e12*2*pi;
hw = h*w;
ne = 1.9e11 / (1e-2)^2;
m = 9.11e-31*0.07;
epsilon = 8.85e-12;
c = 3e8;
e = 1.6e-19;

%   Cavity mode coupling strength
g = 0.4e12 *0.1*2*pi;%coupling_strength(hw,field_1THz,ne,m,e,epsilon)
g_SR = e^2 * ne / m / epsilon / (1+3.8)/ c
Gamma_A = 1e9*2*pi*linspace(2,2,5);%1e12/Q_ref
Gamma_B = 1e9*2*pi*linspace(1,5,5);
wc = 0.4e12 *2*pi;
wk = 0.4e12 *2*pi;
trans = [];
reflecs = [];
absps = [];
tot = [];
w =  linspace(0.3,0.5,1001) *1e12*2*pi; % Angular frequencies

for k = 1:length(Gamma_A)
    tran = [];
    reflec = [];
    absp = [];
for i=1:length(w)
    tran = [tran,abs(Transmission_SR(w(i),wk,wc,g,Gamma_A(k),Gamma_B(k)))^2];
    reflec = [reflec,abs(Reflection_SR(w(i),wk,wc,g,Gamma_A(k),Gamma_B(k)))^2];
    absp = [absp,abs(Absorption_SR(w(i),wk,wc,g,Gamma_A(k),Gamma_B(k)))^2];
end
    trans = [trans;tran];
    reflecs = [reflecs;reflec];
    absps = [absps;absp];
end
figure
hold on
for k = 1:length(Gamma_A)
plot(w/2/pi/1e12,trans(k,:));
[pks,locs,widths,proms] = findpeaks(trans(k,:),w/2/pi/1e12);
widths
end
%plot(w/2/pi/1e12,reflec);
%plot(w/2/pi/1e12,absp);
legend()
xlabel('Frequency(THz)')
ylabel('Transmission/Reflection/Absorption Spectrum')
%legend('Transmittance','Reflectance','Absorption')
hold off
%[w0,sig,t_decay] = fit_lorentzian(w(250:350),sqrt(trans(250:350)));
%t_decay


sp_decay= 0 ;
if sp_decay==1
%   Spontaneous Emission Rate
g = 0.4e12 *2*pi *linspace(0.001,1,100);
g_SR = e^2 * ne / m / epsilon / (1+3.8)/ c
Gamma_A = 0.05*0.4e12*2*pi;%1e12/Q_ref
Gamma_B = 0.05*0.4e12*2*pi;
wc = 0.4e12 *2*pi;
wk = 0.4e12 *2*pi;
sp = [];
for i=1:length(g)
    sp = [sp,spontaneous_emission_rate(g(i),wk,wc,Gamma_A,Gamma_B)];
end
r_sat = 2*Gamma_A*Gamma_B/(Gamma_A+Gamma_B);
figure
plot(log(g/wc),log(sp/r_sat/pi));
end

    %   Functions
%       Physics calculation functions
%   Transfer matrix formula (Macleod, H. A. (Hugh A. (2001). Thin-film optical filters / H.A. Macleod. (Third edition.). Institute of Physics Pub.)
function M = transfer_matrix(k,d,n)
    delta = k*n*d;
    M = [cos(delta),sin(delta)/n*1i;sin(delta)*n*1i,cos(delta)];
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

%   Electric field amplitude spatial enhancement for multilayer structure
function [r, field] = TMM_analysis(w,parameters,N_mesh)
    
    %   Giving thickness and refractive index
    d = parameters(1,:);
    n = parameters(2,:);
    
    %   Defining variables
    k = w;
    E_H_exit = [1;1]; % E and H fields at the end of the multilayer

    x = [sum(d)];
    x_present = sum(d);

    refractive_idx = [n(length(n))];

    E_H = E_H_exit;
    field = [E_H_exit];
    
    %   Calculating the E,H fields with transfer matrices
    N = length(d);
    for m=1:N
        ms = N-m+1;
        if N_mesh==0
            M = transfer_matrix(k,d(ms),n(ms));
            E_H = M * E_H;
        else
            for j=1:N_mesh
                M = transfer_matrix(k,d(ms)/N_mesh,n(ms));
                E_H = M * E_H;
                x_present = x_present - d(ms)/N_mesh;
                x = [x, x_present];
                refractive_idx = [refractive_idx,n(ms)];
                field = [field,E_H];
            end
        end
    end
    %   Calculating reflection coefficients
    n_eff = E_H(2) / E_H(1);
    r = (1-n_eff) / (1+n_eff);
    %   Giving solutions of field
    x = flip(x);
    refractive_idx = flip(refractive_idx);
    E = flip(field(1,:)/abs(E_H(1)));
    H = flip(field(2,:)/abs(E_H(2)));
    
    field = [x;refractive_idx;E;H];
end

%   Transmission coefficient calculation of a unit cell
function [r,t] = Unit_cell_rt_coefficient(K,parameters)
    
    %   Giving thickness and refractive index
    d = parameters(1,:);
    n = parameters(2,:);
    
    t = [];
    r = [];
    for j = 1:length(K)
        %   Defining variables
        E_H_exit = [1;1]; % E and H fields at the end of the multilayer
        x = [sum(d)];
        x_present = sum(d);
        E = [1];
        E_H = E_H_exit;
        
        %   Calculating the E,H fields with transfer matrices
        N = length(d);
        for m=1:N
            ms = N-m+1;
            M = transfer_matrix(K(j),d(ms),n(ms));
            E_H = M * E_H;
        end
        field = [x;E/E(length(E))];
        
        %   Calculating reflection coefficients
        n_eff = E_H(2) / E_H(1);
        r0 = (1-n_eff) / (1+n_eff);
        t0 = 2 / (E_H(2)+E_H(1));
        r = [r,r0];
        t = [t,t0];
    end
end

function [k,w] = bandstructure(w,a,d1,n1,d2,n2)
    DBR_unit_cell = make_DBR_unit_cells(d1,n1,d2,n2);
    [r,t] = Unit_cell_rt_coefficient(w,DBR_unit_cell);
    k = [];
    for i = 1:length(w)
        K = w(i);
        c_k = cos(K*(n1*d1+n2*d2))/abs(t(i));
        k = [k, acos(c_k)/a];
    end
end

%   Get attenuation factor
function attenuation = Get_Attenuation(k,parameters,N_mesh)
    
    %   Giving thickness and refractive index
    d = parameters(1,:);
    n = parameters(2,:);
    
    %   Defining variables
    E_H_exit = [1;1]; % E and H fields at the end of the multilayer
    x = [];
    x_present = sum(d);
    E = [];
    E_H = E_H_exit;
    
    %   Calculating the E,H fields with transfer matrices
    N = length(d);
    for m=1:N
        ms = N-m+1;
        x0 = [];
        E0 = [];
        
        for j=1:N_mesh
            M = transfer_matrix(k,d(ms)/N_mesh,n(ms));
            E_H = M * E_H;
            x_present = x_present - d(ms)/N_mesh;
            x0 = [x0, x_present];
            E0 = [E0, abs(E_H(1))];
        end
        if mod(ms,2)==1
           [max_E,idx] = max(E0);
           E = [E,E0(idx)];
           x = [x,x0(idx)];
        end  
    end
    
    %   Giving solutions of field
    field = [flip(x);log(flip(E))];
    P = polyfit(flip(x),log(flip(E)),1);
    slope = P(1);
    attenuation = slope;
end

%       Make multilayer structures
function DBR = make_DBR(d1,n1,d2,n2,N)
    
    %   Define list of thickness d and refractive index n
    d = [];
    n = [];

    %   Assign element value, first layer is material 1 (d1,n1), second
    %   layer is material 2 (d2,n2), and so on.
    for m=1:N
        if mod(m,2)==1
            d = [d,d1];
            n = [n,n1];
        else
            d = [d,d2];
            n = [n,n2];
        end
    end
    DBR = [d;n];
end

function DBR = make_DBR_unit_cells(d1,n1,d2,n2)
    
    %   Define list of thickness d and refractive index n
    d = [];
    n = [];

    %   Assign element value, first layer is material 1 (d1,n1), second
    %   layer is material 2 (d2,n2), and so on.
    for m=1:3
        if mod(m,2)==1
            d = [d,d1/2.];
            n = [n,n1];
        else
            d = [d,d2];
            n = [n,n2];
        end
    end
    DBR = [d;n];
end

function Tamm = make_mtl_Tamm(DBR,d_metal,n_metal)
    d = DBR(1,:);
    n = DBR(2,:);
    d = [d,d_metal];
    n = [n,n_metal];
    Tamm = [d;n];
end

function Tamm = make_die_Tamm(DBR,d_defect,n_defect)
    d = DBR(1,:);
    n = DBR(2,:);
    d_rev = flip(d);
    n_rev = flip(n);
    d = [d,d_defect];
    n = [n,n_defect];
    d = [d,d_rev];
    n = [n,n_rev];
    Tamm = [d;n];
end

%       Plotting functions
function Smith_Chart(data,title_name)
    figure
    title(title_name)
    hold on
    x_smith = [];
    y_smith = [];
    for i=1:length(data)
        amp = abs(data(i));
        ang = angle(data(i));
        x_smith = [x_smith,amp*cos(ang)];
        y_smith = [y_smith,amp*sin(ang)];
    end
    plot(x_smith,y_smith);
    hold off
    grid("on")
    ax = gca;
    ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';
    axis([-1 1 -1 1])
end

function plot_field(name,field,structure)
    figure
    title(name)
    hold on
    x = field(1,:);
    E = field(3,:);
    block_x = [0];
    x_present = 0;
    Height = max(E);
    d = structure(1,:);
    N = length(d);
    for i=1:N
        if mod(i,2)==1
            block_x=[block_x,d(i)+x_present]; 
            x_present = x_present + d(i);
            patch('Faces',[1 2 3 4],'Vertices',[ block_x(i) 0; block_x(i+1) 0; block_x(i+1) Height; block_x(i) Height],'FaceColor',[198 198 198]./255)
        else
            block_x=[block_x,d(i)+x_present];
            x_present = x_present + d(i);
            patch('Faces',[1 2 3 4],'Vertices',[ block_x(i) 0; block_x(i+1) 0; block_x(i+1) Height; block_x(i) Height],'FaceColor',[255 255 255]./255)
        end
        
    end
    plot(x,E)
    xlabel('position(um)')
    ylabel('Electric Field Enhancement')
    
    hold off
end

function Q = Quality_factor_reflection(w,filter,R)
    wmin = filter(1); wmax = filter(2);
    idx = find(w>wmin & w<wmax);
    T = ones(1,length(R));
    T = T-R;
    [w0,sig,t_decay] = fit_lorentzian(w(idx),T(idx));
    Q = w0/sig;
end

function Q = Quality_factor_energy(field,w0)
    x = field(1,:); refractive_idx = field(2,:); E = field(3,:); H = field(4,:);
    energy_field = 0;
    N = length(x);

    for i=1:N
        if i ==1
            dx = x(2) - x(1);
        else
            dx = x(i) - x(i-1);
        end
        eps = refractive_idx(i)^2;
        energy_field = energy_field + 0.5*(eps*real(E(i))^2 + real(H(i))^2)*dx;
    end
    S1 = real(E(1)*H(1));
    S2 = real(E(N)*H(N));

    Q = energy_field/(S1+S2)*w0;
end

function field_norm = normalizefield(field)
    x = field(1,:); refractive_idx = field(2,:); E = field(3,:); H = field(4,:);
    energy_field = 0;
    N = length(x);

    for i=1:N
        if i ==1
            dx = x(2) - x(1);
        else
            dx = x(i) - x(i-1);
        end
        eps = refractive_idx(i)^2;
        energy_field = energy_field + eps*abs(E(i))^2 *dx;
    end

    L = x(length(x));
    energy_field = energy_field/L;
    norm_fac = sqrt(1/energy_field);
    E_norm = E*norm_fac;
    H_norm = H*norm_fac;
    field_norm = [x;refractive_idx;E_norm;H_norm];
end

function g = coupling_strength(hw,field,ne,m,e,epsilon)
    field_norm = normalizefield(field);
    E = abs(field_norm(3,:));
    x = field_norm(1,:);
    L = x(length(x))*1e-6;
    u0 = max(E);
    g = e/2*sqrt(ne/m/L/epsilon)*u0;
end

function [w0,sig,t_decay] = fit_lorentzian(x,y)
    [y0,idx0] = max(abs(y));
    w0 = x(idx0);
    function y_fit = lorentzian(x,y0,sig)
        y_fit = [];
        for i=1:length(x)
            y_fit = [y_fit,y0/(1+((x(i)-x(idx0))/sig)^2) ];
        end
    end
    
    function err = error(sig,x,y)
        y_fit = lorentzian(x,y0,sig);
        err=sum((abs(y)-y_fit).*(abs(y)-y_fit));
    end
    fun = @(sig)error(sig,x,y);
    sig = fminbnd(fun,0,w0);
    t_decay = 1/sig;
end

function filter_trans(x,y)
[pks,locs,widths,proms] = findpeaks(y,x);
end