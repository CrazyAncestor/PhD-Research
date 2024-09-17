%   Problem 1c
alpha = 1e-4;
theta_m = 1e-1;
g = 10;
l = 1;
w = (g/l)^0.5;
T  = 2*pi/w;

t = linspace(0,0.3*T,100);
vz = [];

for i=1:length(t)
    vz = [vz,velocity(t(i),g,w,alpha,theta_m)];
end
figure
plot(t/T,vz);
xlabel('time (Period)')
ylabel('Vz (m/s)')
title('Velocity of z with time')


%   Problem 2C
%       Potential diagram
global w
global g
global a
w = 3;
g = 10;
a = 4;
x0 =(w*w/g/a)^(1/(a-2))
u0 = a * x0^(a-1);

global x_all
x_all = linspace(0*x0,2*x0,100);
global Vx_all
Vx_all = [];

for i = 1:length(x_all)
    Vx_all = [Vx_all, pot(x0,x_all(i))];
end
figure
plot(x_all,Vx_all)
xlabel('x (m)')
ylabel('Potential (m^2/s^2)')
title('Potential diagram vs. x position')


w0 = (w*w*(a-2)/(1+u0*u0))^0.5
T0 = 2*pi/w0

global eps0
global sign


eps0s = [0.01,0.1,0.3,0.4];

figure
hold on
for i=1:length(eps0s)
    eps0 = eps0s(i);
    sign = 1;
    [ts,xs] = RK4_solution(@v_well,[0,[x0]],1e-2,1000);
    x_max = max(xs);
    x_avg = mean(xs);
    plot(ts,(xs-x_avg)/max(xs-x_avg))
end
xlabel('time (sec)')
ylabel('x (m)')
title('Oscillation of x versus time')
hold off


function vz = velocity(t,g,w,alpha,theta_m)
    vz = g * alpha * t + g * (theta_m^2 / (2*w) * sin(2*w*t) + theta_m * alpha / w * (cos(2*w*t) - 1) );
end

function f = force(x)
    global w
    global g
    global a

    u = a * x ^(a-1);
    f = (w^2 *x - u * g )/(1 + u*u);
end


function V = pot(x0,x)

    t = linspace(x0,x,300);
    sum = 0;
    dt = t(2)-t(1);
    for i= 1:length(t)
        sum = sum - force(t(i)) * dt;
    end
    V = sum;
end

function V = pot_int(x0,x)

end

function v = v_well(t,x)
    global eps0
    global x_all
    global Vx_all
    global sign
    pot_x = interp1(x_all,Vx_all,x);
    if (eps0-pot_x)<0
        sign = -sign;
    end
    v = (2*abs(eps0-pot_x))^0.5*sign;
end

function [xs,ys] = RK4_solution(func,boundary,dx,N)
    x0 = boundary(1); y0 = transpose(boundary(2:length(boundary)));
    x = x0;
    y = y0;
    xs = [x0];
    ys = [y0];
    
    for i=1:N
        k1 = func(x,y);
        k2 = func(x + dx/2.,y + dx/2.*k1);
        k3 = func(x + dx/2.,y + dx/2.*k2);
        k4 = func(x + dx,y + dx*k3);
        y = y + dx/6.*(k1 + k2 * 2 + k3 * 2 + k4);
        x = x + dx;
        xs = [xs,x];
        ys = [ys,y];
    end
end