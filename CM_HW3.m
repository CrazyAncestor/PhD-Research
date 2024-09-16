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
plot(t/T,vz);


function vz = velocity(t,g,w,alpha,theta_m)
    vz = g * alpha * t + g * (theta_m^2 / (2*w) * sin(2*w*t) + theta_m * alpha / w * (cos(2*w*t) - 1) );
end