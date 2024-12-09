syms x y u v a b c d e f h k l o p q r s w gamma eta nu omega g m_th n_th Dia;
P(x,y,u,v) =  exp(a + b * x + c * y + d * u + e * v + f * x^2 + h * y^2 + k * u^2 + l * v^2 + o * x * y + p * u * v + q * x * u + r * y * u + s * x * v + w * y * v);

function fx = fst_deriv(func,var)
    fx = simplify(diff(func,var));
end

function fxx = snd_deriv(func,var)
    fxx = simplify(diff(diff(func,var),var));
end

function fxy = diff_deriv(func,var1,var2)
    fxy = simplify(diff(diff(func,var1),var2));
end

function fx = exp_fst_deriv(func,var)
    fx = simplify(diff(func,var)/func);
end

function fxx = exp_snd_deriv(func,var)
    fxx = simplify(diff(diff(func,var),var)/func);
end

function fxy = exp_diff_deriv(func,var1,var2)
    fxy = simplify(diff(diff(func,var1),var2)/func);
end

%   Calculate 1st and 2nd derivatives
function [Px,Py,Pu,Pv,Pxx,Pyy,Puu,Pvv,Pxv,Pyu,Pxy] = fst_snd_derivs(P,x,y,u,v)
    Px = exp_fst_deriv(P,x);
    Py = exp_fst_deriv(P,y);
    Pu = exp_fst_deriv(P,u);
    Pv = exp_fst_deriv(P,v);
    
    Pxx = exp_snd_deriv(P,x);
    Pyy = exp_snd_deriv(P,y);
    Puu = exp_snd_deriv(P,u);
    Pvv = exp_snd_deriv(P,v);

    Pxv = exp_diff_deriv(P,x,v);
    Pyu = exp_diff_deriv(P,y,u);
    Pxy = exp_diff_deriv(P,x,y);
end

%   Extract each components in the time derivative
function [PT_const, PTx, PTy, PTu, PTv, PTxx, PTyy, PTuu, PTvv, PTxy, PTuv, PTxu, PTyu, PTxv, PTyv] = extract_coef(PT,x,y,u,v) 
    PTxx = snd_deriv(PT,x);
    PTyy = snd_deriv(PT,y);
    PTuu = snd_deriv(PT,u);
    PTvv = snd_deriv(PT,v);
    
    PTxy = diff_deriv(PT,x,y);
    PTuv = diff_deriv(PT,u,v);
    PTxu = diff_deriv(PT,x,u);
    PTyu = diff_deriv(PT,y,u);
    PTxv = diff_deriv(PT,x,v);
    PTyv = diff_deriv(PT,y,v);
    
    PTx = simplify(fst_deriv(PT,x) - (PTxx * x + PTxu * u + PTxv * v + PTxy * y));
    PTy = simplify(fst_deriv(PT,y) - (PTyy * y + PTyu * u + PTyv * v + PTxy * x));
    
    PTu = simplify(fst_deriv(PT,u) - (PTuu * u + PTxu * x + PTuv * v + PTyu * y));
    PTv = simplify(fst_deriv(PT,v) - (PTvv * v + PTuv * u + PTyv * y + PTxv * x));
    
    PT_const = PT - ((PTxx * x^2 + PTyy * y^2 + PTuu * u^2 + PTvv * v^2)/2 + (PTxu * x * u + PTyu * y * u + PTxv * x * v + PTyv * y * v + PTxy * x * y + PTuv * u * v) + (PTx * x + PTy * y + PTu * u + PTv * v));
    PT_const = simplify(PT_const);

    PTxx = PTxx/2;
    PTyy = PTyy/2;
    PTuu = PTuu/2;
    PTvv = PTvv/2;
end

%   Calculate the time-derivative of P of the system
[Px,Py,Pu,Pv,Pxx,Pyy,Puu,Pvv,Pxv,Pyu,Pxy] = fst_snd_derivs(P,x,y,u,v);

PT1 = (gamma + eta) + (gamma/2 * x - nu * y) * Px + (gamma/2 * y + (nu + 4 * Dia) * x + 2 * g * u) * Py;
PT2 = (eta/2 * u - (nu + 2 * Dia) * v) * Pu + (eta/2 * v + (nu + 2 * Dia) * u + 2 * g * x) * Pv;
PT3 = gamma * n_th / 4 * (Pxx + Pyy) + eta * m_th / 4 * (Puu + Pvv);
PT4 = - g * Pxv / 2 - g * Pyu / 2 - Dia * Pxy;

PT = simplify(PT1 + PT2 + PT3 + PT4);

[PT_const, PTx, PTy, PTu, PTv, PTxx, PTyy, PTuu, PTvv, PTxy, PTuv, PTxu, PTyu, PTxv, PTyv] = extract_coef(PT,x,y,u,v)
