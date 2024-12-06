syms x y u v a b c d e f h k l o p q r s w gamma eta nu omega g m_th n_th A B H U;
P(x,y,u,v) =  exp(a + b * x + c * y + d * u + e * v + f * x^2 + h * y^2 + k * u^2 + l * v^2 + o * x * y + p * u * v + q * x * u + r * y * u + s * x * v + w * y * v);
LOGP(x,y,u,v) = a + b * x + c * y + d * u + e * v + f * x^2 + h * y^2 + k * u^2 + l * v^2 + o * x * y + p * u * v + q * x * u + r * y * u + s * x * v + w * y * v;
assume (f<0)
assume (h<0)
assume (k<0)
assume (l<0)

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

function FUNC_FIL = filter(FUNC,var)
    Px = fst_deriv(FUNC,var);
    Pxx = fst_deriv(Px,var);
    FUNC_FIL = simplify(FUNC - Px * var + Pxx * var^2 / 2);
end

P_fil_u = filter(LOGP,u);
P_fil_uv = filter(P_fil_u,v)
P_uv = LOGP - P_fil_uv;

P_uv(u,v) =  exp (k * u^2 + l * v^2 + A * u + B * v + p * u * v);
Final = int(P_uv,u,-inf,inf);
Final(v) = (pi/(-k))^(1/2) * exp(H*v^2 + U*v - A^2/(4*k));
assume(H<0)
Final = int(Final,-inf,inf)