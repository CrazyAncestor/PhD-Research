%Random function for example
syms x y u v a b c d e f h k l o p q r s w gamma eta nu omega g m n;
P(x,y,u,v) =  exp(a + b * x + c * y + d * u + e * v + f * x^2 + h * y^2 + k * u^2 + l * v^2 + o * x * y + p * u * v + q * x * u + r * y * u + s * x * v + w * y * v);

Px = diff(P,x);
Py = diff(P,y);
Pu = diff(P,u);
Pv = diff(P,v);

Pxx = diff(Px,x);
Pyy = diff(Py,y);
Puu = diff(Pu,u);
Pvv = diff(Pv,v);

Px = simplify(Px/P);
Py = simplify(Py/P);
Pu = simplify(Pu/P);
Pv = simplify(Pv/P);

Pxx = simplify(Pxx/P);
Pyy = simplify(Pyy/P);
Puu = simplify(Puu/P);
Pvv = simplify(Pvv/P);

PT1 = (gamma + eta) + (gamma/2 * x - nu * y - g * v) * Px + (gamma/2 * y + nu * x + g * u) * Py;
PT2 = (eta/2 * u - nu * v - g * y) * Pu + (eta/2 * v + nu * u + g * x) * Pv;
PT3 = gamma * n / 4 * (Pxx + Pyy) + eta * m / 4 * (Puu + Pvv);

PT = simplify(PT1 + PT2 + PT3);
PTx = simplify(diff(PT,x));
PTy = simplify(diff(PT,y));
PTu = simplify(diff(PT,u));
PTv = simplify(diff(PT,v));

PTxx = simplify(diff(PTx,x));
PTyy = simplify(diff(PTy,y));
PTuu = simplify(diff(PTu,u));
PTvv = simplify(diff(PTv,v));

PTxu = simplify(diff(PTx,u));
PTyu = simplify(diff(PTy,u));
PTxv = simplify(diff(PTx,v));
PTyv = simplify(diff(PTy,v));
PTxy = simplify(diff(PTx,y));
PTuv = simplify(diff(PTu,v));

PTx = simplify(PTx - (PTxx * x + PTxu * u + PTxv * v + PTxy * y))
PTy = simplify(PTy - (PTyy * y + PTyu * u + PTyv * v + PTxy * x))

PTu = simplify(PTu - (PTuu * u + PTxu * x + PTuv * v + PTyu * y))
PTv = simplify(PTv - (PTvv * v + PTuv * u + PTyv * y + PTxv * x))

PT_const = PT - ((PTxx * x^2 + PTyy * y^2 + PTuu * u^2 + PTvv * v^2)/2 + (PTxu * x * u + PTyu * y * u + PTxv * x * v + PTyv * y * v + PTxy * x * y + PTuv * u * v) + (PTx * x + PTy * y + PTu * u + PTv * v));
PT_const = simplify(PT_const)