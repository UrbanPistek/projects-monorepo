% Question 1
clc
clear
close all

ts = 4;
rm = -4/ts;

os = 50;
theta_min = atand((1/pi)*(log(os/100)));
y = 1/(tand(theta_min));

ess = 0;

syms s
% delta_ds = (s+1.1+1i)*(s+1.1-1i);
delta_ds = (s+2+0.1i)*(s+2-0.1i);

f = expand(delta_ds);
disp(delta_ds);
disp(f);

c = single(coeffs(f));
c = fliplr(c); % flip order
disp(c);

ci = [c 0];
disp(ci);

b = ci(:);
disp(b);

A = [
    1 0 0 0; 
    -1 1 1 0;
    0 -1 0 1; 
    0 -1 0 0;
    ];
disp(A);

% x = [g1; g0; f1; f0];
x = linsolve(A, b);

disp(x);

g1 = x(1); 
g0 = x(2);
f1 = x(3);
f0 = x(4);

% Define Plant TF
nc = [0 0 1];
dc = [0 1 -1];
C = tf(nc, dc); 
printsys(nc, dc);
fprintf("\n\n");

% Define Controller TF
np = [f1 f0];
dp = [g1 g0];
P = tf(np, dp); 
printsys(np, dp);
fprintf("\n\n");
