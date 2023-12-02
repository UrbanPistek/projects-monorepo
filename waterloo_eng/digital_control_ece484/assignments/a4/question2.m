% question 2
clc
clear
close all

ts = 5;
rm = -4/ts;

os = 10;
theta_min = atand((1/pi)*(log(os/100)));
y = 1/(tand(theta_min));

ess = 0.02;
sigma = 1/ess - 1;

syms s
% delta_ds = (s+0.8+1i)*(s+0.8-1i)*(s+1)*(s+2);
% delta_ds = (s+1+0.1i)*(s+1-0.1i)*(s+2)*(s+3);
delta_ds = (s+1+0.0001i)*(s+1-0.0001i)*(s+8)*(s+10);

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
    1 0 0 0 0 0;
    20 1 0 0 0 0;
    100 20 1 1 0 0;
    0 100 20 0 1 0;
    0 0 100 0 0 1;
    0 0 sigma*100 0 0 1;
    ];
disp(A);

% x = [g2; g1; g0; f2; f1; f0];
x = linsolve(A, b);

format long
disp(x);

g2 = x(1); 
g1 = x(2);
g0 = x(3);
f2 = x(4);
f1 = x(5);
f0 = x(6);

% Define Plant TF
nc = [0 0 1];
dc = [1 20 100];
C = tf(nc, dc); 
printsys(nc, dc);
fprintf("\n\n");

% Define Controller TF
np = [f2 f1 f0];
dp = [g2 g1 g0];
P = tf(np, dp); 
printsys(np, dp);
fprintf("\n\n");

