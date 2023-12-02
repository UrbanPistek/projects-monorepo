% Question 4
clc
clear
close all

% ts = 0.25;
ts = 0.4;
rm = -4/ts;

os = 5;
theta_min = atand((1/pi)*(log(os/100)));
y = rm/(tand(theta_min));

ess = 0;

% For plant 
k1 = 1.93;
tau = 0.019;
inv_tau = 1/tau;
k_tau = k1/tau;

syms s
delta_ds = (s+16+0.00001i)*(s+16-0.00001i)*(s+inv_tau);

f = expand(delta_ds);
disp(delta_ds);
disp(f);

c = single(coeffs(f));
c = fliplr(c); % flip order
disp(c);

% ci = [c 0]; % if ess
ci = [c];
disp(ci);

b = ci(:);
disp(b);

A = [
    1 0 0 0; 
    inv_tau 1 0 0; 
    k_tau inv_tau k1 0; 
    0 k_tau 0 k1;
    ];
disp(A);

% x = [g1; g0; f1; f0];
x = linsolve(A, b);

format long
disp("Results: ")
disp(x);

g1 = x(1);
g0 = x(2);
f1 = x(3);
f0 = x(4);

% Define Plant TF
disp("Plant: ")
np = [0 k_tau];
dp = [1 inv_tau 0];
P = tf(np, dp); 
printsys(np, dp);
fprintf("\n\n");

% Define Controller TF
disp("Controller C(s): ")
nc = [f1 f0];
dc = [g1 g0];
C = tf(nc, dc); 
printsys(nc, dc);
fprintf("\n\n");

% % Discretize the TF
disp("Controller D[z]: ")
Ts = 0.001;
D = c2d(C, Ts, 'foh'); % foh = trapezoidal method

format long
[D_num,D_den] = tfdata(D,'v');
disp("Discrete Controller C(s): ")
printsys(D_num, D_den);
fprintf("\n\n");
