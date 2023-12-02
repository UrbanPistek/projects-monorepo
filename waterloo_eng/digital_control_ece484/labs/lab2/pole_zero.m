% zero pole cancellation

clc
clear
close all

ts = 0.25;
% ts = 0.4;
rm = -4/ts;

os = 5;
theta_min = atand((1/pi)*(log(os/100)));
y = rm/(tand(theta_min));

ess = 0;

% For plant 
% From poles
g1 = 1;
g0 = 290; 

k1 = 1.93;
tau = 0.019;
inv_tau = 1/tau;
k1_tau = k1/tau;

% for controller
k = (g0*tau)/k1; 
k_tau = k/tau;

% Define Plant TF
disp("Plant: ")
np = [0 k1_tau];
dp = [1 inv_tau 0];
P = tf(np, dp); 
printsys(np, dp);
fprintf("\n\n");

% Define Controller TF
disp("Controller C(s): ")
nc = [k k_tau];
dc = [1 32];
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
