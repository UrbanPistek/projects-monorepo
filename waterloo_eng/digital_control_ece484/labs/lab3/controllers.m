% controllers
% ======================================
% THIS FILE DEFINES ALL THE CONTROLLERS
%
% USE "lead_controller_design.m" TO RUN ALL CODE
% ======================================

% Controller Parameters
k1 = 1.93;
tau = 0.019;
inv_tau = 1/tau;
k1_tau = k1/tau;
g1 = 1;
g0 = 290; 
k = (g0*tau)/k1; 
k_tau = k/tau;
k2 = 0.0609;
k3 = -4.36;

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

% Discretize the TF
disp("Discrete Controller C(s): ")
Ts = 0.001;
D = c2d(C, Ts, 'foh'); % foh = trapezoidal method
[D_num,D_den] = tfdata(D,'v');
printsys(D_num, D_den);
fprintf("\n\n");

% Lead Controller - C2LD
disp("Lead Controller L(s): ")
nl = [-7 -7*0.35];
dl = [1 2.5];
L = tf(nl, dl); 
printsys(nl, dl);
fprintf("\n\n");

% Discretized Lead Controller
disp("Discrete Lead Controller L(s): ")
Ts = 0.001;
DL = c2d(L, Ts, 'foh'); % foh = trapezoidal method
D2L = DL;
[DL_num, DL_den] = tfdata(DL,'v');
printsys(DL_num, DL_den);
fprintf("\n\n");

% define k3/s^2 as a tf
phi_n = [0 k3];
phi_d = [1 0 0];
PHI = tf(phi_n, phi_d);

% Inner loop plant model 
FP = ((P*C)/(1 + P*C))*(k2)*(PHI);
[FP_num, FP_den] = tfdata(FP,'v');
printsys(FP_num, FP_den);
fprintf("\n\n");

system = zpk(FP);

