% run
clc
clear
close all;

% Define TF
calculations;

tfinal = 5;

% Full TF
n = [(kp*k1)/tau]; %#ok<NBRAK2> 
d = [1 1/tau (kp*k1)/tau];
G = tf(n, d); 
printsys(n, d);
fprintf("\n\n");

% C(s)
fprintf("\n C(s) \n");
nc = kp; 
dc = 1;
C = tf(nc, dc);
printsys(nc, dc);

% P(s)
fprintf("\n P(s) \n");
np = k1; 
dp = [tau 1 0];
P = tf(np, dp);
printsys(np, dp);

% Simulate system
sim("ct_system.slx");

plot_ct_system;