% Caculations
clc
clear

% 
kp = 3.2;
% os = (0.009 + 0.012)/2;
os = (0.009 + 0.012)/4; % scale by a factor of 0.5
tp = (0.859 + 0.723)/2; 

zeta = sqrt(((log(os))^2)/(pi^2 + (log(os))^2));
wn = (pi)/(tp*sqrt(1 - zeta^2));

tau = 1/(2*zeta*wn);
k1 = ((wn^2)*(tau))/(kp);
