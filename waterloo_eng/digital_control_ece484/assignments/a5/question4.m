% Question 4
clc
clear
close all

ts = 2;
rm = -4/ts;

os = 10;
theta_min = atand((1/pi)*(log(os/100)));
y = rm/(tand(theta_min));

ess = 0.02;
sigma = 1/ess - 1;

syms s
delta_ds = (s+2+2i)*(s+2-2i)*(s+8)*(s+9);

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
    6 1 0 0 0 0;
    9 6 1 1 0 0;
    0 9 6 0 1 0;
    0 0 9 0 0 1;
    0 0 -sigma*9 0 0 1;
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
np = [0 1];
dp = [1 6 9];
P = tf(np, dp); 
printsys(np, dp);
fprintf("\n\n");

% Define Controller TF
nc = [f2 f1 f0];
dc = [g2 g1 g0];
C = tf(nc, dc); 
printsys(nc, dc);
fprintf("\n\n");

% Discretize the TF
% Ts = 0.01;
Ts = 0.05;
D = c2d(C, Ts, 'foh'); % foh = trapezoidal method

figure(1);
step(C,'-',D,'--')

% simulate CT system 
tfinal = 3;
sim("ct_system.slx");

figure(2);
grid on
plot(t_sim_ct, y_sim_ct);
hold on
plot(t_sim_ct, r_sim_ct);
title('Response');
xlabel("Time (s)");
ylabel("Response (y(t))");
legend(["y(t)", "r(t)"], Location="northeast");

% simulate 
sim("dt_system.slx");

figure(3);
grid on
plot(t_sim_ct, y_sim_ct);
hold on
plot(t_sim_ct, r_sim_ct);
hold on
plot(t_sim_dt, y_sim_dt);

title('Response');
xlabel("Time (s)");
ylabel("Response (y(t))");
legend(["y(t)", "r(t)", "y[z]"], Location="northeast");

