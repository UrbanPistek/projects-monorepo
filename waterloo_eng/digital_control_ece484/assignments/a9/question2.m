% question 2
clc
clear
close all

% Desired specs
ts = 2;
rm = -4/ts;
os = 20;
theta_min = atand((1/pi)*(log(os/100)));
y = rm/(tand(theta_min));
ess = 0.02;
delta = 1/ess - 1; 

% Define Continous Plant TF
disp("P(s): ")
np = [1 -1];
dp = [1 5 6];
P = tf(np, dp); 
printsys(np, dp);
fprintf("\n\n");

% Get bandwidth of the plant
bw_plant = bandwidth(P);
Ts = 0.3/10; % based on fastest pole
% Ts = floor(bw_plant*10)/100; % round to nearest 10th

% Define discretized plant G[z]
syms z
G = -1*(((exp(Ts) - 1)*(exp(Ts) + 1))*((-2*exp(2*Ts))*z + exp(4*Ts)*z + 2*exp(2*Ts) - 1)) / (2*(((exp(2*Ts))*z - 1)*((exp(4*Ts))*z - 1)));
G1 = double(subs(G, 1)); % Evaluate G1

% get num, and dem
[gnum, gdem] = numden(G); % extract num, dem

% extract coefficients
a = single(coeffs(expand(gdem)));
b = single(coeffs(expand(gnum)));
a = fliplr(a);
b = fliplr(b);
b = b./a(1);
a = a./a(1);

disp("G[z]:")
G = tf(b, a); 
printsys(b, a);
fprintf("\n\n");

% Controller design
syms z
ddes = (z - exp(-3*Ts))*(z - exp(-4*Ts))*(z - exp((-2 + 3i)*Ts))*(z - exp((-2 - 3i)*Ts));

f = expand(ddes);
c = single(coeffs(f));
c = fliplr(c); % flip order
ci = [c 0];
bv = ci(:);

A = [
    a(1) 0 0 0 0 0;
    a(2) a(1) 0 b(1) 0 0;
    a(3) a(2) a(1) b(2) b(1) 0;
    0 a(3) a(2) 0 b(2) b(1); 
    0 0 a(3) 0 0 b(2); 
    -delta -delta -delta G1 G1 G1; 
    ];
disp(A);

% x = [g2; g1; g0; f2; f1; f0];
x = linsolve(A, bv);

disp(x);

g2 = x(1); 
g1 = x(2);
g0 = x(3);
f2 = x(4);
f1 = x(5);
f0 = x(6);

disp("D[z]:")
nd = [f2 f1 f0];
dd = [g2 g1 g0];
D = tf(nd, dd); 
printsys(nd, dd);
fprintf("\n\n");

% Setup transfer functions in simulink
G_num = b;
G_dem = a; 
D_num = nd;
D_dem = dd;

tfinal = 2;
step_amp = 1;
sim('dt_system.slx');

% figure(1);
figure(1);
grid on
time_length = length(y_sim_dt);
t_sim = linspace(0,tfinal,time_length); % Time
plot(t_sim, y_sim_dt);
hold on
plot(t_sim, r_sim_dt(1:time_length));
title('Controller Step Response');
xlabel("Time (s)");
ylabel("Response");
legend(["y(t)", "r(t)"])

% validate specs
check_specs;
