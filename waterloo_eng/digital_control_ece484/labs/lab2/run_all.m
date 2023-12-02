% run ct and dt simulations
clc
clear
close all

% Run question
% pole_place;
pole_zero;

% constants
tfinal = 1;

% Simulate systems
sim("ct_system.slx");
sim("dt_system.slx");

figure(1);
grid on
plot(t_sim_ct, y_sim_ct);
hold on
plot(t_sim_ct, r_sim_ct);
hold on
plot(t_sim_dt, y_sim_dt);

title('Response');
xlabel("Time (s)");
ylabel("Response (y(t))");
legend(["y(t)", "r(t)", "y[k]"]);

figure(2);
grid on
plot(t_sim_ct, r_sim_ct);
hold on
plot(t_sim_ct, u_sim_ct);

title('Control C(t)');
xlabel("Time (s)");
ylabel("Response (y(t))");
legend(["r(t)", "u(t)"]);

figure(3);
grid on
plot(t_sim_dt, r_sim_dt);
hold on
plot(t_sim_dt, u_sim_dt);

title('Control D[z]');
xlabel("Time (s)");
ylabel("Response (y(t))");
legend(["r(t)", "u[k]"]);
