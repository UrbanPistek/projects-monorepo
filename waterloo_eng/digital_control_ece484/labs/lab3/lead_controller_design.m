% Lean Controller Design
% ======================================
% THIS IS THE MAIN FILE
% RUN THIS TO PERFORM ALL SIMULATIONS
% AND CALCULATIONS
% ======================================
%#ok<*NASGU>
clc
clear
close all
format long

% ======================================
% PARAMETERS, VALUES AND CONTROLLERS
% ======================================

% Desired specs
ts = 7;
rm = -4/ts;
os = 45;
theta_min = atand((1/pi)*(log(os/100)));
y = rm/(tand(theta_min));
ess = 0;

% Load Controllers
controllers;

% ======================================
% PART A
% ======================================

% Simulation parameters
r_amp = 0.05;
r_dc_offset = 0.2; 
tfinal = 25;
input_freq = 0.071; % period = 7s

% Simulate systems
sim("ct_full_system.slx");

figure(1);
grid on
plot(t_sim_ct, r_sim_ct);
hold on
plot(t_sim_ct, y_sim_ct);
hold on
plot(t_sim_ct, u_sim_ct);

title('Continous Time with C_2_L_D');
xlabel("Time (s)");
ylabel("Response");
legend(["r(t)", "y(t)", "u(t)"]);

% ======================================
% PART C
% ======================================

% Perform pole placement desgin
pole_equate;

% Run sim using full simulation
L = C2;
input_freq = (1/14); % period = 14s
r_amp = 0.015; 
r_dc_offset = 0.165; 

% Simulate systems
sim("ct_full_system.slx");

figure(2);
grid on
plot(t_sim_ct, r_sim_ct);
hold on
plot(t_sim_ct, y_sim_ct);
hold on
plot(t_sim_ct, u_sim_ct);

title('Continous Time with Designed C_2');
xlabel("Time (s)");
ylabel("Response");
legend(["r(t)", "y(t)", "u(t)"]);

figure(3);
grid on
plot(t_sim_ct, r_sim_ct);
hold on
plot(t_sim_ct, theta_ref_sim_ct);

title('Designed C_2 Theta_r_e_f');
xlabel("Time (s)");
ylabel("Response");
legend(["r(t)", "theta_r_e_f(t)"]);

% check specs of the discretized controller
DL = D2;

input_freq = (1/64); % Look at step response
sim("dt_full_system.slx");

figure(4);
grid on
plot(t_sim_dt, r_sim_dt);
hold on
plot(t_sim_dt, y_sim_dt);

title('D_2 Step Response');
xlabel("Time (s)");
ylabel("Response");
legend(["r(t)", "y(t)"]);

figure(5);
grid on
plot(t_sim_dt, r_sim_dt);
hold on
plot(t_sim_dt, u_sim_dt);

title('D_2 Step Response');
xlabel("Time (s)");
ylabel("Response");
legend(["r(t)", "u(t)"]);

% Verify transient specs on discretized controller
disp("----");
disp("Discretized Lead Controller");
% estimate specs
% Estimate settling time, percent overshoot, and time-to-peak
peak_dt = max(y_sim_dt);
peak_idx_dt = find(y_sim_dt==peak_dt);
tp_dt = t_sim_dt(peak_idx_dt);
yss_dt = mean(y_sim_dt(110000:end));
os_dt = (peak_dt - yss_dt)/peak_dt;
per_os_dt = os_dt*100;

ts_dt = 1;
for i = peak_idx_dt:length(y_sim_dt)
    if (y_sim_dt(i) < (yss_dt + yss_dt*0.02)) && (y_sim_dt(i) > (yss_dt - yss_dt*0.02))

        % verify this does not osillate further
        if y_sim_dt(i) <= max(y_sim_dt(i+1:end))

            ts_dt = t_sim_dt(i);
            fprintf('ts=%f, i=%d, val=%f \n', ts_dt, i, y_sim_dt(i));
            break
        end
        
    end
end

fprintf('tp =%f \n', tp_dt);
fprintf('peak =%f \n', peak_dt);
fprintf('yss =%f \n', yss_dt);
fprintf('os =%f \n', os_dt);
fprintf('percent os =%f \n', per_os_dt);
fprintf('ts =%f \n', ts_dt);