% Lean Controller Design
%#ok<*NASGU>
clc
clear
close all
format long

% Desired specs
ts = 7;
rm = -4/ts;
os = 45;
theta_min = atand((1/pi)*(log(os/100)));
y = rm/(tand(theta_min));
ess = 0;

% Load Controllers
controllers;

% Simulation parameters
r_amp = 0.05;
r_dc_offset = 0.2; 

% Define closed loop inner tf 
G = (P*C)/(1 + P*C);

% constants
tfinal = 25;

% sqaure wave input frequency
input_freq = 0.071; % period = 7s

% Simulate systems
sim("ct_system.slx");

figure(1);
grid on
plot(t_sim_ct, r_sim_ct);
hold on
plot(t_sim_ct, y_sim_ct);
% hold on
% plot(t_sim_ct, u_sim_ct);

title('Continous Time with C_2_L_D');
xlabel("Time (s)");
ylabel("Response");
% legend(["r(t)", "y(t)", "u(t)"]);
legend(["r(t)", "y(t)"]);

% Perform pole placement desgin
pole_equate;

% Run sim using full simulation
L = C2;
input_freq = (1/28); % period = 7s
r_amp = 0.015; 
r_dc_offset = 0.165; 

% Simulate systems
sim("ct_system.slx");

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

% figure(3);
% grid on
% plot(t_sim_ct, r_sim_ct);
% hold on
% plot(t_sim_ct, y_sim_ct);
% 
% title('Continous Time with Designed C_2');
% xlabel("Time (s)");
% ylabel("Response");
% legend(["r(t)", "y(t)"]);
figure(3);
grid on
plot(t_sim_ct, r_sim_ct);
hold on
plot(t_sim_ct, theta_ref_sim_ct);

title('Designed C_2 Theta_r_e_f');
xlabel("Time (s)");
ylabel("Response");
legend(["r(t)", "theta_r_e_f(t)"]);

% Look at step response
input_freq = (1/64); % period = 28s
r_amp = -0.015;
r_dc_offset = 0;

% Simulate systems
sim("ct_system.slx");

figure(4);
grid on
plot(t_sim_ct, r_sim_ct);
hold on
plot(t_sim_ct, y_sim_ct);

title('Designed C_2 Step Response');
xlabel("Time (s)");
ylabel("Response (y(t))");
legend(["r(t)", "y(t)"]);

disp("Lead Controller")
% estimate specs
% Estimate settling time, percent overshoot, and time-to-peak
peak = max(y_sim_ct);
peak_idx = find(y_sim_ct==peak);
tp = t_sim_ct(peak_idx);
yss = mean(y_sim_ct(12000:end));
os = (peak - yss)/peak;
per_os = os*100;

ts = 1;
for i = peak_idx:length(y_sim_ct)
    if (y_sim_ct(i) < (yss + yss*0.02)) && (y_sim_ct(i) > (yss - yss*0.02))

        % verify this does not osillate further
        if y_sim_ct(i) < max(y_sim_ct(i+1:end))

            ts = t_sim_ct(i);
            fprintf('ts=%f, i=%d, val=%f \n', ts, i, y_sim_ct(i));
            break
        end
        
    end
end

fprintf('tp =%f \n', tp);
fprintf('peak =%f \n', peak);
fprintf('yss =%f \n', yss);
fprintf('os =%f \n', os);
fprintf('percent os =%f \n', per_os);
fprintf('ts =%f \n', ts);

% check specs of the discretized controller
DL = D2;

sim("sample_data_system.slx");

figure(5);
grid on
plot(t_sim_dt, r_sim_dt);
hold on
plot(t_sim_dt, y_sim_dt);

title('D_2 Step Response');
xlabel("Time (s)");
ylabel("Response");
legend(["r(t)", "y(t)"]);

figure(6);
grid on
plot(t_sim_dt, r_sim_dt);
hold on
plot(t_sim_dt, u_sim_dt);

title('D_2 Step Response');
xlabel("Time (s)");
ylabel("Response");
legend(["r(t)", "u(t)"]);

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

% Look at step response
input_freq = (1/28); % period = 7s
r_amp = -0.075;
r_dc_offset = 0.175;

% Simulate systems
sim("sample_data_system.slx");

% figure(7);
% grid on
% plot(t_sim_dt, r_sim_dt);
% hold on
% plot(t_sim_dt, y_sim_dt);
% 
% title('D_2 Step Response with Larger Input');
% xlabel("Time (s)");
% ylabel("Response");
% legend(["r(t)", "y(t)"]);
figure(7);
grid on
plot(t_sim_dt, r_sim_dt);
hold on
plot(t_sim_dt, theta_ref_sim_dt);
hold on
plot(t_sim_dt, theta_ref_sat_sim_dt);

title('D_2 Step Theta_r_e_f with Larger Input');
xlabel("Time (s)");
ylabel("Response");
legend(["r(t)", "theta_r_e_f(t)", "theta_s_a_t(t)"]);

% Simulation parameters
tfinal = 25;
r_amp = 0.05;
r_dc_offset = 0.2; 
input_freq = 0.071; % period = 7s

DL = D2L;

% Simulate systems
sim("sample_data_system.slx");

figure(8);
grid on
plot(t_sim_dt, r_sim_dt);
hold on
plot(t_sim_dt, y_sim_dt);

title('Discretized C_2_L_D Controller');
xlabel("Time (s)");
ylabel("Response");
legend(["r(t)", "y(t)"]);

% COMMENTED OUT AS LIKELY NOT NEEDED 
%
% % anti-stiction simulation
% tfinal = 25;
% stiction_constant = 0.69;
% sim("anti_stiction.slx");
% 
% figure(8);
% grid on
% plot(t_sim_dt, r_sim_dt);
% hold on
% plot(t_sim_dt, y_sim_dt);
% 
% title('Anti Stiction Response');
% xlabel("Time (s)");
% ylabel("Response (y(t))");
% legend(["r(t)", "y(t)"]);
% 
% figure(9);
% grid on
% plot(t_sim_dt, r_sim_dt);
% hold on
% plot(t_sim_dt, u_sim_dt);
% hold on
% plot(t_sim_dt, e_sim_dt);
% 
% title('Anti Stiction Control');
% xlabel("Time (s)");
% ylabel("Response (y(t))");
% legend(["r(t)", "u(t)", "e(t)"]);
% 
% % test stiction step response
% input_freq = (1/80); % period = 7s
% sim("anti_stiction.slx");
% 
% figure(10);
% grid on
% plot(t_sim_dt, r_sim_dt);
% hold on
% plot(t_sim_dt, y_sim_dt);
% 
% title('Anti Stiction Step Response');
% xlabel("Time (s)");
% ylabel("Response (y(t))");
% legend(["r(t)", "y(t)"]);
% 
% disp("----");
% disp("Stiction Transient Specs");
% % estimate specs
% % Estimate settling time, percent overshoot, and time-to-peak
% peak_dt = max(y_sim_dt);
% peak_idx_dt = find(y_sim_dt==peak_dt);
% tp_dt = t_sim_dt(peak_idx_dt);
% yss_dt = mean(y_sim_dt(110000:end));
% os_dt = (peak_dt - yss_dt)/peak_dt;
% per_os_dt = os_dt*100;
% 
% ts_dt = 1;
% for i = peak_idx_dt:length(y_sim_dt)
%     if (y_sim_dt(i) < (yss_dt + yss_dt*0.02)) && (y_sim_dt(i) > (yss_dt - yss_dt*0.02))
% 
%         % verify this does not osillate further
%         if y_sim_dt(i) < max(y_sim_dt(i+1:end))
% 
%             ts_dt = t_sim_dt(i);
%             fprintf('ts=%f, i=%d, val=%f \n', ts_dt, i, y_sim_dt(i));
%             break
%         end
%         
%     end
% end
% 
% fprintf('tp =%f \n', tp_dt);
% fprintf('peak =%f \n', peak_dt);
% fprintf('yss =%f \n', yss_dt);
% fprintf('os =%f \n', os_dt);
% fprintf('percent os =%f \n', per_os_dt);
% fprintf('ts =%f \n', ts_dt);

