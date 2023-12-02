% run

% plot y(t)
vals = [0.01 0.1 2.5];
load_sim_var;

figure(1);
sim("./ct_data_system.slx");
plot(t_sim, r_sim);
hold on 
plot(t_sim, y_sim);
hold on

for ts = vals

   disp(ts)

    T = ts;
    load_sim_var;

    sim("./sampled_data_system.slx");
    plot(t_sim_d, y_sim_d);

    hold on
end

title("System Output y(t)");
xlabel("t, (s)");
ylabel("output y(t)");
legend(["r(t)", "Continuous y(t)", "Discrete y(t), T = 0.01", "Discrete y(t), T = 0.1", "Discrete y(t), T = 2.5"], 'Location','northwest');

% plot (u(t)
figure(2);
load_sim_var;

sim("./ct_data_system.slx");
plot(t_sim, r_sim);
hold on 
plot(t_sim, u_sim);
hold on

for ts = vals

   disp(ts)

    T = ts;
    load_sim_var;

    sim("./sampled_data_system.slx");
    plot(t_sim_d, u_sim_d);

    hold on
end

title("System Control Signal u(t)");
xlabel("t, (s)");
ylabel("control u(t)");
legend(["r(t)", "Continuous u(t)", "Discrete u(t), T = 0.01", "Discrete u(t), T = 0.1", "Discrete u(t), T = 2.5"], 'Location','northwest');