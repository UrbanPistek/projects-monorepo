% plot dt

figure(1);
plot(t_sim_d, r_sim_d);
hold on
plot(t_sim_d, y_sim_d);
hold on  
legend("r(t)", "y(t)");
