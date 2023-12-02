% plot ct

fig = figure(1);
plot(t_sim, y_sim);
title('Saturation Response, kp=3.2');
xlabel("Time (s)");
ylabel("Response (y(t))");
hold on
plot(t_sim, r_sim);

hold on
grid on
plot(t_sim, r_sat_sim);
legend(["y(t)", "r(t)", "r_s_a_t(t)"]);
axis([0 5 0 1.1]);
saveas(fig,'sat_resp.png');