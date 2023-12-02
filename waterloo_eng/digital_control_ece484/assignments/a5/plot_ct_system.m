% plot ct

figure(2);
grid on
plot(t_sim, y_sim);
hold on
plot(t_sim, r_sim);
title('Response');
xlabel("Time (s)");
ylabel("Response (y(t))");

legend(["y(t)", "r(t)"], Location="northeast");