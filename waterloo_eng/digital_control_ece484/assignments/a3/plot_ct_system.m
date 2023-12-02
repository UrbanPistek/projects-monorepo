% plot ct

figure(1);
grid on
plot(t_sim, y_sim);
title(sprintf('Response k=%d', k));
xlabel("Time (s)");
ylabel("Response (y(t))");
hold on