% pole regulat design 
% ======================================
% THIS FILE DESIGNS OUR LEAD CONTROLLER
%
% USE "lead_controller_design.m" TO RUN ALL CODE
% ======================================

syms s
ddes = (s+0.6+0.01i)*(s+0.6-0.01i)*(s+0.7)*(s+0.8);

f = expand(ddes);
disp(ddes);
disp(f);
c = single(coeffs(f));
c = fliplr(c); % flip order
disp(c);

g2 = c(1);
g1 = c(2);
g0 = 0; 
f2 = (c(3) - g0)/(k2*k3);
f1 = (c(4))/(k2*k3);
f0 = (c(5))/(k2*k3);

disp("Designed Lead Controller C(s): ")
nc2 = [f2 f1 f0];
dc2 = [g2 g1 g0];
C2 = tf(nc2, dc2); 
printsys(nc2, dc2);
fprintf("\n\n");

% Discretized Lead Controller
disp("Discrete Lead Controller L(s): ")
Ts = 0.001;
D2 = c2d(C2, Ts, 'foh'); % foh = trapezoidal method
[D2_num, D2_den] = tfdata(D2,'v');
printsys(D2_num, D2_den);
fprintf("\n\n");