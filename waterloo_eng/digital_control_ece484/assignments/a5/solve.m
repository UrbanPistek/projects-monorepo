% solve system of odes
clc
clear
close all

% Solve system of differential equations using euler integration
[t,y] = ode45(@vdp1,[0 25],[pi/2; 0]);

plot(t,y(:,1),'-x',t,y(:,2),'-o')
title('Solution');
xlabel('Time t');
ylabel('Solution x(t)');
legend('x1','x2')

function dydt = vdp1(t,x) %#ok<INUSL> 
g = 9.8;
l = g/2; 
m = 2/g; 
D = 0.1;
u = 0.1;

dydt = [x(2); (-g/l)*sin(x(1)) - (D/(m*l))*x(2) + (1/(m*l))*u];
end

