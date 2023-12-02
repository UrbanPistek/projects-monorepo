% Question 3
clc
clear
close all; 

% Given parameters
r = 10;
l = 25; 

%30s 
time = linspace(0, 30, 2000); 

% Angles beta
B1 = pi/2;
B2 = (-5*pi)/6;
B3 = -pi/6;

% Set initial conditions
theta0 = [0;0;0];

% Set initial set of inputs
inputs = [r l B1 B2 B3];

% ---------------------------------------------------------
% First plot
ua = [-2;1;1];
[t, q] = ode45(@(t,q) createODE_constant_input(t, q, ua, inputs), time, theta0); 

% Results
x = q(:,1); 
y = q(:,2); 
theta = q(:,3); 

% Plot all values 
figure(1);
subplot(2,2,1)
plot(x,y)
grid on
title('Position')
xlabel('x (cm)') 
ylabel('y (cm)')
legend('U=[-2;1;1]')

subplot(2,2,2)
plot(t,x)
grid on
title('x vs t')
xlabel('t (s)') 
ylabel('x (cm)')

subplot(2,2,3)
plot(t,y)
grid on
title('y vs t')
xlabel('t (s)') 
ylabel('y (cm)')

subplot(2,2,4)
plot(t,theta)
grid on
title('theta vs t')
xlabel('t (s)') 
ylabel('theta (rad)')
axis([0 30 -2*pi 2*pi])

% ---------------------------------------------------------
% Second plot - Straight Line
ub = [1;-1;0];
[t, q] = ode45(@(t,q) createODE_constant_input(t, q, ub, inputs), time, theta0); 

% Results
x = q(:,1); 
y = q(:,2); 
theta = q(:,3); 

% Plot all values 
figure(2);
subplot(2,2,1)
plot(x,y)
grid on
title('Position')
xlabel('x (cm)') 
ylabel('y (cm)')
legend('U=[1;-1;0]')

subplot(2,2,2)
plot(t,x)
grid on
title('x vs t')
xlabel('t (s)') 
ylabel('x (cm)')

subplot(2,2,3)
plot(t,y)
grid on
title('y vs t')
xlabel('t (s)') 
ylabel('y (cm)')

subplot(2,2,4)
plot(t,theta)
grid on
title('theta vs t')
xlabel('t (s)') 
ylabel('theta (rad)')
axis([0 30 -2*pi 2*pi])

% ---------------------------------------------------------
% Third plot - 2m diameter circle
uc = [4;4;-4];
[t, q] = ode45(@(t,q) createODE_constant_input(t, q, uc, inputs), time, theta0); 

% Results
x = q(:,1); 
y = q(:,2); 
theta = q(:,3); 

% Plot all values 
figure(3);
subplot(2,2,1)
plot(x,y)
grid on
title('Position')
xlabel('x (cm)') 
ylabel('y (cm)')
legend('U=[4;4;-4]')

subplot(2,2,2)
plot(t,x)
grid on
title('x vs t')
xlabel('t (s)') 
ylabel('x (cm)')

subplot(2,2,3)
plot(t,y)
grid on
title('y vs t')
xlabel('t (s)') 
ylabel('y (cm)')

subplot(2,2,4)
plot(t,theta)
grid on
title('theta vs t')
xlabel('t (s)') 
ylabel('theta (rad)')

% ---------------------------------------------------------
% Fourth plot - Spiral
[t, q] = ode45(@(t,q) createODE_variable_input(t, q, inputs), time, theta0); 

% Results
x = q(:,1); 
y = q(:,2); 
theta = q(:,3); 

% Plot all values 
figure(4);
subplot(2,2,1)
plot(x,y)
grid on
title('Position')
xlabel('x (cm)') 
ylabel('y (cm)')
legend('U=[t;-t;4]')

subplot(2,2,2)
plot(t,x)
grid on
title('x vs t')
xlabel('t (s)') 
ylabel('x (cm)')

subplot(2,2,3)
plot(t,y)
grid on
title('y vs t')
xlabel('t (s)') 
ylabel('y (cm)')

subplot(2,2,4)
plot(t,theta)
grid on
title('theta vs t')
xlabel('t (s)') 
ylabel('theta (rad)')

function dqdt = createODE_constant_input(~, q, u, inputs) 

    r = inputs(1);
    l = inputs(2);
    
    % Angles beta
    B1 = inputs(3);
    B2 = inputs(4);
    B3 = inputs(5);
    
    G1 = [cos(B1 + q(3)) sin(B1 + q(3)) (l)]; 
    G2 = [cos(B2 + q(3)) sin(B2 + q(3)) (l)]; 
    G3 = [cos(B3 + q(3)) sin(B3 + q(3)) (l)];
    
    G = (1/r)*[G1;G2;G3]; 
    dqdt = inv(G)*u; %#ok<MINV> 
end

function dqdt = createODE_variable_input(t, q, inputs) 

    r = inputs(1);
    l = inputs(2);
    
    % Angles beta
    B1 = inputs(3);
    B2 = inputs(4);
    B3 = inputs(5);
    
    G1 = [cos(B1 + q(3)) sin(B1 + q(3)) (l)]; 
    G2 = [cos(B2 + q(3)) sin(B2 + q(3)) (l)]; 
    G3 = [cos(B3 + q(3)) sin(B3 + q(3)) (l)];
    
    G = (1/r)*[G1;G2;G3]; 
    u = [1*t; -1*t; 4];
    dqdt = inv(G)*u; %#ok<MINV> 
end