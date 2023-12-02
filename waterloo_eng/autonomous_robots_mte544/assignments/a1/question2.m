% Question 2
clc
clear

% Transformation from {e} -> {3}
psi = -pi/2; 
l3 = 1;
Re = [cos(psi) -sin(psi); sin(psi) cos(psi)];
qe = [l3; 0]; 
Ge3 = [Re qe; 0 0 1];

% Transformation from {3} -> {2}
theta = -pi/3; 
l2 = 1;
R3 = [cos(theta) -sin(theta); sin(theta) cos(theta)];
q3 = [l2; 0]; 
G32 = [R3 q3; 0 0 1];

% Transformation from {2} -> {1}
gamma = pi/4; 
l1 = 1;
R2 = [cos(gamma) -sin(gamma); sin(gamma) cos(gamma)];
q2 = [l1; 0]; 
G21 = [R2 q2; 0 0 1];

% Transformation from {1} -> {s}
alpha = pi/4; 
x = 1; 
y = 1;
R1 = [cos(alpha) -sin(alpha); sin(alpha) cos(alpha)];
q1 = [x; y]; 
G1s = [R1 q1; 0 0 1];

% Transformation Matrix
Ges = G1s*G21*G32*Ge3;

% Calculate Transformed Coordinates 
% p = [0 0];
px = 0;
py = 0; 
p = [px; py; 1]; 
ps1 = Ges*p;

% p = [1 2];
px = 1;
py = 2; 
p = [px; py; 1]; 
ps2 = Ges*p;
