% Load variables 
clc

% constants
% T = 0.1;
d = 0; 
tfinal = 5;

% 1st TF 2/s
n2 = [1 7 12]; %#ok<NBRAK> 
d2 = [1 2 3 0];
C = tf(n2, d2); 
printsys(n2, d2);

% 2nd TF
n3 = [1 3 2]; %#ok<NBRAK> 
d3 = [1 13 54 72];
P = tf(n3, d3);
printsys(n3, d3);
