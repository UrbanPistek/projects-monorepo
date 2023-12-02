% Load variables 
clc

% constants
% T = 0.1;
d = 0; 
tfinal = 5;

% 1st TF
n1 = [T T];
d1 = [1 -1];
D = tf(n1, d1, T); 
printsys(n1, d1);

% 2nd TF 2/s
n2 = [2]; %#ok<NBRAK> 
d2 = [1 0];
C = tf(n2, d2); 
printsys(n2, d2);

% 3rd TF
n3 = [1]; %#ok<NBRAK> 
d3 = [1 2];
P = tf(n3, d3);
printsys(n3, d3);
