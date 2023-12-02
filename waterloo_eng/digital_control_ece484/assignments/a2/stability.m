% transfer functions
clc

% Using theorem:
syms s
N = (s+1)*(s+2);
D = (s+3)*(s+4)*(s+6);
f = (s+3)*(s+4);
g = (s)*(s+1)*(s+2);

delta = N*f + D*g;
disp(delta)

eqn = delta == 0;

roots = solve(delta);
disp("Roots:")
disp(roots)

zeros = double(solve(eqn));
disp("Numeric Value for Roots:")
disp(zeros)