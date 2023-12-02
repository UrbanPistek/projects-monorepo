% example
clc
clear
close all

syms s
delta_ds = (s+1+1.9i)*(s+1-1.9i)*(s+2)*(s+3);

f = expand(delta_ds);
disp(delta_ds);
disp(f);

c = single(coeffs(f));
c = fliplr(c); % flip order
disp(c);

ci = [c 0];
disp(ci);

b = ci(:);
disp(b);

A = [
    1 0 0 0 0 0;
    1 1 0 0 0 0;
    1/4 1 1 1 0 0;
    0 1/4 1 0 1 0;
    0 0 1/4 0 0 1;
    0 0 -19/4 0 0 1;
    ];
disp(A);

% x = [g2; g1; g0; f2; f1; f0];
x = linsolve(A, b);

format long
disp(x);
