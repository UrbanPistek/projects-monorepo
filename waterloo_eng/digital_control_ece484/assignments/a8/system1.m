% assignment 8
clc
clear
close all

n = [2.96 -12.04 11.90];
d = [1 -5.54 9.49 -4.95];
G = tf(n, d); 
printsys(n, d);
fprintf("\n\n");

nd = [0 1];
dd = [0 1];
D = tf(nd, dd); 
printsys(nd, dd);
fprintf("\n\n"); 

L = D*G;

figure(1)
rlocus(L);

figure(2);
nyquist(L);
