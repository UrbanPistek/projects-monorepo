% test

clc
clear
close all

n = [0 0 1];
d = [1 -0.5 -3];
G = tf(n, d); 
printsys(n, d);
fprintf("\n\n");

nd = [0 1];
dd = [0 1];
D = tf(nd, dd); 
printsys(nd, dd);
fprintf("\n\n"); 

L = D*G;

figure(1);
rlocus(L);

figure(2);
nyquist(L);