clc
clear
close all

n = [1.49 -2.86];
d = [1 -3.72 2.72];
G = tf(n, d); 
printsys(n, d);
fprintf("\n\n");

nd = [1 -0.5];
dd = [1 -0.25];
D = tf(nd, dd); 
printsys(nd, dd);
fprintf("\n\n"); 

L = G*D;

figure(1)
rlocus(L);

figure(2);
nyquist(L);
