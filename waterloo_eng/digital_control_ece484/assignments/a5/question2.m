% Question 2
clc
clear
close all

syms s 
C = (2*s^3 + 14*s^2 + 28*s + 16)/((s+1)^3);
C = simplify(C);

disp("Base TF: ")
disp(C)
pretty(C)

% a) Right side
syms T z
x = (1/T)*((z-1)/z);
D1 = (2*x^3 + 14*x^2 + 28*x + 16)/((x+1)^3);
D1 = simplify(D1);

disp("a) TF: ")
disp(D1)
pretty(D1);

% b) Left side
syms T z
x = (1/T)*(z-1);
D2 = (2*x^3 + 14*x^2 + 28*x + 16)/((x+1)^3);
D2 = simplify(D2);

disp("b) TF: ")
disp(D2)
pretty(D2);

% c) Trapezoidal
syms T z
x = (2/T)*((z-1)/(z+1));
D3 = (2*x^3 + 14*x^2 + 28*x + 16)/((x+1)^3);
D3 = simplify(D3);

disp("c) TF: ")
disp(D3)
pretty(D3);
