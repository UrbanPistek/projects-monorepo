% question 3
clc
clear
close all
format long

A = [
    0 1 0; 
    0 0 1;
    2 3 4;
];

B = [0 0 1];

C = [1 1 1;];

n = 3;

I = eye(n);

syms f1 f2 f3
F = [f1 f2 f3];

% Find eigenvalues of A 
eigen = eig(A);
disp(eigen);

% for each eigenvalue
for i = 1:size(eigen)
    disp(eigen(i))

    temp = [(A - eigen(i)*I) B'];

    disp(temp);

    k = rank(temp);
    disp("rank:")
    disp(k);
end

% Find F
syms z
Q = z*I - (A - B'*F);
disp(Q);

determinant = det(Q);
disp(determinant);

syms z
c = coeffs(determinant, z);
c = fliplr(c);
disp(c);

% desired poles
syms Ts
ddes = (z - exp(-0.1*Ts))*(z - exp((-0.2 + 0.3i)*Ts))*(z - exp((-0.2 - 0.3i)*Ts));
f = expand(ddes);
cd = coeffs(f, z);
cd = fliplr(cd);
disp(cd);

f3 = cd(2) + 4;
f2 = cd(3) + 3; 
f1 = cd(4) + 2; 

F = [f1 f2 f3]; 

W = A - B'*F; 
eigen2 = eig(W);
disp(eigen2);
