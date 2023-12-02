% question 4
clc
clear
close all
format long

A = [
    0.25 0;
    0 4;
];

B = [0 1];

C = [1 1;];

n = 2;

I = eye(n);

syms f1 f2
F = [f1 f2];

% Find eigenvalues of A 
eigen = eig(A);
disp(eigen);

% for each eigenvalue
for i = 1:size(eigen)

    temp = [(A - eigen(i)*I) B'];

    disp(temp);

    k = rank(temp);
    disp("rank:")
    disp(k);
end

% not possible design controller to because rank of temp matrix using all 
% eigenvalues is not equal to n 