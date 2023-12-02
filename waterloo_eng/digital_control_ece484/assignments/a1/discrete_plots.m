clc
clear

% plot discrete function
figure(1);
n = (-5:0.25:5);
y = 5*cos(pi*(n/2)-(pi/2)); 
stem(n,y);

% plot regular function
figure(2);
t = linspace(0, 5, 100);
k = 1;
y = (k^2).*exp(-k.*t).*cos(5.*k.*t);
plot(t,y)

hold on
k = 2; 
y = (k^2).*exp(-k.*t).*cos(5.*k.*t);
plot(t,y)
legend

% sample data
figure(3);
k = 1;
y = (k^2).*exp(-k.*t).*cos(5.*k.*t);
plot(t,y)

hold on
n = (0:1:5);
yk = (k^2).*exp(-k.*n).*cos(5.*k.*n);
stem(n,yk);

% hold
figure(4);
k = 1;
n = (0:1:5);
yk = (k^2).*exp(-k.*n).*cos(5.*k.*n);
plot(n,yk)

hold on
stairs(n,yk)
