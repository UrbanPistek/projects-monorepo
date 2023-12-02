clear all
close all
clc

%example code written by: Christian Mele

%time step of analysis
dt = 0.1; % (relatively slow refresh rate)

%% Prediction estimation - this is your "Priori"
% we will be using IMU values for the project, however in this example we
% use a block with a spring attached to it 

xhat = [0.5;1;0]; %mean(mu) estimate for the "first" step
P = 1; %covariance initial estimation between the 
% the covariance matrix P is the weighted covariance between the  
% motion model definition - establish your robots "movement"
k = 1; %spring value
m = 10; %mass

A = [1, dt, 1/2*dt^2; 0, 1, dt; k/m, 0, 0]; %this is the state space
B = [0;0;1/m]; %this is the "input" 
B = [0;0;1]; % we can also treat the input as the "acceleration" for that step as calculated by an IMU!
Q = eye(3,3)*0.05; % this is the model variance/covariance (usually we treat it as the input matrix squared).
% these are the discrete components, found by performing a taylor's
% expansion on the continuous model



%% Measurement Model - Update component 
C = [-1,0, 0; 0,0.6,0]; %what this represents is our "two" sensors, both with linear relationships 
% to position and velocity respectively 

% in actual environments, what this does is translate our measurement to a
% voltage or some other metric that can be ripped directly from the sensor
% when taking online measurements. We compare those values as our "error"

R = [0.05, 0, ; 0, 0.05, ]; % this is the sensor model variance-usually characterized to accompany 
% the sensor model already starting 

%runs for ten seconds
Tfinal = 10;
T = 0:dt:Tfinal;
x = zeros(3,length(T) + 1);
x(:,1) = xhat;
y = zeros(2,length(T));

%% Main run loop
for k=1:length(T)

    u = 0.01; %normally you'd initialise this above
    %% Simulate motion with random motion disturbance

    w = [Q(1,1)*randn(1);Q(2,2)*randn(1);Q(3,3)*randn(1)];
    % update state - this is a simulated motion and is PURELY for fake
    % sensing and would essentially be
    x(:,k+1) = A*x(:,k) + B*u + w;

    %taking a measurement - simulating a sensor
    % create our sensor disturbance 
    v = [R(1,1)*randn(1);R(2,2)*randn(1)];
    % create this simulated sensor measurement
    y(:,k) = C*x(:,k+1) + v;


    %% Kalman Filter Estimation
    
    %Prediction update
    xhat_k = A*xhat + B*u; %we do not put noise on our prediction
    P_predict = A*P*A' + Q; %this co-variance is the prediction of essentially how the measurement and sensor model move together
    % in relation to each state and helps scale our kalman gain by giving
    % the ratio. By Definition, P is the variance of the state space, and
    % by applying it to the motion model we're getting a motion uncertainty
    % which can be propogated and applied to the measurement model and
    % expand its uncertainty as well

    % Measurement Update and Kalman Gain
    K = P_predict*C'*inv(C*P_predict*C' + R); %the pseudo inverse of the measurement model, as it relates to the model covariance
    % if we don't have a measurement for velocity, the P-matrix tells the
    % measurement model how the two should move together (and is normalised
    % in the process with added noise), which is how the kalman gain is
    % created --> detailing "how" the error should be scaled based on the
    % covariance. If you expand P_predict out, it's clearly the
    % relationship and cross-projected relationships, of the states from a
    % measurement and motion model perspective, with a moving scalar to
    % help drive that relationship towards zero (P should stabilise).

    xhat = xhat_k + K*(y(:,k) - C*xhat_k);
    P = (1-K*C)*P_predict; %the full derivation for this is kind of complex relying on 
    % some pretty cool probability knowledge
    
    %store estimates
    xhat_S(:,k) = xhat_k;
    x_S(:,k) = xhat;
    y_hat(:,k) = C*xhat;
end

%Plot full trajectory 
figure(1)
hold on
plot(T,x_S)
hold on
plot(T,x(:,2:end))
legend('position est.','vel estimate', 'accel est', 'true pos', 'true vel', 'true accel');

% predictions
figure(2)
hold on
plot(T,xhat_S)
hold on
plot(T,x(:,2:end))
