% Question 1
clc
clear
close all
format long;

% T = [0.02, 0.2, 1, 2];
T = [0.002];

for idx = 1:length(T)

    disp(T(idx))

    % a)
    
    % Solve system of differential equations using euler integration
    tspan = 0:T(idx):25;
    [ta,ya] = ode45(@vdp1,tspan,[pi/2; 0]);
    
    % Transpose the results
    ta = ta';
    ya = ya';
    
    figure(1*idx);
    plot(ta,ya(1,:),'-x',ta,ya(2,:),'-o')
    title(sprintf(" Solution T=%f", T(idx)));
    xlabel('Time (t)');
    ylabel('Solution x(t)');
    legend('x1','x2')
    
    % Defining the three differential equations of the problem
    g = 9.8;
    l = g/2; 
    m = 2/g; 
    D = 0.1;
    u = 0.1;
    f = @(t,y) [y(2); (-g/l)*sin(y(1)) - (D/(m*l))*y(2) + (1/(m*l))*u;];
    
    [t,y] = explicit_euler(f, [0,25], [pi/2;0;], T(idx)); % Calling Euler function
    
    x1 = y(1,:);
    x2 = y(2,:);
    
    % Save euler integration results
    x1e = y(1,:);
    x2e = y(2,:);
    
    figure(2*idx);
    grid on
    plot(t, x1, '--', t, x2, '--');
    title(sprintf("Euler Integration T=%f", T(idx)));
    legend('x1(t)', 'x2(t)', 'Location', 'NorthEast')
    xlabel('t')
    ylabel('Solutions')
    
    figure(3*idx);
    grid on
    plot(t, x1, '--', t, x2, '--', ta, y(1,:), ta, y(2,:));
    title(sprintf("Euler Integration Comparison T=%f", T(idx)));
    legend('x1(t)', 'x2(t)', 'x1_a(t)', 'x2_a(t)', 'Location', 'NorthEast')
    xlabel('t')
    ylabel('Solutions')
    
    % b)
    
    format long;
    f = @(x)[x(1)*x(2) + sin(x(1)); x(1) + x(2)^2 - 2];
    x = [0.1;0.1];
    xa = fsolve (f, x);
    
    disp("fsolve: ")
    disp(xa)
    
    n = 20;
    tol = 0.00001;
    
    % Jacobian
    J = @(x)[cos(x(1)) + x(2), x(1); 1, 2*x(2)];
    
    for i = 1: n
        Dx = -J(x)\f(x); % solve for increment
        x = x + Dx; % add on to get new guess
        err = f(x); % see if f(x) is really zero
        
        if err < tol
            iter = i;
            break
        end
    
    end
    
    disp("newtons: ")
    disp(x)
    
    % c)
    g = 9.8;
    l = g/2; 
    m = 2/g; 
    D = 0.1;
    u = 0.1;
    f = @(t,y) [y(2); (-g/l)*sin(y(1)) - (D/(m*l))*y(2) + (1/(m*l))*u;];
    
    % Jacobian
    J = @(t, y)[0, 1; (-g/l)*cos(y(1)), -D/(m*l)];
    
    [t,y] = trapezoidal_integration(f, J, [0,25], [pi/2;0;], T(idx)); % Calling Euler function
    
    x1 = y(1,:);
    x2 = y(2,:);
    
    figure(4*idx);
    grid on
    plot(t, x1, '--', t, x2, '--');
    title(sprintf("Trapezoid Integration T=%f", T(idx)));
    legend('x1(t)', 'x2(t)', 'Location', 'NorthEast')
    xlabel('t')
    ylabel('Solutions')
    
    figure(5*idx);
    grid on
    plot(t, x1, '--', t, x2, '--', t, x1e, t, x2e);
    title(sprintf("Trapezoid Integration vs Euler Integration T=%f", T(idx)));
    legend('x1(t)', 'x2(t)', 'x1_e(t)', 'x2_e(t)', 'Location', 'NorthEast')
    xlabel('t')
    ylabel('Solutions')

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For ode45
function dydt = vdp1(t,x) %#ok<INUSL> 
g = 9.8;
l = g/2; 
m = 2/g; 
D = 0.1;
u = 0.1;

dydt = [x(2); (-g/l)*sin(x(1)) - (D/(m*l))*x(2) + (1/(m*l))*u];
end

function [x, y] = explicit_euler( f, xRange, y_initial, h )
    % This function uses Euler’s explicit method to solve the ODE
    % dv/dt=f(t,v); x refers to independent and y refers to dependent variables
    % f defines the differential equation of the problem
    % xRange = [x1, x2] where the solution is sought on
    % y_initial = column vector of initial values for y at x1
    % numSteps = number of equally-sized steps to take from x1 to x2
    % x = row vector of values of x
    % y = matrix whose k-th column is the approximate solution at x(k)
    
    x(1) = xRange(1);
    numSteps = ( xRange(2) - xRange(1) ) /h ;
    y(:,1) = y_initial(:);
    
    for k = 1 : numSteps
        x(k + 1) = x(k) + h; %#ok<AGROW> 
        y(:,k+1) = y(:,k) + h * f( x(k), y(:,k) );
    end

end

function [x, y] = trapezoidal_integration(f, J, xRange, y_initial, h)
    
    n = 20;
    tol = 0.00001;
    
    x(1) = xRange(1);
    numSteps = ( xRange(2) - xRange(1) ) /h ;
    y(:,1) = y_initial(:);
    
    for k = 1 : numSteps
        x(k + 1) = x(k) + h; %#ok<AGROW> 
        
        % newtons method for y_n + 1 
        y_k_1 = y(:, k);
        for i = 1: n
            Dx = -J(x(k), y_k_1)\f(x(k), y_k_1); % solve for increment
            y_k_1 = y_k_1 + Dx; % add on to get new guess

            err = f(x(k), y_k_1); % see if f(x) is really zero
            
            if err < tol
                break
            end
        end
        
        % calculate trapezoidal approzimation
        y(:,k+1) = y(:,k) + (1/2)*h*(f(x(k), y(:,k)) + f(x(k+1), y_k_1));
    end

end
