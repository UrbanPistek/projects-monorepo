% run
clc
clear

% constants
tfinal = 25;

% 1st TF 2/s
ks = [1, 0.1, 0.01];

for k = ks

    fprintf('\n k =%f \n', k);

    % Define TF
    n = [-0.01 0 k^2];
    d = [k (15*k^2 + 2*k) (30*k^2 + 2*k) (30*k^2)];
    G = tf(n, d); 
    printsys(n, d);
    fprintf("\n\n");
    
    % Simulate system
    sim("ct_system.slx");
    
    plot_ct_system;
    
    % Estimate settling time, percent overshoot, and time-to-peak
    peak = max(y_sim);
    peak_idx = find(y_sim==peak);
    tp = t_sim(peak_idx);
    yss = mean(y_sim(10000:12000));
    os = peak - yss;
    
    ts = 1;
    for i = peak_idx:length(y_sim)
        if (y_sim(i) < (yss + yss*0.02)) && (y_sim(i) > (yss - yss*0.02))

            % verify this does not osillate further
            if y_sim(i) > max(y_sim(i+1:end))

                ts = t_sim(i);
                fprintf('ts=%f, i=%d, val=%f \n', ts, i, y_sim(i));
                break
            end
            
        end
    end

    fprintf('tp =%f \n', tp);
    fprintf('peak =%f \n', peak);
    fprintf('yss =%f \n', yss);
    fprintf('os =%f \n', os);
    fprintf('ts =%f \n', ts);

    % Individual Plots
    figure();
    grid on
    plot(t_sim, y_sim);
    title(sprintf('Response k=%d', k));
    xlabel("Time (s)");
    ylabel("Response (y(t))");

end

figure(1);
legend(["k=1", "k=0.1", "k=0.01"], "Location","southeast");
