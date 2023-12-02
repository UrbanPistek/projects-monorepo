% verify specs

% length of data
t_len = length(t_sim);
steady_state_start = cast(t_len*0.75, "uint32");

peak = max(y_sim_dt);
peak_idx = find(y_sim_dt==peak);
tp = t_sim(peak_idx);
yss = mean(y_sim_dt(steady_state_start:end));
os = (peak - yss)/peak;
per_os = os*100;

ts = 1;
for i = peak_idx:length(y_sim_dt)
    if (y_sim_dt(i) < (yss + yss*0.02)) && (y_sim_dt(i) > (yss - yss*0.02))

        ts = t_sim(i);
        % fprintf('ts=%f, i=%d, val=%f \n', ts, i, y_sim_ct(i));
        break
    end
end

fprintf('tp =%f \n', tp);
fprintf('peak =%f \n', peak);
fprintf('yss =%f \n', yss);
fprintf('percent os =%f \n', per_os);
fprintf('ts =%f \n', ts);