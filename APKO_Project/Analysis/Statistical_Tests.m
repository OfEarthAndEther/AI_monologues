%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Statistical_Tests.m
%
%  Unified statistical testing framework for NIA comparison.
%  Implements: Wilcoxon Rank-Sum, Friedman Rank, Holm post-hoc correction.
%
%  All tests are applied at significance level alpha = 0.05.
%
%  Usage:
%    results = Statistical_Tests(data, algo_names)
%    % data       : N_runs x N_algos matrix of final fitness values
%    % algo_names : 1 x N_algos cell array of algorithm name strings
%
%  Returns:
%    results.wilcoxon   : p-values for APKO vs each competitor (pairwise)
%    results.holm       : Holm-corrected rejection decisions (1=reject H0)
%    results.friedman_p : Friedman global p-value
%    results.friedman_rank : Mean Friedman ranks per algorithm
%    results.summary    : formatted summary table
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function results = Statistical_Tests(data, algo_names)

[N_runs, N_algos] = size(data);
fprintf('\n========= STATISTICAL ANALYSIS =========\n');
fprintf('Runs per algorithm : %d\n', N_runs);
fprintf('Algorithms         : %d\n', N_algos);
fprintf('Significance level : alpha = 0.05\n\n');

%% ---- 1. Wilcoxon Rank-Sum Test: APKO vs each competitor -----------
% Assumes column 1 = APKO (proposed algorithm)
p_values = zeros(1, N_algos);
h_values = zeros(1, N_algos);
w_stats  = zeros(1, N_algos);

APKO_data = data(:, 1);   % Column 1 must be APKO

for j = 2:N_algos
    comp_data = data(:, j);

    % Check if data is identical (no test needed)
    if all(APKO_data == comp_data)
        p_values(j) = 1.0;
        h_values(j) = 0;
        w_stats(j)  = NaN;
        continue;
    end

    % Wilcoxon signed-rank test (paired, one-tailed: APKO < competitor)
    try
        [p_values(j), h_values(j), ~] = WilcoxonSignedRank(APKO_data, comp_data, 0.05);
        w_stats(j) = h_values(j);
    catch
        p_values(j) = 1.0;
        h_values(j) = 0;
    end
end

%% ---- 2. Holm Post-Hoc Correction (controls FWER) ------------------
% Apply Holm step-down procedure to all pairwise p-values
comp_indices = 2:N_algos;
p_comp = p_values(comp_indices);
[p_sorted, sort_order] = sort(p_comp);
m = length(p_comp);
alpha = 0.05;
holm_reject = false(1, m);
for k = 1:m
    if p_sorted(k) < alpha / (m - k + 1)
        holm_reject(k) = true;
    else
        break;   % Holm is step-down: stop at first non-rejection
    end
end
% Map back to original order
holm_results = false(1, N_algos);
holm_results(comp_indices(sort_order)) = holm_reject;

%% ---- 3. Friedman Rank Test (global non-parametric ANOVA) ----------
% Rank each run's results across all algorithms
ranks_matrix = zeros(N_runs, N_algos);
for r = 1:N_runs
    [~, ~, ic] = unique(data(r,:));
    % Average ranks for ties
    for k = 1:N_algos
        same = (data(r,:) == data(r,k));
        ranks_matrix(r,k) = mean(find(sort(data(r,:)) == data(r,k)));
    end
    % Proper rank assignment
    [sorted_row, ~] = sort(data(r,:));
    for k = 1:N_algos
        tied_positions = find(sorted_row == data(r,k));
        ranks_matrix(r,k) = mean(tied_positions);
    end
end

mean_ranks = mean(ranks_matrix, 1);

% Friedman statistic: chi-squared distributed with (N_algos-1) df
N = N_runs; k = N_algos;
SS_t = N * sum((mean_ranks - (k+1)/2).^2);
friedman_stat = 12*SS_t / (k*(k+1));

% p-value from chi-squared distribution
friedman_p = 1 - chi2cdf_approx(friedman_stat, k-1);

%% ---- 4. Compile Results -------------------------------------------
results.wilcoxon_p     = p_values;
results.wilcoxon_h     = h_values;
results.holm_reject    = holm_results;
results.friedman_p     = friedman_p;
results.friedman_stat  = friedman_stat;
results.friedman_ranks = mean_ranks;
results.algo_names     = algo_names;

%% ---- 5. Print Summary Table ----------------------------------------
fprintf('--- Friedman Test ---\n');
fprintf('Chi-squared statistic : %.4f\n', friedman_stat);
fprintf('p-value               : %.6f\n', friedman_p);
if friedman_p < 0.05
    fprintf('Result: SIGNIFICANT global difference (p < 0.05)\n');
else
    fprintf('Result: No significant global difference detected\n');
end
fprintf('\n--- Algorithm Rankings (Friedman Mean Rank, lower = better) ---\n');
[sorted_ranks, ri] = sort(mean_ranks);
for i = 1:N_algos
    fprintf('  Rank %2d : %-20s  (mean rank = %.3f)\n', ...
            i, algo_names{ri(i)}, sorted_ranks(i));
end

fprintf('\n--- Wilcoxon Signed-Rank: APKO vs Competitors ---\n');
fprintf('%-20s  %10s  %12s  %15s\n', 'Competitor', 'p-value', 'Significant', 'Holm-corrected');
fprintf('%s\n', repmat('-', 1, 65));
for j = 2:N_algos
    sig_str   = ternary(p_values(j) < 0.05, '+', '~');
    holm_str  = ternary(holm_results(j), 'Reject H0', 'Accept H0');
    fprintf('%-20s  %10.6f  %12s  %15s\n', ...
            algo_names{j}, p_values(j), sig_str, holm_str);
end
fprintf('\n(+) = APKO significantly better at alpha=0.05\n');
fprintf('(~) = No significant difference\n');

results.summary_printed = true;
end


%% =====================================================================
%% Local: Wilcoxon Signed-Rank Test (manual implementation)
%%   Tests H0: median(x-y) = 0
%%   H1 (one-tailed): median(x-y) < 0  (x < y, x=APKO is better)
%% =====================================================================
function [p, h, W] = WilcoxonSignedRank(x, y, alpha)
    d = x - y;
    d = d(d ~= 0);   % Remove zero differences
    n = length(d);

    if n == 0
        p = 1; h = 0; W = NaN; return;
    end

    [~, sort_idx] = sort(abs(d));
    sorted_d = d(sort_idx);
    ranks = 1:n;

    % Handle ties: assign average rank
    abs_d_sorted = abs(sorted_d);
    unique_vals = unique(abs_d_sorted);
    for uv = unique_vals'
        tied = abs_d_sorted == uv;
        ranks(tied) = mean(ranks(tied));
    end

    W_plus  = sum(ranks(sorted_d > 0));
    W_minus = sum(ranks(sorted_d < 0));
    W = min(W_plus, W_minus);

    % Normal approximation (valid for n > 10)
    mu_W = n*(n+1)/4;
    sig_W = sqrt(n*(n+1)*(2*n+1)/24);
    z = (W - mu_W) / sig_W;
    p = normcdf_approx(z);   % One-tailed p-value
    h = (p < alpha);
end


%% =====================================================================
%% Helper: Normal CDF approximation (no Statistics Toolbox needed)
%% =====================================================================
function p = normcdf_approx(z)
    % Abramowitz & Stegun approximation 26.2.17
    t = 1 / (1 + 0.2316419 * abs(z));
    poly = t*(0.319381530 + t*(-0.356563782 + t*(1.781477937 + ...
           t*(-1.821255978 + t*1.330274429))));
    p_pos = 1 - (1/sqrt(2*pi))*exp(-z^2/2)*poly;
    if z >= 0; p = p_pos; else; p = 1 - p_pos; end
end


%% =====================================================================
%% Helper: Chi-squared CDF (regularised incomplete gamma function)
%% =====================================================================
function p = chi2cdf_approx(x, k)
    % Wilson-Hilferty approximation for chi-squared CDF
    if x <= 0; p = 0; return; end
    mu = k; sig = sqrt(2*k);
    z = (x - mu) / sig;
    p = normcdf_approx(z);
    p = max(0, min(1, p));
end


%% =====================================================================
%% Helper: Ternary operator
%% =====================================================================
function out = ternary(cond, a, b)
    if cond; out = a; else; out = b; end
end
