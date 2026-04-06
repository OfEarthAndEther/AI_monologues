%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Statistical_Tests.m
%
%  Unified statistical testing framework for NIA comparison.
%  Implements: Wilcoxon Rank-Sum (Mann-Whitney U), Friedman Rank,
%              Holm post-hoc correction.
%
%  All tests are applied at significance level alpha = 0.05.
%
%  BUG FIX v1.1 (BUG-2):
%    v1.0 used the Wilcoxon SIGNED-RANK test (paired test).
%    This was WRONG because the 50 runs of APKO and the 50 runs of any
%    competitor are produced under DIFFERENT random seeds:
%      APKO:       rng(r + 1*1000 + f*100)   [a=1]
%      Competitor: rng(r + j*1000 + f*100)   [a=j, j≥2]
%    Different seeds → INDEPENDENT samples → paired test is invalid.
%
%    The correct test for two independent groups is the
%    Wilcoxon RANK-SUM test (= Mann-Whitney U test).
%    H0: the two distributions are identical.
%    H1 (one-tailed): APKO samples tend to be smaller (better).
%
%    Implementation: MannWhitneyU() below — no Statistics Toolbox needed.
%
%  Usage:
%    results = Statistical_Tests(data, algo_names)
%    % data       : N_runs x N_algos matrix of final fitness values
%    % algo_names : 1 x N_algos cell array of algorithm name strings
%
%  Returns struct with fields:
%    .wilcoxon_p      p-values (Mann-Whitney U): APKO vs each competitor
%    .wilcoxon_h      Rejection flags at alpha=0.05
%    .holm_reject     Holm-corrected rejection decisions
%    .friedman_p      Friedman global test p-value
%    .friedman_stat   Friedman chi-squared statistic
%    .friedman_ranks  Mean Friedman ranks per algorithm (lower = better)
%    .U_stats         Mann-Whitney U statistic for each pair
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function results = Statistical_Tests(data, algo_names)

[N_runs, N_algos] = size(data);
fprintf('\n========= STATISTICAL ANALYSIS =========\n');
fprintf('Runs per algorithm : %d\n', N_runs);
fprintf('Algorithms         : %d\n', N_algos);
fprintf('Significance level : alpha = 0.05\n');
fprintf('Pairwise test      : Wilcoxon Rank-Sum (Mann-Whitney U)\n');
fprintf('                     [independent groups — NOT signed-rank]\n\n');

%% ---- 1. Mann-Whitney U Test: APKO vs each competitor ---------------
%  Column 1 MUST be APKO (the proposed algorithm).
%  Each pair (APKO, competitor_j) is tested independently.
p_values = zeros(1, N_algos);
h_values = zeros(1, N_algos);
U_stats  = zeros(1, N_algos);

APKO_data = data(:, 1);

for j = 2:N_algos
    comp_data = data(:, j);

    if all(APKO_data == comp_data)
        p_values(j) = 1.0;
        h_values(j) = 0;
        U_stats(j)  = N_runs * N_runs / 2;   % U = n1*n2/2 when identical
        continue;
    end

    [p_values(j), h_values(j), U_stats(j)] = ...
        MannWhitneyU(APKO_data, comp_data, 0.05);
end

%% ---- 2. Holm Step-Down Correction (controls FWER) ------------------
comp_indices = 2:N_algos;
p_comp       = p_values(comp_indices);
[p_sorted, sort_order] = sort(p_comp);
m            = length(p_comp);
alpha        = 0.05;
holm_reject  = false(1, m);
for k = 1:m
    if p_sorted(k) < alpha / (m - k + 1)
        holm_reject(k) = true;
    else
        break;   % Step-down: stop at first non-rejection
    end
end
holm_results = false(1, N_algos);
holm_results(comp_indices(sort_order)) = holm_reject;

%% ---- 3. Friedman Rank Test -----------------------------------------
%  Ranks each run's results across all algorithms (lower fitness = rank 1).
ranks_matrix = zeros(N_runs, N_algos);
for r = 1:N_runs
    [sorted_row, ~] = sort(data(r,:));
    for k = 1:N_algos
        tied_pos = find(sorted_row == data(r,k));
        ranks_matrix(r,k) = mean(tied_pos);
    end
end

mean_ranks = mean(ranks_matrix, 1);

N_f  = N_runs;
k_f  = N_algos;
SS_t = N_f * sum((mean_ranks - (k_f+1)/2).^2);
friedman_stat = 12 * SS_t / (k_f*(k_f+1));
friedman_p    = 1 - chi2cdf_approx(friedman_stat, k_f-1);

%% ---- 4. Compile Results --------------------------------------------
results.wilcoxon_p     = p_values;
results.wilcoxon_h     = h_values;
results.U_stats        = U_stats;
results.holm_reject    = holm_results;
results.friedman_p     = friedman_p;
results.friedman_stat  = friedman_stat;
results.friedman_ranks = mean_ranks;
results.algo_names     = algo_names;

%% ---- 5. Print Summary ----------------------------------------------
fprintf('--- Friedman Test (global ranking) ---\n');
fprintf('Chi-squared statistic : %.4f  (df=%d)\n', friedman_stat, k_f-1);
fprintf('p-value               : %.6f\n', friedman_p);
if friedman_p < 0.05
    fprintf('Result: SIGNIFICANT global difference (p < 0.05)\n');
else
    fprintf('Result: No significant global difference (p >= 0.05)\n');
end

fprintf('\n--- Friedman Mean Ranks (lower = better) ---\n');
[sorted_ranks, ri] = sort(mean_ranks);
for i = 1:N_algos
    fprintf('  Rank %2d : %-22s  (mean rank = %.3f)\n', ...
            i, algo_names{ri(i)}, sorted_ranks(i));
end

fprintf('\n--- Mann-Whitney U: APKO vs Each Competitor ---\n');
fprintf('%-22s  %10s  %10s  %10s  %15s\n', ...
        'Competitor', 'U stat', 'p-value', 'Signif.', 'Holm-corrected');
fprintf('%s\n', repmat('-', 1, 75));
for j = 2:N_algos
    sig_str  = ternary(p_values(j) < 0.05, '+', '~');
    holm_str = ternary(holm_results(j), 'Reject H0', 'Accept H0');
    fprintf('%-22s  %10.1f  %10.6f  %10s  %15s\n', ...
            algo_names{j}, U_stats(j), p_values(j), sig_str, holm_str);
end
fprintf('\n(+) = APKO significantly better at alpha=0.05 (one-tailed)\n');
fprintf('(~) = No statistically significant difference\n');
fprintf('\nNote: Wilcoxon Rank-Sum (Mann-Whitney U) used — INDEPENDENT groups.\n');
fprintf('      Runs seeded differently per algorithm; paired test invalid.\n\n');

results.summary_printed = true;
end


%% =====================================================================
%% Mann-Whitney U Test  (= Wilcoxon Rank-Sum for independent groups)
%%
%%  H0: distributions of x and y are identical
%%  H1 (one-tailed, lower tail): x tends to be smaller than y
%%      i.e. P(x < y) > 0.5  →  APKO is better
%%
%%  Method: Pool and rank all n1+n2 observations together.
%%          U1 = R1 - n1*(n1+1)/2  where R1 = sum of ranks of group x.
%%          Normal approximation: valid for n1,n2 > 10.
%%
%%  Inputs:  x (N_runs x 1) APKO fitness values
%%           y (N_runs x 1) competitor fitness values
%%           alpha          significance level
%%  Outputs: p   one-tailed p-value (P(U ≤ u) under H0)
%%           h   1 if H0 rejected at alpha, else 0
%%           U   Mann-Whitney U statistic for group x
%% =====================================================================
function [p, h, U] = MannWhitneyU(x, y, alpha)
    x = x(:);  y = y(:);
    n1 = length(x);  n2 = length(y);

    % Pool groups and label them
    pooled  = [x; y];
    labels  = [ones(n1,1); 2*ones(n2,1)];

    % Rank pooled data (average ranks for ties)
    [~, sort_idx] = sort(pooled);
    ranks = zeros(n1+n2, 1);
    ranks(sort_idx) = 1:(n1+n2);

    % Resolve ties: assign mean rank to tied values
    unique_vals = unique(pooled);
    for uv = unique_vals'
        tied_mask     = (pooled == uv);
        ranks(tied_mask) = mean(ranks(tied_mask));
    end

    % Sum of ranks for APKO group (group 1 = x)
    R1 = sum(ranks(labels == 1));

    % Mann-Whitney U statistic for group 1 (x = APKO)
    U  = R1 - n1*(n1+1)/2;

    % Normal approximation with continuity correction
    mu_U   = n1*n2/2;
    sig_U  = sqrt(n1*n2*(n1+n2+1)/12);

    % Tie correction for sigma
    % sigma_U^2 = n1*n2/12 * [(n1+n2+1) - sum(t^3-t)/(n1+n2)/(n1+n2-1)]
    t_corr = 0;
    for uv = unique_vals'
        t_k    = sum(pooled == uv);
        t_corr = t_corr + (t_k^3 - t_k);
    end
    N_tot = n1 + n2;
    sig_U_corr = sqrt(n1*n2/12 * ((N_tot+1) - t_corr/(N_tot*(N_tot-1))));
    if sig_U_corr > 0
        sig_U = sig_U_corr;
    end

    % One-tailed z-score: test if U < mu_U (APKO wins)
    % Continuity correction: U_cc = U + 0.5 (conservative)
    z = (U + 0.5 - mu_U) / sig_U;   % One-tailed: lower tail
    p = normcdf_approx(z);           % P(Z ≤ z)

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


%% =====================================================================
%% Helper: Normal CDF approximation (no Statistics Toolbox needed)
%%   Abramowitz & Stegun 26.2.17 — max error < 7.5e-8
%% =====================================================================
function p = normcdf_approx(z)
    t    = 1 / (1 + 0.2316419 * abs(z));
    poly = t*(0.319381530 + t*(-0.356563782 + t*(1.781477937 + ...
           t*(-1.821255978 + t*1.330274429))));
    p_upper = (1/sqrt(2*pi)) * exp(-z^2/2) * poly;
    if z >= 0
        p = p_upper;          % Lower-tail (APKO wins: small U)
    else
        p = 1 - p_upper;
    end
end


%% =====================================================================
%% Helper: Chi-squared CDF via Wilson-Hilferty approximation
%% =====================================================================
function p = chi2cdf_approx(x, k)
    if x <= 0; p = 0; return; end
    % Wilson-Hilferty: chi2(k) ~ Normal(k, 2k) for large k
    mu  = k;
    sig = sqrt(2*k);
    z   = (x - mu) / sig;
    p   = normcdf_approx(-z);   % Upper tail -> 1 - lower_tail
    p   = 1 - max(0, min(1, p));
end


%% =====================================================================
%% Helper: Ternary operator
%% =====================================================================
function out = ternary(cond, a, b)
    if cond; out = a; else; out = b; end
end
