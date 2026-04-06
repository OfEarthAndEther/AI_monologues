%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Generate_Tables.m
%
%  Generates Mean, Std-dev, and Friedman Rank tables from experiment
%  results, formatted for direct copy into LaTeX or Word report.
%
%  Usage:
%    Generate_Tables(results_cell, algo_names, func_names, out_dir)
%
%  Inputs:
%    results_cell : cell array {N_funcs x N_algos} where each cell
%                  contains a (N_runs x 1) vector of final fitnesses
%    algo_names   : cell array of algorithm name strings
%    func_names   : cell array of function name strings
%    out_dir      : output directory for saved files (default: 'Results/')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Generate_Tables(results_cell, algo_names, func_names, out_dir)

if nargin < 4; out_dir = 'Results/'; end
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

[N_funcs, N_algos] = size(results_cell);
N_runs = length(results_cell{1,1});

Mean_table  = zeros(N_funcs, N_algos);
Std_table   = zeros(N_funcs, N_algos);
Rank_table  = zeros(N_funcs, N_algos);

%% --- Compute Mean, Std, and per-function Friedman ranks -------------
for f = 1:N_funcs
    for a = 1:N_algos
        Mean_table(f,a) = mean(results_cell{f,a});
        Std_table(f,a)  = std(results_cell{f,a});
    end
    % Rank algorithms on this function by mean (lower = better = lower rank)
    [~, ri] = sort(Mean_table(f,:));
    for a = 1:N_algos
        Rank_table(f, ri(a)) = a;
    end
end

Mean_Rank = mean(Rank_table, 1);   % Average Friedman rank per algorithm

%% --- Print Plain-Text Table -----------------------------------------
fprintf('\n========= RESULTS TABLE: MEAN ± STD =========\n');
header = sprintf('%-12s', 'Function');
for a = 1:N_algos
    header = [header, sprintf('  %-18s', algo_names{a})]; %#ok<AGROW>
end
fprintf('%s\n', header);
fprintf('%s\n', repmat('-', 1, 12 + 20*N_algos));

for f = 1:N_funcs
    row = sprintf('%-12s', func_names{f});
    for a = 1:N_algos
        cell_str = sprintf('%.2e±%.1e', Mean_table(f,a), Std_table(f,a));
        % Bold marker for best on this function
        if Rank_table(f,a) == 1
            cell_str = ['*' cell_str]; %#ok<AGROW>
        end
        row = [row, sprintf('  %-18s', cell_str)]; %#ok<AGROW>
    end
    fprintf('%s\n', row);
end

fprintf('%s\n', repmat('-', 1, 12 + 20*N_algos));
row_rank = sprintf('%-12s', 'Mean Rank');
for a = 1:N_algos
    row_rank = [row_rank, sprintf('  %-18.3f', Mean_Rank(a))]; %#ok<AGROW>
end
fprintf('%s\n', row_rank);

%% --- Save to CSV for import into LaTeX/Excel -----------------------
csv_path = fullfile(out_dir, 'Results_Mean.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, 'Function');
for a = 1:N_algos; fprintf(fid, ',%s_Mean,%s_Std,%s_Rank', ...
    algo_names{a}, algo_names{a}, algo_names{a}); end
fprintf(fid, '\n');
for f = 1:N_funcs
    fprintf(fid, '%s', func_names{f});
    for a = 1:N_algos
        fprintf(fid, ',%.6e,%.6e,%d', ...
            Mean_table(f,a), Std_table(f,a), Rank_table(f,a));
    end
    fprintf(fid, '\n');
end
fprintf(fid, 'MeanRank');
for a = 1:N_algos; fprintf(fid, ',%.3f,,', Mean_Rank(a)); end
fprintf(fid, '\n');
fclose(fid);
fprintf('\nResults saved to: %s\n', csv_path);

%% --- Generate LaTeX Table Fragment ----------------------------------
latex_path = fullfile(out_dir, 'Results_LaTeX.tex');
fid = fopen(latex_path, 'w');
fprintf(fid, '\\begin{table}[htbp]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Comparison of Mean $\\pm$ Std (Best results in \\textbf{bold})}\n');
fprintf(fid, '\\label{tab:results}\n');
fprintf(fid, '\\resizebox{\\textwidth}{!}{\n');
col_spec = ['l', repmat('c', 1, N_algos)];
fprintf(fid, '\\begin{tabular}{%s}\n', col_spec);
fprintf(fid, '\\toprule\n');

% Header
fprintf(fid, 'Function');
for a = 1:N_algos; fprintf(fid, ' & %s', strrep(algo_names{a},'_','-')); end
fprintf(fid, ' \\\\\n\\midrule\n');

% Data rows
for f = 1:N_funcs
    fprintf(fid, '%s', func_names{f});
    for a = 1:N_algos
        if Rank_table(f,a) == 1
            fprintf(fid, ' & \\textbf{%.2e$\\pm$%.1e}', Mean_table(f,a), Std_table(f,a));
        else
            fprintf(fid, ' & %.2e$\\pm$%.1e', Mean_table(f,a), Std_table(f,a));
        end
    end
    fprintf(fid, ' \\\\\n');
end

fprintf(fid, '\\midrule\n');
fprintf(fid, 'Mean Rank');
for a = 1:N_algos
    if Mean_Rank(a) == min(Mean_Rank)
        fprintf(fid, ' & \\textbf{%.2f}', Mean_Rank(a));
    else
        fprintf(fid, ' & %.2f', Mean_Rank(a));
    end
end
fprintf(fid, ' \\\\\n');
fprintf(fid, '\\bottomrule\n\\end{tabular}}\n\\end{table}\n');
fclose(fid);
fprintf('LaTeX table saved to: %s\n', latex_path);
end
