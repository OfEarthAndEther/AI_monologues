%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Plot_Convergence.m
%
%  Generates publication-quality convergence curve plots.
%  Each plot shows all algorithms' best-so-far fitness vs. iteration.
%
%  Usage:
%    Plot_Convergence(conv_curves, algo_names, func_name, out_dir)
%
%  Inputs:
%    conv_curves : N_algos x 1 cell array, each cell = (N_runs x N_iter)
%                 matrix of convergence data across all runs
%    algo_names  : cell array of algorithm names
%    func_name   : string - benchmark function name (for title/filename)
%    out_dir     : output directory (default: 'Results/Figures/')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Plot_Convergence(conv_curves, algo_names, func_name, out_dir)

if nargin < 4; out_dir = 'Results/Figures/'; end
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

N_algos = length(algo_names);

%% --- Color & marker scheme (publication quality) --------------------
colors = [
    0.00, 0.45, 0.70;   % Blue      - APKO (proposed)
    0.85, 0.33, 0.10;   % Red       - PKO base
    0.93, 0.69, 0.13;   % Orange    - IPKO
    0.49, 0.18, 0.56;   % Purple    - MPKO
    0.47, 0.67, 0.19;   % Green     - EPKO
    0.30, 0.75, 0.93;   % Cyan      - WOA
    0.64, 0.08, 0.18;   % Dark Red  - GWO
    0.50, 0.50, 0.50;   % Grey      - HHO
    0.00, 0.50, 0.00;   % Dark Green- SCA
    0.75, 0.00, 0.75;   % Magenta   - MPA
    0.25, 0.25, 0.75;   % Indigo    - AOA
    0.90, 0.60, 0.00;   % Amber     - GJO
];
% Extend colors if more algorithms exist
while size(colors,1) < N_algos
    colors(end+1,:) = rand(1,3); %#ok<AGROW>
end

line_styles = {'-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':'};
markers     = {'o','s','^','d','v','p','h','*','+','x','<','>'};
line_widths = [2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5];
% APKO gets thicker line to stand out
marker_interval = max(1, round(size(conv_curves{1},2)/20));   % Plot 20 markers

%% --- Compute median convergence (median is robust to outlier runs) --
median_curves = cell(N_algos, 1);
for a = 1:N_algos
    if size(conv_curves{a}, 1) == 1
        median_curves{a} = conv_curves{a};   % Single run
    else
        median_curves{a} = median(conv_curves{a}, 1);
    end
end
N_iter = length(median_curves{1});

%% --- Create figure --------------------------------------------------
fig = figure('Position', [100 100 900 600], 'Color', 'white');
hold on; grid on; box on;

handles = zeros(1, N_algos);
for a = 1:N_algos
    curve = median_curves{a};
    % Replace zeros/negatives with eps for log scale
    curve(curve <= 0) = eps;

    % Marker positions at regular intervals
    marker_idx = 1:marker_interval:N_iter;

    handles(a) = semilogy(1:N_iter, curve, ...
        'Color',     colors(a,:), ...
        'LineStyle', line_styles{mod(a-1,12)+1}, ...
        'LineWidth', line_widths(min(a, length(line_widths))), ...
        'Marker',    markers{mod(a-1,12)+1}, ...
        'MarkerSize', 6, ...
        'MarkerIndices', marker_idx, ...
        'MarkerFaceColor', colors(a,:), ...
        'DisplayName', algo_names{a});
end

%% --- Legend and labels ----------------------------------------------
lgd = legend(handles, algo_names, ...
    'Location', 'northeast', ...
    'FontSize', 10, ...
    'Interpreter', 'none');
lgd.Box = 'on';

xlabel('Iteration', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Best Fitness (log scale)', 'FontSize', 13, 'FontWeight', 'bold');
title_str = sprintf('Convergence Curve — %s', strrep(func_name,'_',' '));
title(title_str, 'FontSize', 14, 'FontWeight', 'bold');

set(gca, 'FontSize', 11, 'YScale', 'log', 'GridAlpha', 0.3);
xlim([1, N_iter]);

%% --- Save as PNG and PDF --------------------------------------------
fname_base = fullfile(out_dir, sprintf('Conv_%s', ...
    regexprep(func_name, '[^a-zA-Z0-9_]', '_')));

try
    saveas(fig, [fname_base, '.png']);
    print(fig, [fname_base, '.pdf'], '-dpdf', '-r300');
catch
    % PDF export may fail on some systems; PNG is always saved
end

fprintf('Convergence plot saved: %s.png\n', fname_base);
hold off;
close(fig);
end
