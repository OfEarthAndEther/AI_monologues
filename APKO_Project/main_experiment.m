%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  main_experiment.m
%
%  PRIMARY EXPERIMENT RUNNER
%  Track 2: APKO vs 7 recent NIA competitors
%
%  Settings (per instructor guidelines - Image 2):
%    - 60,000 FEs per function per run
%    - 50 independent runs per function
%    - CEC-2014 (30 functions), CEC-2017 (29 functions),
%      and 5 engineering problems
%    - Outputs: Mean, Std, Rank tables; Convergence curves; Statistical tests
%
%  SETUP BEFORE RUNNING:
%    1. Add all subfolders to path (run setup_paths.m first)
%    2. Compile CEC2014/CEC2017 MEX files in their respective folders
%    3. Set RUN_CEC14, RUN_CEC17, RUN_ENG flags as needed
%
%  Runtime estimate: ~12-24 hours for full experiment (50 runs x 59 funcs)
%  Use QUICK_TEST=true for a fast 5-run validation first.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;

%% ---- Path Setup ----------------------------------------------------
setup_paths();

%% ---- Configuration -------------------------------------------------
N_RUNS   = 50;      % Independent runs per function (per guidelines)
MAX_FES  = 60000;   % Max function evaluations per run (per guidelines)
DIM      = 30;      % Problem dimension (standard; also test DIM=10,20)

QUICK_TEST = false;  % Set true for fast 5-run debug (ignores N_RUNS)
if QUICK_TEST
    N_RUNS = 5;
    MAX_FES = 5000;
    fprintf('[QUICK TEST MODE] Runs=%d, MaxFEs=%d\n', N_RUNS, MAX_FES);
end

% Function evaluation budget -> iterations given population size
POPSIZE = 30;
MAX_ITER = floor(MAX_FES / POPSIZE);   % 60000/30 = 2000 iterations

% Which benchmark suites to run
RUN_CEC14 = true;
RUN_CEC17 = true;
RUN_ENG   = true;

OUT_DIR = 'Results/Track2_NIA/';
if ~exist(OUT_DIR, 'dir'); mkdir(OUT_DIR); end

%% ---- Algorithm Registry --------------------------------------------
% Column 1 MUST be APKO (proposed) for statistical testing convention
algo_handles = {
    @(P,T,L,U,D,F) APKO(P,T,L,U,D,F),   'APKO (Proposed)';
    @(P,T,L,U,D,F) WOA(P,T,L,U,D,F),    'WOA';
    @(P,T,L,U,D,F) GWO(P,T,L,U,D,F),    'GWO';
    @(P,T,L,U,D,F) HHO(P,T,L,U,D,F),    'HHO';
    @(P,T,L,U,D,F) SCA(P,T,L,U,D,F),    'SCA';
    @(P,T,L,U,D,F) MPA(P,T,L,U,D,F),    'MPA';
    @(P,T,L,U,D,F) AOA(P,T,L,U,D,F),    'AOA';
    @(P,T,L,U,D,F) GJO(P,T,L,U,D,F),    'GJO';
};
algo_funcs = algo_handles(:,1);
algo_names = algo_handles(:,2);
N_ALGOS = length(algo_funcs);

fprintf('============================================\n');
fprintf('  APKO: Track 2 NIA Comparison Experiment\n');
fprintf('  Algorithms  : %d\n', N_ALGOS);
fprintf('  Runs/func   : %d\n', N_RUNS);
fprintf('  Max FEs     : %d\n', MAX_FES);
fprintf('  Dimension   : %d\n', DIM);
fprintf('============================================\n\n');

%% ====================================================================
%%  SECTION 1: CEC-2014 Benchmark (30 functions)
%% ====================================================================
if RUN_CEC14
    fprintf('\n--- CEC-2014 Benchmark (D=%d) ---\n', DIM);
    N_CEC14 = 30;
    func_names_14 = arrayfun(@(i) sprintf('F%02d_CEC14',i), 1:N_CEC14, 'UniformOutput',false);

    results_14   = cell(N_CEC14, N_ALGOS);
    conv_14      = cell(N_CEC14, N_ALGOS);

    for f = 1:N_CEC14
        [Fobj, LB, UB] = CEC2014_Wrapper(f, DIM);
        fprintf('  CEC14-F%02d : ', f);
        for a = 1:N_ALGOS
            run_fitness  = zeros(N_RUNS, 1);
            run_conv     = zeros(N_RUNS, MAX_ITER);
            for r = 1:N_RUNS
                rng(r + a*1000 + f*100);   % Reproducible seeding
                try
                    [best_f, ~, conv] = algo_funcs{a}(POPSIZE, MAX_ITER, LB, UB, DIM, Fobj);
                    run_fitness(r) = best_f;
                    if length(conv) >= MAX_ITER
                        run_conv(r,:) = conv(1:MAX_ITER);
                    else
                        run_conv(r,:) = [conv, repmat(conv(end), 1, MAX_ITER-length(conv))];
                    end
                catch ME
                    warning('Algorithm %s failed on F%d run %d: %s', ...
                            algo_names{a}, f, r, ME.message);
                    run_fitness(r) = Inf;
                    run_conv(r,:)  = Inf;
                end
            end
            results_14{f,a} = run_fitness;
            conv_14{f,a}    = run_conv;
            fprintf('%s ', algo_names{a});
        end
        fprintf('\n');
    end

    % --- Generate outputs for CEC2014
    data_matrix_14 = cellfun(@mean, results_14);
    Generate_Tables(results_14, algo_names, func_names_14, [OUT_DIR 'CEC2014/']);

    stat_data_14 = zeros(N_RUNS, N_ALGOS, N_CEC14);
    for f=1:N_CEC14
        for a=1:N_ALGOS; stat_data_14(:,a,f)=results_14{f,a}; end
    end
    % Aggregate across all functions for global statistical test
    stat_pool_14 = reshape(permute(stat_data_14,[1,3,2]), N_RUNS*N_CEC14, N_ALGOS);
    stat_results_14 = Statistical_Tests(stat_pool_14, algo_names);

    % Convergence plots for all 30 functions
    fig_dir = [OUT_DIR 'CEC2014/Figures/'];
    for f = 1:N_CEC14
        curves_f = conv_14(f,:)';
        Plot_Convergence(curves_f, algo_names, func_names_14{f}, fig_dir);
    end

    save([OUT_DIR 'CEC2014/workspace_CEC14.mat'], ...
         'results_14','conv_14','stat_results_14','algo_names','func_names_14');
    fprintf('CEC2014 complete. Results saved.\n');
end

%% ====================================================================
%%  SECTION 2: CEC-2017 Benchmark (29 functions, F1-F29)
%% ====================================================================
if RUN_CEC17
    fprintf('\n--- CEC-2017 Benchmark (D=%d) ---\n', DIM);
    N_CEC17 = 29;
    func_names_17 = arrayfun(@(i) sprintf('F%02d_CEC17',i), 1:N_CEC17, 'UniformOutput',false);

    results_17 = cell(N_CEC17, N_ALGOS);
    conv_17    = cell(N_CEC17, N_ALGOS);

    for f = 1:N_CEC17
        [Fobj, LB, UB] = CEC2017_Wrapper(f, DIM);
        fprintf('  CEC17-F%02d : ', f);
        for a = 1:N_ALGOS
            run_fitness = zeros(N_RUNS, 1);
            run_conv    = zeros(N_RUNS, MAX_ITER);
            for r = 1:N_RUNS
                rng(r + a*2000 + f*200);
                try
                    [best_f, ~, conv] = algo_funcs{a}(POPSIZE, MAX_ITER, LB, UB, DIM, Fobj);
                    run_fitness(r) = best_f;
                    if length(conv) >= MAX_ITER
                        run_conv(r,:) = conv(1:MAX_ITER);
                    else
                        run_conv(r,:) = [conv, repmat(conv(end),1,MAX_ITER-length(conv))];
                    end
                catch ME
                    warning('Algorithm %s failed on CEC17-F%d run %d: %s', ...
                            algo_names{a}, f, r, ME.message);
                    run_fitness(r) = Inf; run_conv(r,:) = Inf;
                end
            end
            results_17{f,a} = run_fitness;
            conv_17{f,a}    = run_conv;
            fprintf('%s ', algo_names{a});
        end
        fprintf('\n');
    end

    Generate_Tables(results_17, algo_names, func_names_17, [OUT_DIR 'CEC2017/']);
    stat_pool_17 = zeros(N_RUNS*N_CEC17, N_ALGOS);
    for f=1:N_CEC17
        rows = (f-1)*N_RUNS+1 : f*N_RUNS;
        for a=1:N_ALGOS; stat_pool_17(rows,a)=results_17{f,a}; end
    end
    stat_results_17 = Statistical_Tests(stat_pool_17, algo_names);

    fig_dir = [OUT_DIR 'CEC2017/Figures/'];
    for f = 1:N_CEC17
        Plot_Convergence(conv_17(f,:)', algo_names, func_names_17{f}, fig_dir);
    end

    save([OUT_DIR 'CEC2017/workspace_CEC17.mat'], ...
         'results_17','conv_17','stat_results_17','algo_names','func_names_17');
    fprintf('CEC2017 complete. Results saved.\n');
end

%% ====================================================================
%%  SECTION 3: Engineering Design Problems (5 problems)
%% ====================================================================
if RUN_ENG
    fprintf('\n--- Engineering Design Problems ---\n');
    eng_problems = {'TCS', 'PVD', 'WBD', 'SRD', 'TBT'};
    N_ENG = length(eng_problems);

    results_eng = cell(N_ENG, N_ALGOS);
    conv_eng    = cell(N_ENG, N_ALGOS);
    best_solutions_eng = cell(N_ENG, N_ALGOS);

    for e = 1:N_ENG
        P = Engineering_Problems(eng_problems{e});
        fprintf('  %s (D=%d): ', P.name, P.Dim);

        % Adjust max iterations for engineering problem dimensions
        ITER_ENG = floor(MAX_FES / POPSIZE);

        for a = 1:N_ALGOS
            run_fitness  = zeros(N_RUNS, 1);
            run_conv     = zeros(N_RUNS, ITER_ENG);
            best_sol     = zeros(N_RUNS, P.Dim);
            for r = 1:N_RUNS
                rng(r + a*3000 + e*300);
                try
                    [best_f, best_x, conv] = algo_funcs{a}( ...
                        POPSIZE, ITER_ENG, P.LB, P.UB, P.Dim, P.Fobj);
                    run_fitness(r) = best_f;
                    best_sol(r,:)  = best_x;
                    if length(conv) >= ITER_ENG
                        run_conv(r,:) = conv(1:ITER_ENG);
                    else
                        run_conv(r,:) = [conv, repmat(conv(end),1,ITER_ENG-length(conv))];
                    end
                catch ME
                    warning('Engineering %s failed: %s', eng_problems{e}, ME.message);
                    run_fitness(r) = Inf; run_conv(r,:)=Inf;
                end
            end
            results_eng{e,a} = run_fitness;
            conv_eng{e,a}    = run_conv;
            % Store best solution across all runs
            [~, best_r] = min(run_fitness);
            best_solutions_eng{e,a} = best_sol(best_r,:);
            fprintf('%s ', algo_names{a});
        end
        fprintf('\n');
    end

    Generate_Tables(results_eng, algo_names, eng_problems, [OUT_DIR 'Engineering/']);

    % Print best solutions for engineering problems
    fprintf('\n--- Best Solutions Found ---\n');
    for e = 1:N_ENG
        P = Engineering_Problems(eng_problems{e});
        fprintf('\n%s:\n', P.name);
        for a = 1:N_ALGOS
            [bval, ~] = min(results_eng{e,a});
            fprintf('  %-25s : f* = %.6f | x* = %s\n', algo_names{a}, bval, ...
                    mat2str(best_solutions_eng{e,a}, 4));
        end
    end

    fig_dir = [OUT_DIR 'Engineering/Figures/'];
    for e = 1:N_ENG
        Plot_Convergence(conv_eng(e,:)', algo_names, eng_problems{e}, fig_dir);
    end

    save([OUT_DIR 'Engineering/workspace_Engineering.mat'], ...
         'results_eng','conv_eng','best_solutions_eng','algo_names','eng_problems');
    fprintf('\nEngineering problems complete.\n');
end

%% ====================================================================
%%  Final Summary
%% ====================================================================
fprintf('\n\n============================================\n');
fprintf('  EXPERIMENT COMPLETE\n');
fprintf('  All results saved to: %s\n', OUT_DIR);
fprintf('============================================\n');
fprintf('Next steps:\n');
fprintf('  1. Review Results/Track2_NIA/CEC2014/Results_Mean.csv\n');
fprintf('  2. Review Results/Track2_NIA/CEC2017/Results_Mean.csv\n');
fprintf('  3. Review convergence plots in */Figures/ folders\n');
fprintf('  4. Copy LaTeX tables from *Results_LaTeX.tex\n');
fprintf('  5. Run main_sota_comparison.m for Track 1 (PKO variants)\n');
