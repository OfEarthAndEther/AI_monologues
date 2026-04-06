%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  main_sota_comparison.m
%
%  TRACK 1: APKO vs Existing PKO Variants (SOTA Comparison)
%  Demonstrates improvement over the PKO research lineage.
%
%  Algorithms compared:
%    1. APKO  (proposed)
%    2. PKO   (Bouaouda et al., 2024 - base algorithm)
%    3. IPKO  (Cong et al., 2024 - maritime UAV)
%    4. MPKO  (Wang & Liu, 2025 - multi-strategy)
%    5. EPKO  (Benfeng et al., 2025 - expert systems)
%
%  This comparison specifically validates that the proposed improvements
%  provide genuine performance gains over the existing PKO research family.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;
setup_paths();

%% ---- Configuration -------------------------------------------------
N_RUNS   = 50;
MAX_FES  = 60000;
DIM      = 30;
POPSIZE  = 30;
MAX_ITER = floor(MAX_FES / POPSIZE);

QUICK_TEST = false;
if QUICK_TEST; N_RUNS=5; MAX_FES=5000; MAX_ITER=floor(MAX_FES/POPSIZE); end

OUT_DIR = 'Results/Track1_SOTA/';
if ~exist(OUT_DIR,'dir'); mkdir(OUT_DIR); end

%% ---- PKO Variant Registry ------------------------------------------
sota_handles = {
    @(P,T,L,U,D,F) APKO(P,T,L,U,D,F),   'APKO (Proposed)';
    @(P,T,L,U,D,F) PKO(P,T,L,U,D,F),    'PKO (Base)';
    @(P,T,L,U,D,F) IPKO(P,T,L,U,D,F),   'IPKO';
    @(P,T,L,U,D,F) MPKO(P,T,L,U,D,F),   'MPKO';
    @(P,T,L,U,D,F) EPKO(P,T,L,U,D,F),   'EPKO';
};
sota_funcs = sota_handles(:,1);
sota_names = sota_handles(:,2);
N_SOTA = length(sota_funcs);

fprintf('============================================\n');
fprintf('  APKO: Track 1 SOTA Comparison\n');
fprintf('  PKO variants : %d\n', N_SOTA);
fprintf('  Runs/func    : %d\n', N_RUNS);
fprintf('  Dimension    : %d\n', DIM);
fprintf('============================================\n\n');

%% ====================================================================
%%  CEC-2017 (primary benchmark for SOTA comparison)
%% ====================================================================
fprintf('--- SOTA: CEC-2017 Benchmark ---\n');
N_CEC17 = 29;
func_names_17 = arrayfun(@(i) sprintf('F%02d_CEC17',i), 1:N_CEC17, 'UniformOutput',false);

results_sota = cell(N_CEC17, N_SOTA);
conv_sota    = cell(N_CEC17, N_SOTA);

for f = 1:N_CEC17
    [Fobj, LB, UB] = CEC2017_Wrapper(f, DIM);
    fprintf('  CEC17-F%02d : ', f);
    for a = 1:N_SOTA
        run_fitness = zeros(N_RUNS, 1);
        run_conv    = zeros(N_RUNS, MAX_ITER);
        for r = 1:N_RUNS
            rng(r + a*5000 + f*500);
            try
                [best_f, ~, conv] = sota_funcs{a}(POPSIZE, MAX_ITER, LB, UB, DIM, Fobj);
                run_fitness(r) = best_f;
                run_conv(r,:)  = PadConv(conv, MAX_ITER);
            catch ME
                warning('%s failed on F%d r%d: %s', sota_names{a}, f, r, ME.message);
                run_fitness(r) = Inf; run_conv(r,:) = Inf;
            end
        end
        results_sota{f,a} = run_fitness;
        conv_sota{f,a}    = run_conv;
        fprintf('%s ', sota_names{a});
    end
    fprintf('\n');
end

Generate_Tables(results_sota, sota_names, func_names_17, [OUT_DIR 'CEC2017/']);
stat_pool = zeros(N_RUNS*N_CEC17, N_SOTA);
for f=1:N_CEC17
    rows=(f-1)*N_RUNS+1:f*N_RUNS;
    for a=1:N_SOTA; stat_pool(rows,a)=results_sota{f,a}; end
end
stat_results_sota = Statistical_Tests(stat_pool, sota_names);

fig_dir = [OUT_DIR 'CEC2017/Figures/'];
for f=1:N_CEC17
    Plot_Convergence(conv_sota(f,:)', sota_names, func_names_17{f}, fig_dir);
end

%% ====================================================================
%%  CEC-2014 (secondary benchmark)
%% ====================================================================
fprintf('\n--- SOTA: CEC-2014 Benchmark ---\n');
N_CEC14 = 30;
func_names_14 = arrayfun(@(i) sprintf('F%02d_CEC14',i), 1:N_CEC14, 'UniformOutput',false);

results_sota14 = cell(N_CEC14, N_SOTA);
conv_sota14    = cell(N_CEC14, N_SOTA);

for f = 1:N_CEC14
    [Fobj, LB, UB] = CEC2014_Wrapper(f, DIM);
    fprintf('  CEC14-F%02d : ', f);
    for a = 1:N_SOTA
        run_fitness = zeros(N_RUNS, 1);
        run_conv    = zeros(N_RUNS, MAX_ITER);
        for r = 1:N_RUNS
            rng(r + a*6000 + f*600);
            try
                [best_f, ~, conv] = sota_funcs{a}(POPSIZE, MAX_ITER, LB, UB, DIM, Fobj);
                run_fitness(r) = best_f;
                run_conv(r,:)  = PadConv(conv, MAX_ITER);
            catch ME
                run_fitness(r) = Inf; run_conv(r,:) = Inf;
            end
        end
        results_sota14{f,a} = run_fitness;
        conv_sota14{f,a}    = run_conv;
        fprintf('%s ', sota_names{a});
    end
    fprintf('\n');
end

Generate_Tables(results_sota14, sota_names, func_names_14, [OUT_DIR 'CEC2014/']);
fig_dir14 = [OUT_DIR 'CEC2014/Figures/'];
for f=1:N_CEC14
    Plot_Convergence(conv_sota14(f,:)', sota_names, func_names_14{f}, fig_dir14);
end

%% ====================================================================
%%  Engineering Problems: SOTA Track
%% ====================================================================
fprintf('\n--- SOTA: Engineering Problems ---\n');
eng_problems = {'TCS','PVD','WBD','SRD','TBT'};
N_ENG = length(eng_problems);
results_eng_sota = cell(N_ENG, N_SOTA);

for e = 1:N_ENG
    P = Engineering_Problems(eng_problems{e});
    fprintf('  %s: ', P.name);
    ITER_ENG = floor(MAX_FES/POPSIZE);
    for a = 1:N_SOTA
        run_fitness = zeros(N_RUNS,1);
        for r = 1:N_RUNS
            rng(r + a*7000 + e*700);
            try
                [best_f,~,~] = sota_funcs{a}(POPSIZE,ITER_ENG,P.LB,P.UB,P.Dim,P.Fobj);
                run_fitness(r) = best_f;
            catch; run_fitness(r) = Inf; end
        end
        results_eng_sota{e,a} = run_fitness;
        fprintf('%s ', sota_names{a});
    end
    fprintf('\n');
end
Generate_Tables(results_eng_sota, sota_names, eng_problems, [OUT_DIR 'Engineering/']);

%% ====================================================================
%%  APKO Component Contribution Analysis (Ablation Study)
%%  Tests the contribution of each improvement individually
%% ====================================================================
fprintf('\n--- Ablation Study: Component Analysis ---\n');
ablation_results = Run_Ablation_Study(N_RUNS, MAX_ITER, POPSIZE, DIM, OUT_DIR);

%% ---- Save and Summarize --------------------------------------------
save([OUT_DIR 'workspace_SOTA.mat'], ...
     'results_sota','results_sota14','results_eng_sota','stat_results_sota', ...
     'sota_names','func_names_17','func_names_14','ablation_results');

fprintf('\n============================================\n');
fprintf('  SOTA COMPARISON COMPLETE\n');
fprintf('  Results saved to: %s\n', OUT_DIR);
fprintf('============================================\n');
PrintImprovementSummary(results_sota, sota_names, func_names_17);
end


%% =====================================================================
%% Ablation Study: Enable/disable APKO components one at a time
%% =====================================================================
function ablation_results = Run_Ablation_Study(N_RUNS, MAX_ITER, POPSIZE, DIM, OUT_DIR)

% Test on representative subset: F1,F5,F10,F15,F20,F25 of CEC2017
test_funcs = [1, 5, 10, 15, 20, 25];
N_TEST = length(test_funcs);

fprintf('  Testing on CEC2017 functions: %s\n', mat2str(test_funcs));

variants = {
    'APKO-Full',         @(P,T,L,U,D,F) APKO(P,T,L,U,D,F);
    'w/o AdaptBF',       @(P,T,L,U,D,F) APKO_NoAdaptBF(P,T,L,U,D,F);
    'w/o DynAngle',      @(P,T,L,U,D,F) APKO_NoAngle(P,T,L,U,D,F);
    'w/o CosinePE',      @(P,T,L,U,D,F) APKO_NoCosPE(P,T,L,U,D,F);
    'w/o TentOBL',       @(P,T,L,U,D,F) APKO_NoTentOBL(P,T,L,U,D,F);
    'w/o CauchyEscape',  @(P,T,L,U,D,F) APKO_NoCauchy(P,T,L,U,D,F);
    'w/o EntropyRst',    @(P,T,L,U,D,F) APKO_NoEntropy(P,T,L,U,D,F);
    'w/o RankedComm',    @(P,T,L,U,D,F) APKO_NoRankedComm(P,T,L,U,D,F);
};

abl_names = variants(:,1);
abl_funcs = variants(:,2);
N_ABL = size(variants,1);

abl_data = cell(N_TEST, N_ABL);
for fi = 1:N_TEST
    f = test_funcs(fi);
    [Fobj, LB, UB] = CEC2017_Wrapper(f, DIM);
    fprintf('    F%02d: ', f);
    for a = 1:N_ABL
        run_fit = zeros(N_RUNS,1);
        for r = 1:N_RUNS
            rng(r + a*9000 + f*900);
            try
                [bf,~,~] = abl_funcs{a}(POPSIZE,MAX_ITER,LB,UB,DIM,Fobj);
                run_fit(r) = bf;
            catch; run_fit(r) = Inf; end
        end
        abl_data{fi,a} = run_fit;
        fprintf('%s ', abl_names{a});
    end
    fprintf('\n');
end

abl_funclabels = arrayfun(@(i) sprintf('CEC17-F%02d',test_funcs(i)), 1:N_TEST,'UniformOutput',false);
Generate_Tables(abl_data, abl_names, abl_funclabels, [OUT_DIR 'Ablation/']);
ablation_results.data = abl_data;
ablation_results.names = abl_names;
fprintf('  Ablation study complete.\n');
end


%% =====================================================================
%% APKO Ablation Variants (each disables exactly one improvement)
%% These are minimal wrappers; the full APKO.m serves as reference.
%% =====================================================================

% Variant 1: Fixed BF=8 (no adaptive beating factor)
function [bf,bp,cc] = APKO_NoAdaptBF(P,T,lb,ub,D,F)
    % Call APKO but override BF to be fixed inside
    % NOTE: For full ablation, a flag-based APKO is cleaner.
    % This approximation fixes BF by setting gamma=0 (no decay).
    [bf,bp,cc] = APKO_Flagged(P,T,lb,ub,D,F,'fix_BF',true);
end

function [bf,bp,cc] = APKO_NoAngle(P,T,lb,ub,D,F)
    [bf,bp,cc] = APKO_Flagged(P,T,lb,ub,D,F,'fix_angle',true);
end

function [bf,bp,cc] = APKO_NoCosPE(P,T,lb,ub,D,F)
    [bf,bp,cc] = APKO_Flagged(P,T,lb,ub,D,F,'linear_PE',true);
end

function [bf,bp,cc] = APKO_NoTentOBL(P,T,lb,ub,D,F)
    [bf,bp,cc] = APKO_Flagged(P,T,lb,ub,D,F,'uniform_init',true);
end

function [bf,bp,cc] = APKO_NoCauchy(P,T,lb,ub,D,F)
    [bf,bp,cc] = APKO_Flagged(P,T,lb,ub,D,F,'no_cauchy',true);
end

function [bf,bp,cc] = APKO_NoEntropy(P,T,lb,ub,D,F)
    [bf,bp,cc] = APKO_Flagged(P,T,lb,ub,D,F,'no_entropy',true);
end

function [bf,bp,cc] = APKO_NoRankedComm(P,T,lb,ub,D,F)
    [bf,bp,cc] = APKO_Flagged(P,T,lb,ub,D,F,'random_comm',true);
end


%% =====================================================================
%% APKO_Flagged: APKO with ablation flags
%%   Flags (each disables one improvement):
%%     fix_BF       : BF=8 fixed (no adaptive beating factor)
%%     fix_angle    : fixed random crest angle (no dynamic update)
%%     linear_PE    : linear PE decay (original PKO schedule)
%%     uniform_init : uniform random init (no Tent+OBL)
%%     no_cauchy    : no Cauchy mutation escape
%%     no_entropy   : no entropy-based restart
%%     random_comm  : uniform random commensalism (no ranking)
%% =====================================================================
function [Best_fitness, Best_position, Convergence_curve] = ...
         APKO_Flagged(Popsize, Maxiteration, LB, UB, Dim, Fobj, varargin)

% Parse flags
flags = struct('fix_BF',false,'fix_angle',false,'linear_PE',false, ...
               'uniform_init',false,'no_cauchy',false,'no_entropy',false, ...
               'random_comm',false);
for k = 1:2:length(varargin)
    flags.(varargin{k}) = varargin{k+1};
end

BF0=8; gamma_BF=1.5; PEmax=0.5; PEmin=0;
sigma0=1.0; H_entr=0.15; k_restart=0.2; entropy_window=10;

if flags.uniform_init
    X = rand(Popsize,Dim).*(UB-LB)+LB;
else
    X = TentOBL_Init(Popsize,Dim,UB,LB);
end

Fitness = zeros(1,Popsize);
for i=1:Popsize; Fitness(i)=Fobj(X(i,:)); end
[~,si]=sort(Fitness); Best_position=X(si(1),:); Best_fitness=Fitness(si(1));
Convergence_curve=zeros(1,Maxiteration);
Convergence_curve(1)=Best_fitness;
X_1=zeros(Popsize,Dim); fitnessn=zeros(1,Popsize);
entropy_history=zeros(1,entropy_window); entropy_history(:)=Inf;

if flags.fix_angle
    Fixed_Crest = 2*pi*rand;
end

t=1;
while t < Maxiteration+1
    if flags.fix_BF
        BF = 8;
    else
        BF = max(BF0*(1+log(Dim)/Dim)*exp(-gamma_BF*t/Maxiteration), 1.5);
    end

    if flags.fix_angle
        Crest_angle = Fixed_Crest;
    else
        Crest_angle = 2*pi*rand*(1-t/Maxiteration)+(pi/2)*sin(pi*t/Maxiteration);
    end

    o = exp(-t/Maxiteration)^2;

    if flags.linear_PE
        PE = PEmax-(PEmax-PEmin)*(t/Maxiteration);
    else
        PE = max((PEmax/2)*(1+cos(pi*t/Maxiteration)), PEmin);
    end

    cauchy_scale = sigma0*(1-t/Maxiteration)^2;
    cauchy_scale = max(cauchy_scale,1e-4);

    if ~flags.random_comm
        [~,rank_idx]=sort(Fitness);
        top_half=rank_idx(1:floor(Popsize/2));
        bot_half=rank_idx(floor(Popsize/2)+1:end);
    end

    for i=1:Popsize
        if rand<0.8
            j=i; while j==i; s=randperm(Popsize); j=s(1); end
            beatingRate=rand*(Fitness(j))/(max(Fitness(i),1e-300));
            alpha=2*randn(1,Dim)-1;
            if rand<0.5
                T_val=beatingRate-((t)^(1/BF)/(Maxiteration)^(1/BF));
                X_1(i,:)=X(i,:)+alpha.*T_val.*(X(j,:)-X(i,:));
            else
                T_val=(exp(1)-exp(((t-1)/Maxiteration)^(1/BF)))*cos(Crest_angle);
                X_1(i,:)=X(i,:)+alpha.*T_val.*(X(j,:)-X(i,:));
            end
        else
            alpha=2*randn(1,Dim)-1;
            b=X(i,:)+o^2*randn.*Best_position;
            HA=rand*Fitness(i)/max(Best_fitness,1e-300);
            X_1(i,:)=X(i,:)+HA*o*alpha.*(b-Best_position);
        end
    end

    for i=1:Popsize
        X_1(i,:)=Clamp(X_1(i,:),LB,UB); fitnessn(i)=Fobj(X_1(i,:));
        if fitnessn(i)<Fitness(i); Fitness(i)=fitnessn(i); X(i,:)=X_1(i,:); end
        if Fitness(i)<Best_fitness; Best_fitness=Fitness(i); Best_position=X(i,:); end
    end

    for i=1:Popsize
        alpha=2*randn(1,Dim)-1;
        if rand>(1-PE)
            if flags.random_comm
                m=randi(Popsize); n=randi(Popsize);
            else
                m=top_half(randi(length(top_half)));
                n=bot_half(randi(length(bot_half)));
            end
            X_1(i,:)=X(m,:)+o*alpha.*abs(X(i,:)-X(n,:));
            X_1(i,:)=Clamp(X_1(i,:),LB,UB); fitnessn(i)=Fobj(X_1(i,:));
            if fitnessn(i)>=Fitness(i) && ~flags.no_cauchy
                cp=X(i,:)+cauchy_scale*tan(pi*(rand(1,Dim)-0.5));
                cp=Clamp(cp,LB,UB); cf=Fobj(cp);
                if cf<Fitness(i); fitnessn(i)=cf; X_1(i,:)=cp; end
            end
        else; X_1(i,:)=X(i,:); fitnessn(i)=Fitness(i); end
        if fitnessn(i)<Fitness(i); Fitness(i)=fitnessn(i); X(i,:)=X_1(i,:); end
        if Fitness(i)<Best_fitness; Best_fitness=Fitness(i); Best_position=X(i,:); end
    end

    if ~flags.no_entropy
        H_now=EntH(Fitness); entropy_history(mod(t-1,entropy_window)+1)=H_now;
        if t>entropy_window && (mean(entropy_history)/log(Popsize))<H_entr
            num_r=max(1,round(k_restart*Popsize)); [~,wi]=sort(Fitness,'descend');
            for r=1:num_r
                idx=wi(r); X(idx,:)=Best_position+randn(1,Dim).*abs(UB-LB)*0.1*(1-t/Maxiteration);
                X(idx,:)=Clamp(X(idx,:),LB,UB); Fitness(idx)=Fobj(X(idx,:));
                if Fitness(idx)<Best_fitness; Best_fitness=Fitness(idx); Best_position=X(idx,:); end
            end
            entropy_history(:)=Inf;
        end
    end

    Convergence_curve(t)=Best_fitness; t=t+1;
end
end

function x=Clamp(x,lb,ub); FU=x>ub;FL=x<lb; x=(x.*(~(FU+FL)))+ub.*FU+lb.*FL; end
function H=EntH(F); N=length(F); mn=min(F); mx=max(F); if abs(mx-mn)<1e-300; H=0; return; end; fn=(F-mn)/(mx-mn); nb=max(3,round(sqrt(N))); [c,~]=histcounts(fn,nb); p=c(c>0)/N; H=-sum(p.*log(p)); end

%% =====================================================================
%% Print improvement summary: how many functions APKO ranks 1st
%% =====================================================================
function PrintImprovementSummary(results, algo_names, func_names)
    N_funcs = size(results,1);
    N_algos = size(results,2);
    fprintf('\n--- Win/Tie/Loss Summary: APKO vs Each Variant ---\n');
    apko_means = cellfun(@mean, results(:,1));
    for a = 2:N_algos
        comp_means = cellfun(@mean, results(:,a));
        wins  = sum(apko_means < comp_means);
        ties  = sum(abs(apko_means - comp_means) < 1e-10);
        losses= sum(apko_means > comp_means);
        fprintf('  APKO vs %-15s : W=%2d / T=%2d / L=%2d out of %d functions\n', ...
                algo_names{a}, wins, ties, losses, N_funcs);
    end
end

%% Helper: Pad or trim convergence curve to target length
function c = PadConv(conv, target)
    if length(conv) >= target
        c = conv(1:target);
    else
        c = [conv, repmat(conv(end), 1, target-length(conv))];
    end
end
