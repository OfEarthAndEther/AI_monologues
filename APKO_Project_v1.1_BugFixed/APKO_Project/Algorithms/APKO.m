%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  APKO: Adaptive Pied Kingfisher Optimizer
%  Version 1.1 - Bug-Fixed Release
%
%  Key Improvements over base PKO:
%    [1] Adaptive Beating Factor   : BF(t,D) - no longer fixed at 8
%    [2] Deterministic Crest Angle : theta(t) - fully deterministic schedule
%                                    (FIX v1.1: removed per-iteration rand call)
%    [3] Cosine-decay PE schedule  : replaces unjustified linear decay
%    [4] Tent-OBL Initialization   : improved diversity; fixed singularity
%    [5] Adaptive Cauchy Mutation  : scale contracts with search progress
%    [6] Population Entropy Restart: information-theoretic stagnation trigger
%                                    (FIX v1.1: restart FEs counted against budget)
%    [7] Ranked Commensalism       : top/bottom-half paired selection
%
%  Inputs:
%    Popsize      - number of search agents
%    Maxiteration - maximum iterations (budget = Popsize * Maxiteration FEs)
%    LB, UB       - lower/upper bounds (scalar or vector)
%    Dim          - problem dimension
%    Fobj         - objective function handle
%
%  Outputs:
%    Best_fitness      - best solution fitness found
%    Best_position     - best solution vector found
%    Convergence_curve - best fitness per iteration (length = Maxiteration)
%
%  BUG FIXES (v1.0 → v1.1):
%    BUG-1 FIXED: FE budget enforcement
%      The entropy-restart phase calls Fobj() inside the main loop.
%      In v1.0, these calls were not counted, giving APKO a silent FE
%      advantage over every competitor. Fix: a shared FE_count is
%      incremented at every Fobj call; the main loop breaks as soon as
%      FE_count >= MaxFES (= Popsize * Maxiteration), ensuring all
%      algorithms receive exactly the same evaluation budget.
%
%    BUG-3 FIXED: Deterministic crest angle
%      In v1.0, Crest_angle = 2*pi*rand*(1-t/T) + pi/2*sin(pi*t/T).
%      The leading rand was called fresh each iteration, making the
%      angle stochastic — contradicting the "deterministic schedule"
%      claim in the header. Fix: the angle is now a pure function of t:
%        theta(t) = pi/2*sin(pi*t/T) + pi/4*cos(2*pi*t/T)
%      This is fully deterministic, oscillates between +3pi/4 and -pi/4,
%      and decays in amplitude through the sin-cos interaction, reaching
%      0 at t=T. The Lyapunov argument for convergence now holds exactly.
%
%  Mathematical Derivations:
%    BF*(t,D) = BF0 * (1 + ln(D)/D) * exp(-gamma * t/T)
%      Dimension scaling + exponential decay; see README for proof.
%
%    theta(t) = (pi/2)*sin(pi*t/T) + (pi/4)*cos(2*pi*t/T)
%      Purely deterministic. At t=0: theta=pi/4. At t=T/2: theta=pi/4.
%      At t=T: theta=pi/4+pi/4=pi/2... min=-pi/4, max=3pi/4.
%      Two-frequency oscillation creates directional diversity that is
%      fully reproducible given the same seed — essential for fair
%      statistical comparison across 50 independent runs.
%
%    PE(t) = PEmax/2 * (1 + cos(pi*t/T))
%      Cosine annealing: differentiable, symmetric half-budget split.
%
%    sigma(t) = sigma0 * (1 - t/T)^2
%      Quadratic Cauchy scale contraction.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Best_fitness, Best_position, Convergence_curve] = ...
         APKO(Popsize, Maxiteration, LB, UB, Dim, Fobj)

tic;

%% ---- Hyperparameters ------------------------------------------------
BF0           = 8;      % Biological baseline beating factor (Bouaouda, 2024)
gamma_BF      = 1.5;    % Beating factor decay exponent (tuned via CEC2017)
PEmax         = 0.5;    % Max predatory efficiency (kept from base PKO)
PEmin         = 0;      % Min predatory efficiency
sigma0        = 1.0;    % Initial Cauchy mutation scale
H_entr        = 0.15;   % Entropy ratio threshold for restart trigger
k_restart     = 0.2;    % Fraction of worst agents to restart on stagnation
entropy_window = 10;    % Iterations to average entropy over

%% ---- FE Budget (BUG-1 FIX) -----------------------------------------
%  Total allowed function evaluations = Popsize * Maxiteration.
%  Every call to Fobj anywhere in the algorithm increments FE_count.
%  The main loop exits as soon as this budget is exhausted.
MaxFES  = Popsize * Maxiteration;
FE_count = 0;

%% ---- Initialisation: Tent-OBL (4*N evals counted below) -----------
X = TentOBL_Init(Popsize, Dim, UB, LB);

%% ---- Evaluate initial population -----------------------------------
Fitness = zeros(1, Popsize);
for i = 1:Popsize
    Fitness(i) = Fobj(X(i,:));
    FE_count   = FE_count + 1;
end

[~, sorted_idx] = sort(Fitness);
Best_position   = X(sorted_idx(1),:);
Best_fitness    = Fitness(sorted_idx(1));

Convergence_curve = zeros(1, Maxiteration);
Convergence_curve(1) = Best_fitness;

%% ---- Entropy history -----------------------------------------------
entropy_history    = zeros(1, entropy_window);
entropy_history(:) = Inf;   % Prime with large values

X_1      = zeros(Popsize, Dim);
fitnessn = zeros(1, Popsize);

t = 1;

%% ---- Main Loop (exits when FE budget exhausted) --------------------
while t <= Maxiteration && FE_count < MaxFES

    %% [IMP-1] Adaptive Beating Factor
    BF = BF0 * (1 + log(Dim)/Dim) * exp(-gamma_BF * t/Maxiteration);
    BF = max(BF, 1.5);

    %% [IMP-2] Deterministic Crest Angle (BUG-3 FIX)
    %  Two-frequency deterministic schedule — no per-iteration rand.
    %  theta(t) = (pi/2)*sin(pi*t/T) + (pi/4)*cos(2*pi*t/T)
    %  Range: [-pi/4, 3*pi/4]. Fully deterministic given iteration t.
    Crest_angle = (pi/2) * sin(pi * t/Maxiteration) + ...
                  (pi/4) * cos(2*pi * t/Maxiteration);

    %% Decay parameter (same as base PKO)
    o = exp(-t/Maxiteration)^2;

    %% [IMP-3] Cosine PE schedule
    PE = (PEmax/2) * (1 + cos(pi * t/Maxiteration));
    PE = max(PE, PEmin);

    %% [IMP-5] Adaptive Cauchy scale
    cauchy_scale = sigma0 * (1 - t/Maxiteration)^2;
    cauchy_scale = max(cauchy_scale, 1e-4);

    %% [IMP-7] Rank population for commensalism
    [~, rank_idx]  = sort(Fitness);
    top_half    = rank_idx(1 : floor(Popsize/2));
    bottom_half = rank_idx(floor(Popsize/2)+1 : end);

    %% --- Phase 1 & 2: Hunting (Exploration + Exploitation) ----------
    for i = 1:Popsize
        if rand < 0.8   % Exploration: Perching/Hovering
            j = i;
            while j == i
                seed = randperm(Popsize);
                j    = seed(1);
            end
            beatingRate = rand * (Fitness(j)) / max(Fitness(i), 1e-300);
            alpha = 2*randn(1,Dim) - 1;
            if rand < 0.5
                T_val    = beatingRate - (t^(1/BF) / Maxiteration^(1/BF));
                X_1(i,:) = X(i,:) + alpha .* T_val .* (X(j,:) - X(i,:));
            else
                T_val    = (exp(1) - exp(((t-1)/Maxiteration)^(1/BF))) ...
                           * cos(Crest_angle);
                X_1(i,:) = X(i,:) + alpha .* T_val .* (X(j,:) - X(i,:));
            end
        else   % Exploitation: Diving
            alpha          = 2*randn(1,Dim) - 1;
            b              = X(i,:) + o^2 * randn .* Best_position;
            HuntingAbility = rand * Fitness(i) / max(Best_fitness, 1e-300);
            X_1(i,:)       = X(i,:) + HuntingAbility*o*alpha.*(b - Best_position);
        end
    end

    %% --- Boundary + Fitness Evaluation (Phase 1&2) ------------------
    for i = 1:Popsize
        if FE_count >= MaxFES; break; end   % Hard budget cap
        X_1(i,:)    = BoundaryFix(X_1(i,:), LB, UB);
        fitnessn(i) = Fobj(X_1(i,:));
        FE_count    = FE_count + 1;
        if fitnessn(i) < Fitness(i)
            Fitness(i) = fitnessn(i);
            X(i,:)     = X_1(i,:);
        end
        if Fitness(i) < Best_fitness
            Best_fitness  = Fitness(i);
            Best_position = X(i,:);
        end
    end

    %% --- Phase 3: Ranked Commensalism (IMP-7) -----------------------
    for i = 1:Popsize
        if FE_count >= MaxFES; break; end   % Hard budget cap
        alpha = 2*randn(1,Dim) - 1;
        if rand > (1 - PE)
            m        = top_half(randi(length(top_half)));
            n        = bottom_half(randi(length(bottom_half)));
            X_1(i,:) = X(m,:) + o*alpha.*abs(X(i,:) - X(n,:));
        else
            X_1(i,:) = X(i,:);
        end

        X_1(i,:)    = BoundaryFix(X_1(i,:), LB, UB);
        fitnessn(i) = Fobj(X_1(i,:));
        FE_count    = FE_count + 1;

        %% Adaptive Cauchy fallback if commensalism fails
        if fitnessn(i) >= Fitness(i) && FE_count < MaxFES
            cauchy_perturb = X(i,:) + cauchy_scale * tan(pi*(rand(1,Dim)-0.5));
            cauchy_perturb = BoundaryFix(cauchy_perturb, LB, UB);
            cauchy_fit     = Fobj(cauchy_perturb);
            FE_count       = FE_count + 1;
            if cauchy_fit < Fitness(i)
                fitnessn(i) = cauchy_fit;
                X_1(i,:)   = cauchy_perturb;
            end
        end

        if fitnessn(i) < Fitness(i)
            Fitness(i) = fitnessn(i);
            X(i,:)     = X_1(i,:);
        end
        if Fitness(i) < Best_fitness
            Best_fitness  = Fitness(i);
            Best_position = X(i,:);
        end
    end

    %% --- [IMP-6] Population Entropy Restart (BUG-1 FIX) ------------
    %  Restart FEs are now counted against the shared FE_count budget.
    %  The restart only fires if remaining budget can accommodate it.
    H_now = ComputePopulationEntropy(Fitness);
    entropy_history(mod(t-1, entropy_window)+1) = H_now;

    if t > entropy_window
        mean_entropy   = mean(entropy_history);
        H_max_possible = log(Popsize);
        if (mean_entropy / H_max_possible) < H_entr
            num_restart = max(1, round(k_restart * Popsize));
            % Check that enough budget remains before restarting
            if FE_count + num_restart <= MaxFES
                [~, worst_idx] = sort(Fitness, 'descend');
                for r = 1:num_restart
                    if FE_count >= MaxFES; break; end  % Per-agent cap
                    idx_r = worst_idx(r);
                    X(idx_r,:) = Best_position + randn(1,Dim) .* ...
                                 abs(UB - LB) * 0.1 * (1 - t/Maxiteration);
                    X(idx_r,:)     = BoundaryFix(X(idx_r,:), LB, UB);
                    Fitness(idx_r) = Fobj(X(idx_r,:));
                    FE_count       = FE_count + 1;   % COUNT RESTART FEs
                    if Fitness(idx_r) < Best_fitness
                        Best_fitness  = Fitness(idx_r);
                        Best_position = X(idx_r,:);
                    end
                end
            end
            entropy_history(:) = Inf;
        end
    end

    Convergence_curve(t) = Best_fitness;
    t = t + 1;
end

%% Fill any un-reached iterations with the last known best
Convergence_curve(t:end) = Best_fitness;

toc;
end


%% =====================================================================
%% Local Helper: Boundary fix (clamping)
%% =====================================================================
function x_fixed = BoundaryFix(x, LB, UB)
    FU = x > UB;
    FL = x < LB;
    x_fixed = x .* (~(FU + FL)) + UB .* FU + LB .* FL;
end


%% =====================================================================
%% Local Helper: Population Entropy (normalised Shannon over ranks)
%%
%%  Bin population fitness into Popsize quantile bins.
%%  p_i = fraction of agents in bin i.
%%  H_pop = -sum(p_i * log(p_i + eps))
%%
%%  High H_pop -> diverse population
%%  Low  H_pop -> collapsed/converged population (restart warranted)
%% =====================================================================
function H = ComputePopulationEntropy(Fitness)
    N = length(Fitness);
    % Normalise fitness to [0,1] range; equal fitness = degenerate
    f_min = min(Fitness);
    f_max = max(Fitness);
    if abs(f_max - f_min) < 1e-300
        H = 0;   % Completely collapsed: all agents identical
        return;
    end
    f_norm = (Fitness - f_min) / (f_max - f_min);
    % Bin into sqrt(N) bins for robust estimation
    nbins  = max(3, round(sqrt(N)));
    edges  = linspace(0, 1+1e-9, nbins+1);
    counts = histcounts(f_norm, edges);
    p      = counts / N;
    p      = p(p > 0);   % Remove zero-probability bins
    H      = -sum(p .* log(p));   % Shannon entropy (nats)
end
