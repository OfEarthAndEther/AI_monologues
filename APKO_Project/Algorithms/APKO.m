%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  APKO: Adaptive Pied Kingfisher Optimizer
%  Version 1.0 - Proposed Algorithm
%
%  Key Improvements over base PKO:
%    [1] Adaptive Beating Factor   : BF(t,D) - no longer fixed at 8
%    [2] Dynamic Crest Angle       : theta(t) - cosine-annealed schedule
%    [3] Cosine-decay PE schedule  : replaces unjustified linear decay
%    [4] Tent-OBL Initialization   : improved diversity; fixed singularity
%    [5] Adaptive Cauchy Mutation  : scale contracts with search progress
%    [6] Population Entropy Restart: information-theoretic stagnation trigger
%    [7] Ranked Commensalism       : top/bottom-half paired selection
%
%  Inputs:
%    Popsize      - number of search agents
%    Maxiteration - maximum iterations
%    LB, UB       - lower/upper bounds (scalar or vector)
%    Dim          - problem dimension
%    Fobj         - objective function handle
%
%  Outputs:
%    Best_fitness      - best solution fitness found
%    Best_position     - best solution vector found
%    Convergence_curve - fitness trace across iterations
%
%  Mathematical Derivations:
%    BF*(t,D) = BF0 * (1 + ln(D)/D) * exp(-gamma * t/T)
%      - BF0=8 is the biological baseline (Bouaouda, 2024)
%      - Dimension scaling term (1+ln(D)/D) reflects that larger
%        search spaces need more hovering patterns
%      - Exponential decay: heavier exploration early, focused late
%    
%    theta(t) = 2*pi*rand * (1 - t/T) + pi/2*sin(pi*t/T)
%      - Oscillatory component adds directional diversity
%      - Decaying random component ensures convergence
%
%    PE(t) = PEmax/2 * (1 + cos(pi * t/T))
%      - Cosine annealing ensures smooth probability transition
%      - Mathematically superior to linear decay (differentiable,
%        natural transition point at midpoint)
%
%    Adaptive Cauchy scale: sigma(t) = sigma0 * (1 - t/T)^2
%      - Contracts quadratically to prevent late-iteration disruption
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Best_fitness, Best_position, Convergence_curve] = ...
         APKO(Popsize, Maxiteration, LB, UB, Dim, Fobj)

tic;

%% ---- Hyperparameters ------------------------------------------------
BF0      = 8;       % Biological baseline beating factor (Bouaouda, 2024)
gamma_BF = 1.5;     % Beating factor decay exponent (tuned via CEC2017)
PEmax    = 0.5;     % Max predatory efficiency (kept from base PKO)
PEmin    = 0;       % Min predatory efficiency
sigma0   = 1.0;     % Initial Cauchy mutation scale
H_entr   = 0.15;    % Entropy ratio threshold for restart trigger
k_restart = 0.2;    % Fraction of worst agents to restart on stagnation
entropy_window = 10; % Iterations to average entropy over

%% ---- Initialisation: Tent-OBL --------------------------------------
X = TentOBL_Init(Popsize, Dim, UB, LB);  % Improved initialization

%% ---- Evaluate initial population -----------------------------------
Fitness = zeros(1, Popsize);
for i = 1:Popsize
    Fitness(i) = Fobj(X(i,:));
end

[~, sorted_idx] = sort(Fitness);
Best_position   = X(sorted_idx(1),:);
Best_fitness    = Fitness(sorted_idx(1));

Convergence_curve = zeros(1, Maxiteration);
Convergence_curve(1) = Best_fitness;

%% ---- Entropy history -----------------------------------------------
entropy_history = zeros(1, entropy_window);
entropy_history(:) = Inf;  % prime with large values

X_1 = zeros(Popsize, Dim);   % Proposed positions buffer
fitnessn = zeros(1, Popsize);

t = 1;

%% ---- Main Loop -----------------------------------------------------
while t < Maxiteration + 1

    %% [IMP-1] Adaptive Beating Factor
    %  Derivation: higher dim -> more hovering patterns needed
    %  Exponential decay: intensive early exploration, fine late
    BF = BF0 * (1 + log(Dim)/Dim) * exp(-gamma_BF * t/Maxiteration);
    BF = max(BF, 1.5);   % Floor: prevent BF from collapsing to < 1

    %% [IMP-2] Dynamic Crest Angle
    %  Oscillatory + decaying term spans multiple dive angles per phase
    Crest_angle = 2*pi*rand*(1 - t/Maxiteration) + ...
                  (pi/2)*sin(pi*t/Maxiteration);

    %% Decay parameter (convergence control, same as base PKO)
    o = exp(-t/Maxiteration)^2;

    %% [IMP-3] Cosine PE schedule (replaces linear decay)
    %  PE(t) = PEmax/2 * (1 + cos(pi*t/T))
    PE = (PEmax/2) * (1 + cos(pi * t/Maxiteration));
    PE = max(PE, PEmin);

    %% [IMP-5] Adaptive Cauchy scale (contracts with search progress)
    cauchy_scale = sigma0 * (1 - t/Maxiteration)^2;
    cauchy_scale = max(cauchy_scale, 1e-4);  % Numerical floor

    %% Rank population for commensalism (IMP-7)
    [~, rank_idx] = sort(Fitness);           % rank_idx(1) = best
    top_half    = rank_idx(1 : floor(Popsize/2));
    bottom_half = rank_idx(floor(Popsize/2)+1 : end);

    %% --- Phase 1 & 2: Hunting (Exploration + Exploitation) ----------
    for i = 1:Popsize
        if rand < 0.8   %% Exploration: Perching/Hovering
            % Select a random different agent j
            j = i;
            while j == i
                seed = randperm(Popsize);
                j = seed(1);
            end

            beatingRate = rand * (Fitness(j)) / (max(Fitness(i),1e-300));
            alpha = 2*randn(1,Dim) - 1;

            if rand < 0.5
                %% Perching: BF now adaptive
                T_val = beatingRate - ((t)^(1/BF) / (Maxiteration)^(1/BF));
                X_1(i,:) = X(i,:) + alpha .* T_val .* (X(j,:) - X(i,:));
            else
                %% Hovering: Dynamic crest angle
                T_val = (exp(1) - exp(((t-1)/Maxiteration)^(1/BF))) ...
                        * cos(Crest_angle);
                X_1(i,:) = X(i,:) + alpha .* T_val .* (X(j,:) - X(i,:));
            end

        else  %% Exploitation: Diving
            alpha = 2*randn(1,Dim) - 1;
            b = X(i,:) + o^2 * randn .* Best_position;
            HuntingAbility = rand * Fitness(i) / max(Best_fitness, 1e-300);
            X_1(i,:) = X(i,:) + HuntingAbility*o*alpha.*(b - Best_position);
        end
    end

    %% --- Boundary + Fitness Evaluation (Phase 1&2) ------------------
    for i = 1:Popsize
        X_1(i,:) = BoundaryFix(X_1(i,:), LB, UB);
        fitnessn(i) = Fobj(X_1(i,:));
        if fitnessn(i) < Fitness(i)
            Fitness(i)  = fitnessn(i);
            X(i,:)      = X_1(i,:);
        end
        if Fitness(i) < Best_fitness
            Best_fitness  = Fitness(i);
            Best_position = X(i,:);
        end
    end

    %% --- Phase 3: Ranked Commensalism (IMP-7) -----------------------
    for i = 1:Popsize
        alpha = 2*randn(1,Dim) - 1;
        if rand > (1 - PE)
            %% [IMP-7] Pick m from top half, n from bottom half
            m = top_half(randi(length(top_half)));
            n = bottom_half(randi(length(bottom_half)));
            X_1(i,:) = X(m,:) + o*alpha.*abs(X(i,:) - X(n,:));
        else
            X_1(i,:) = X(i,:);  % No update: keep position
        end

        X_1(i,:) = BoundaryFix(X_1(i,:), LB, UB);
        fitnessn(i) = Fobj(X_1(i,:));

        %% Adaptive Cauchy fallback if commensalism fails (from EPKO)
        if fitnessn(i) >= Fitness(i)
            cauchy_perturb = X(i,:) + cauchy_scale * tan(pi*(rand(1,Dim)-0.5));
            cauchy_perturb = BoundaryFix(cauchy_perturb, LB, UB);
            cauchy_fit = Fobj(cauchy_perturb);
            if cauchy_fit < Fitness(i)
                fitnessn(i) = cauchy_fit;
                X_1(i,:)   = cauchy_perturb;
            end
        end

        if fitnessn(i) < Fitness(i)
            Fitness(i)  = fitnessn(i);
            X(i,:)      = X_1(i,:);
        end
        if Fitness(i) < Best_fitness
            Best_fitness  = Fitness(i);
            Best_position = X(i,:);
        end
    end

    %% --- [IMP-6] Population Entropy Restart -------------------------
    %  H_pop = normalised Shannon entropy over fitness ranks
    %  When H_pop drops below threshold, restart worst k% of agents
    H_now = ComputePopulationEntropy(Fitness);
    entropy_history(mod(t-1, entropy_window)+1) = H_now;

    if t > entropy_window
        mean_entropy = mean(entropy_history);
        H_max_possible = log(Popsize);  % Max entropy for Popsize agents
        if (mean_entropy / H_max_possible) < H_entr
            % Stagnation detected: restart worst k% agents
            num_restart = max(1, round(k_restart * Popsize));
            [~, worst_idx] = sort(Fitness, 'descend');
            for r = 1:num_restart
                idx_r = worst_idx(r);
                % Restart toward global best with Gaussian perturbation
                X(idx_r,:) = Best_position + randn(1,Dim) .* ...
                             abs(UB - LB) * 0.1 * (1 - t/Maxiteration);
                X(idx_r,:) = BoundaryFix(X(idx_r,:), LB, UB);
                Fitness(idx_r) = Fobj(X(idx_r,:));
                if Fitness(idx_r) < Best_fitness
                    Best_fitness  = Fitness(idx_r);
                    Best_position = X(idx_r,:);
                end
            end
            % Reset entropy history after restart
            entropy_history(:) = Inf;
        end
    end

    Convergence_curve(t) = Best_fitness;
    t = t + 1;
end

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
