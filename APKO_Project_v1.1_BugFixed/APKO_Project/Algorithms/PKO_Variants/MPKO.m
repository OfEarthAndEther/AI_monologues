%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MPKO.m  --  Multi-Strategy Improved Pied Kingfisher Optimizer
%
%  Based on: Wang & Liu (2025), Cluster Computing, 28, 979
%  DOI: 10.1007/s10586-025-05722-1
%
%  Improvements over base PKO:
%    [1] Leap Strategy for global search
%    [2] Dynamic Adjustment Strategy (DAS) for phase probabilities
%    [3] Dynamic Foraging Strategy (DFS) in commensalism
%    [4] Adventure Exploration Phase (4th phase) - selective restart
%
%  NOTE: DAS fitness variance threshold is empirically set (GAP in SOTA).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Best_fitness, Best_position, Convergence_curve] = ...
         MPKO(Popsize, Maxiteration, LB, UB, Dim, Fobj)

tic;
BF = 8;
Crest_angles = 2*pi*rand;
var_threshold = 1e-6;   % DAS variance threshold (empirical -- known gap)
stagnation_limit = 10;  % Adventure phase trigger (empirical -- known gap)

X = Initialization(Popsize, Dim, UB, LB);
Fitness = zeros(1, Popsize);
Convergence_curve = zeros(1, Maxiteration);
for i = 1:Popsize; Fitness(i) = Fobj(X(i,:)); end

[~, si] = sort(Fitness);
Best_position = X(si(1),:);
Best_fitness  = Fitness(si(1));
Convergence_curve(1) = Best_fitness;

t = 1;
PEmax=0.5; PEmin=0;
X_1=zeros(Popsize,Dim); fitnessn=zeros(1,Popsize);
stag_count = zeros(1,Popsize);  % Per-agent stagnation counter
prev_fitness = Fitness;

while t < Maxiteration + 1
    o = exp(-t/Maxiteration)^2;

    %% DAS: Dynamic Adjustment Strategy
    fitness_var = var(Fitness);
    if fitness_var < var_threshold
        p_explore = 0.9;   % High variance = stagnated -> more explore
    else
        p_explore = 0.8;   % Default exploration probability
    end

    %% Leap neighbourhood radius (shrinks as population converges)
    diversity = mean(std(X, 0, 1));
    leap_radius = max(diversity, 0.01 * mean(abs(UB - LB)));

    for i = 1:Popsize
        if rand < p_explore   % DAS-modulated exploration
            j = i;
            while i==j; seed=randperm(Popsize); j=seed(1); end
            beatingRate = rand*(Fitness(j))/(Fitness(i)+eps);
            alpha = 2*randn(1,Dim)-1;

            if rand < 0.5
                %% Leap Strategy: occasional large-step jumps
                if rand < 0.15
                    leap_center = X(i,:) + leap_radius*randn(1,Dim);
                    X_1(i,:) = leap_center;
                else
                    T = beatingRate - ((t)^(1/BF)/(Maxiteration)^(1/BF));
                    X_1(i,:) = X(i,:) + alpha.*T.*(X(j,:)-X(i,:));
                end
            else
                T = (exp(1)-exp(((t-1)/Maxiteration)^(1/BF)))*cos(Crest_angles);
                X_1(i,:) = X(i,:) + alpha.*T.*(X(j,:)-X(i,:));
            end
        else
            alpha = 2*randn(1,Dim)-1;
            b = X(i,:) + o^2*randn.*Best_position;
            HA = rand*(Fitness(i))/(Best_fitness+eps);
            X_1(i,:) = X(i,:) + HA*o*alpha.*(b-Best_position);
        end
    end

    for i = 1:Popsize
        X_1(i,:) = ClampBounds(X_1(i,:), LB, UB);
        fitnessn(i) = Fobj(X_1(i,:));
        if fitnessn(i) < Fitness(i); Fitness(i)=fitnessn(i); X(i,:)=X_1(i,:); end
        if Fitness(i) < Best_fitness; Best_fitness=Fitness(i); Best_position=X(i,:); end
    end

    %% DFS: Dynamic Foraging Strategy (ranked commensalism)
    [~, rank_idx] = sort(Fitness);
    top_half   = rank_idx(1:floor(Popsize/2));
    bot_half   = rank_idx(floor(Popsize/2)+1:end);
    PE = PEmax-(PEmax-PEmin)*(t/Maxiteration);

    for i = 1:Popsize
        alpha = 2*randn(1,Dim)-1;
        if rand > (1-PE)
            m = top_half(randi(length(top_half)));
            n = bot_half(randi(length(bot_half)));
            X_1(i,:) = X(m,:) + o*alpha.*abs(X(i,:)-X(n,:));
        else; X_1(i,:) = X(i,:); end
        X_1(i,:) = ClampBounds(X_1(i,:), LB, UB);
        fitnessn(i) = Fobj(X_1(i,:));
        if fitnessn(i) < Fitness(i); Fitness(i)=fitnessn(i); X(i,:)=X_1(i,:); end
        if Fitness(i) < Best_fitness; Best_fitness=Fitness(i); Best_position=X(i,:); end
    end

    %% Adventure Exploration Phase (4th phase)
    for i = 1:Popsize
        if Fitness(i) >= prev_fitness(i) - eps
            stag_count(i) = stag_count(i) + 1;
        else
            stag_count(i) = 0;
        end
        if stag_count(i) >= stagnation_limit
            X(i,:) = Best_position + randn(1,Dim).*abs(UB-LB)*0.05;
            X(i,:) = ClampBounds(X(i,:), LB, UB);
            Fitness(i) = Fobj(X(i,:));
            stag_count(i) = 0;
            if Fitness(i) < Best_fitness
                Best_fitness=Fitness(i); Best_position=X(i,:);
            end
        end
    end
    prev_fitness = Fitness;

    Convergence_curve(t) = Best_fitness;
    t = t+1;
end
toc;
end

function X = Initialization(N, dim, ub, lb)
    if numel(ub)==1; X=rand(N,dim).*(ub-lb)+lb;
    else
        X=zeros(N,dim);
        for d=1:dim; X(:,d)=rand(N,1).*(ub(d)-lb(d))+lb(d); end
    end
end

function x = ClampBounds(x, lb, ub)
    FU=x>ub; FL=x<lb; x=(x.*(~(FU+FL)))+ub.*FU+lb.*FL;
end
