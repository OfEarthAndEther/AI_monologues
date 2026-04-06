%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  IPKO.m  --  Improved Pied Kingfisher Optimizer
%
%  Based on: Cong et al. (2024), Applied Sciences, 14(24), 11816
%  DOI: 10.3390/app142411816
%
%  Improvements over base PKO:
%    [1] Refraction Opposite-Based Learning (ROBL) initialization
%    [2] Variable Spiral Sine-Cosine Search (VSSC) in exploitation
%    [3] Cauchy mutation for local optima escape
%
%  NOTE: Refraction index n_r is FIXED (GAP-M6 in SOTA analysis).
%        This is a known limitation used for comparison purposes.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Best_fitness, Best_position, Convergence_curve] = ...
         IPKO(Popsize, Maxiteration, LB, UB, Dim, Fobj)

tic;
n_r = 1.5;   % Refraction index (fixed -- see GAP-M6)
BF  = 8;
Crest_angles = 2*pi*rand;

%% ROBL Initialization
X = ROBL_Init(Popsize, Dim, UB, LB, n_r);

Fitness = zeros(1, Popsize);
Convergence_curve = zeros(1, Maxiteration);
for i = 1:Popsize
    Fitness(i) = Fobj(X(i,:));
end

[~, si] = sort(Fitness);
Best_position = X(si(1),:);
Best_fitness  = Fitness(si(1));
Convergence_curve(1) = Best_fitness;

t    = 1;
PEmax = 0.5; PEmin = 0;
X_1   = zeros(Popsize, Dim);
fitnessn = zeros(1, Popsize);

while t < Maxiteration + 1
    o = exp(-t/Maxiteration)^2;

    for i = 1:Popsize
        if rand < 0.8   % Exploration (same as PKO)
            j = i;
            while i == j; seed = randperm(Popsize); j = seed(1); end
            beatingRate = rand*(Fitness(j))/(Fitness(i)+eps);
            alpha = 2*randn(1,Dim)-1;
            if rand < 0.5
                T = beatingRate - ((t)^(1/BF)/(Maxiteration)^(1/BF));
                X_1(i,:) = X(i,:) + alpha.*T.*(X(j,:)-X(i,:));
            else
                T = (exp(1)-exp(((t-1)/Maxiteration)^(1/BF)))*cos(Crest_angles);
                X_1(i,:) = X(i,:) + alpha.*T.*(X(j,:)-X(i,:));
            end
        else  % Exploitation: Variable Spiral Sine-Cosine
            r1 = rand; r2 = rand;
            b_spiral = 1;   % Spiral shape constant
            l_spiral = (r1 - 0.5) * 2;   % Random in [-1,1]
            D_best   = abs(Best_position - X(i,:));
            X_1(i,:) = D_best .* exp(b_spiral*l_spiral) .* ...
                       cos(2*pi*l_spiral) + Best_position + ...
                       r2 * sin(pi*r1) .* (Best_position - X(i,:));
        end
    end

    for i = 1:Popsize
        X_1(i,:) = ClampBounds(X_1(i,:), LB, UB);
        fitnessn(i) = Fobj(X_1(i,:));
        if fitnessn(i) < Fitness(i); Fitness(i)=fitnessn(i); X(i,:)=X_1(i,:); end
        if Fitness(i) < Best_fitness; Best_fitness=Fitness(i); Best_position=X(i,:); end
    end

    %% Cauchy mutation on worst 20%
    [~, worst_idx] = sort(Fitness,'descend');
    n_cauchy = max(1, round(0.2*Popsize));
    for k = 1:n_cauchy
        idx = worst_idx(k);
        X_cauchy = X(idx,:) + tan(pi*(rand(1,Dim)-0.5));
        X_cauchy = ClampBounds(X_cauchy, LB, UB);
        f_cauchy = Fobj(X_cauchy);
        if f_cauchy < Fitness(idx)
            Fitness(idx) = f_cauchy; X(idx,:) = X_cauchy;
        end
        if Fitness(idx) < Best_fitness
            Best_fitness=Fitness(idx); Best_position=X(idx,:);
        end
    end

    PE = PEmax - (PEmax-PEmin)*(t/Maxiteration);
    for i = 1:Popsize
        alpha = 2*randn(1,Dim)-1;
        if rand > (1-PE)
            X_1(i,:) = X(randi([1,Popsize]),:) + ...
                       o*alpha.*abs(X(i,:)-X(randi([1,Popsize]),:));
        else; X_1(i,:) = X(i,:); end
        X_1(i,:) = ClampBounds(X_1(i,:), LB, UB);
        fitnessn(i) = Fobj(X_1(i,:));
        if fitnessn(i) < Fitness(i); Fitness(i)=fitnessn(i); X(i,:)=X_1(i,:); end
        if Fitness(i) < Best_fitness; Best_fitness=Fitness(i); Best_position=X(i,:); end
    end

    Convergence_curve(t) = Best_fitness;
    t = t+1;
end
toc;
end

function X = ROBL_Init(N, dim, ub, lb, n_r)
    if numel(ub)==1; ub=ub*ones(1,dim); end
    if numel(lb)==1; lb=lb*ones(1,dim); end
    X_base = rand(N,dim).*(ub-lb) + lb;
    X_ref  = (lb+ub)/n_r - X_base/n_r;
    for d=1:dim
        X_ref(:,d) = min(max(X_ref(:,d), lb(d)), ub(d));
    end
    X_comb = [X_base; X_ref];
    % Select N with best diversity (simple random 50/50 selection)
    sel = randperm(2*N, N);
    X   = X_comb(sel, :);
end

function x = ClampBounds(x, lb, ub)
    FU=x>ub; FL=x<lb;
    x=(x.*(~(FU+FL)))+ub.*FU+lb.*FL;
end
