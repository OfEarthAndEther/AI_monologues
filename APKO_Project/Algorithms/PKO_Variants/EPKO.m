%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  EPKO.m  --  Enhanced Pied Kingfisher Optimizer
%
%  Based on: Benfeng et al. (2025), Expert Systems with Applications,
%            278, 127416. DOI: 10.1016/j.eswa.2025.127416
%
%  Five-mechanism enhancement over base PKO:
%    [1] Tent Chaos + OBL initialization
%    [2] Levy Flight perturbation in exploration (decaying scale)
%    [3] Randomized Sign Selection (RSS) in diving phase
%    [4] Enhanced commensalism with Levy fallback
%    [5] Nelder-Mead Simplex post-processing on top-k agents
%
%  NOTE: Levy sigma/beta are empirically tuned (GAP in SOTA analysis).
%        Simplex post-processing changes stochastic nature of algorithm.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Best_fitness, Best_position, Convergence_curve] = ...
         EPKO(Popsize, Maxiteration, LB, UB, Dim, Fobj)

tic;
BF = 8;
Crest_angles = 2*pi*rand;
levy_sigma = 1.0;    % Levy scale (empirical - known gap)
levy_beta  = 1.5;    % Levy exponent (standard Mantegna)
k_simplex  = 3;      % Top-k agents for Nelder-Mead refinement

%% Tent+OBL Initialization
X = TentOBL_Lite(Popsize, Dim, UB, LB);

Fitness = zeros(1, Popsize);
Convergence_curve = zeros(1, Maxiteration);
for i = 1:Popsize; Fitness(i) = Fobj(X(i,:)); end

[~, si] = sort(Fitness);
Best_position = X(si(1),:);
Best_fitness  = Fitness(si(1));
Convergence_curve(1) = Best_fitness;

t=1; PEmax=0.5; PEmin=0;
X_1=zeros(Popsize,Dim); fitnessn=zeros(1,Popsize);

while t < Maxiteration + 1
    o = exp(-t/Maxiteration)^2;
    % Decaying Levy scale: sigma * (t/T)^(-beta) capped to avoid explosion
    L_scale = levy_sigma * max((t/Maxiteration)^(-levy_beta), 0.01);
    L_scale = min(L_scale, 5.0);   % Prevent unbounded early steps

    for i = 1:Popsize
        if rand < 0.8
            j = i;
            while i==j; seed=randperm(Popsize); j=seed(1); end
            beatingRate = rand*(Fitness(j))/(Fitness(i)+eps);
            alpha = 2*randn(1,Dim)-1;

            %% [2] Levy augmented alpha
            levy_step = LevyFlight(Dim, levy_beta) * L_scale;

            if rand < 0.5
                T = beatingRate - ((t)^(1/BF)/(Maxiteration)^(1/BF));
                X_1(i,:) = X(i,:) + (alpha + levy_step).*T.*(X(j,:)-X(i,:));
            else
                T = (exp(1)-exp(((t-1)/Maxiteration)^(1/BF)))*cos(Crest_angles);
                X_1(i,:) = X(i,:) + (alpha + levy_step).*T.*(X(j,:)-X(i,:));
            end
        else
            alpha = 2*randn(1,Dim)-1;
            b = X(i,:) + o^2*randn.*Best_position;
            HA = rand*(Fitness(i))/(Best_fitness+eps);
            %% [3] RSS: randomize sign of perturbation term
            rss_sign = 2*(rand > 0.5) - 1;   % +1 or -1 with equal prob
            X_1(i,:) = X(i,:) + rss_sign*HA*o*alpha.*(b-Best_position);
        end
    end

    for i = 1:Popsize
        X_1(i,:) = ClampBounds(X_1(i,:), LB, UB);
        fitnessn(i) = Fobj(X_1(i,:));
        if fitnessn(i) < Fitness(i); Fitness(i)=fitnessn(i); X(i,:)=X_1(i,:); end
        if Fitness(i) < Best_fitness; Best_fitness=Fitness(i); Best_position=X(i,:); end
    end

    %% [4] Enhanced Commensalism with Levy fallback
    PE = PEmax-(PEmax-PEmin)*(t/Maxiteration);
    for i = 1:Popsize
        alpha = 2*randn(1,Dim)-1;
        if rand > (1-PE)
            X_1(i,:) = X(randi([1,Popsize]),:) + ...
                       o*alpha.*abs(X(i,:)-X(randi([1,Popsize]),:));
            X_1(i,:) = ClampBounds(X_1(i,:), LB, UB);
            fitnessn(i) = Fobj(X_1(i,:));
            if fitnessn(i) >= Fitness(i)
                % Levy fallback
                levy_fb = X(i,:) + LevyFlight(Dim, levy_beta)*L_scale;
                levy_fb = ClampBounds(levy_fb, LB, UB);
                f_levy  = Fobj(levy_fb);
                if f_levy < Fitness(i)
                    fitnessn(i) = f_levy; X_1(i,:) = levy_fb;
                end
            end
        else
            X_1(i,:)  = X(i,:);
            fitnessn(i) = Fitness(i);
        end
        if fitnessn(i) < Fitness(i); Fitness(i)=fitnessn(i); X(i,:)=X_1(i,:); end
        if Fitness(i) < Best_fitness; Best_fitness=Fitness(i); Best_position=X(i,:); end
    end

    %% [5] Nelder-Mead Simplex refinement on top-k agents
    if mod(t, 20) == 0   % Apply every 20 iterations (computational budget)
        [~, ri] = sort(Fitness);
        for k = 1:min(k_simplex, Popsize)
            idx = ri(k);
            [x_nm, f_nm] = SimplexStep(X(idx,:), Fitness(idx), ...
                                        Best_position, LB, UB, Fobj);
            if f_nm < Fitness(idx)
                X(idx,:) = x_nm; Fitness(idx) = f_nm;
                if f_nm < Best_fitness
                    Best_fitness=f_nm; Best_position=x_nm;
                end
            end
        end
    end

    Convergence_curve(t) = Best_fitness;
    t = t+1;
end
toc;
end

%% Levy flight using Mantegna algorithm
function step = LevyFlight(dim, beta)
    sigma_u = (gamma(1+beta)*sin(pi*beta/2) / ...
              (gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u = randn(1,dim) * sigma_u;
    v = randn(1,dim);
    step = u ./ (abs(v).^(1/beta));
    step = min(max(step, -5), 5);   % Clamp extreme values
end

%% Simplified 1-step simplex refinement toward best (deterministic)
function [x_out, f_out] = SimplexStep(x_in, f_in, x_best, lb, ub, Fobj)
    alpha_nm = 0.5;   % Contraction coefficient
    x_out = x_in + alpha_nm*(x_best - x_in);
    x_out = ClampBounds(x_out, lb, ub);
    f_out = Fobj(x_out);
    if f_out >= f_in; x_out = x_in; f_out = f_in; end
end

function X = TentOBL_Lite(N, dim, ub, lb)
    if numel(ub)==1; ub=ub*ones(1,dim); end
    if numel(lb)==1; lb=lb*ones(1,dim); end
    X_base = rand(N,dim).*(ub-lb)+lb;
    X_obl  = (lb+ub) - X_base;
    for d=1:dim; X_obl(:,d)=min(max(X_obl(:,d),lb(d)),ub(d)); end
    X_all  = [X_base; X_obl];
    idx    = randperm(2*N, N);
    X      = X_all(idx,:);
end

function x = ClampBounds(x, lb, ub)
    FU=x>ub; FL=x<lb; x=(x.*(~(FU+FL)))+ub.*FU+lb.*FL;
end
