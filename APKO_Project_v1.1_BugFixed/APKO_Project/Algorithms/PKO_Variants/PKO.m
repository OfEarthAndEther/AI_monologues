%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  PKO.m  --  Pied Kingfisher Optimizer  (Base Algorithm)
%
%  Source: Bouaouda et al. (2024), Neural Computing and Applications
%  Version: 1.0 (as published, with minor notation cleanup)
%
%  This is the UNMODIFIED reference implementation used for baseline
%  comparison. No improvements are applied here.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Best_fitness, Best_position, Convergence_curve] = ...
         PKO(Popsize, Maxiteration, LB, UB, Dim, Fobj)

tic;
BF           = 8;                  % Fixed beating factor (biological constant)
Crest_angles = 2*pi*rand;          % Single fixed crest angle (random at start)

X       = Initialization(Popsize, Dim, UB, LB);
Fitness = zeros(1, Popsize);
Convergence_curve = zeros(1, Maxiteration);

for i = 1:Popsize
    Fitness(i) = Fobj(X(i,:));
end

[~, sorted_indexes] = sort(Fitness);
Best_position = X(sorted_indexes(1),:);
Best_fitness  = Fitness(sorted_indexes(1));
Convergence_curve(1) = Best_fitness;

t    = 1;
PEmax = 0.5;
PEmin = 0;

X_1     = zeros(Popsize, Dim);
fitnessn = zeros(1, Popsize);

while t < Maxiteration + 1
    o = exp(-t/Maxiteration)^2;

    for i = 1:Popsize
        if rand < 0.8   % Exploration
            j = i;
            while i == j
                seed = randperm(Popsize);
                j = seed(1);
            end
            beatingRate = rand * (Fitness(j)) / (Fitness(i) + eps);
            alpha = 2*randn(1,Dim) - 1;

            if rand < 0.5
                T = beatingRate - ((t)^(1/BF) / (Maxiteration)^(1/BF));
                X_1(i,:) = X(i,:) + alpha.*T.*(X(j,:) - X(i,:));
            else
                T = (exp(1) - exp(((t-1)/Maxiteration)^(1/BF))) * cos(Crest_angles);
                X_1(i,:) = X(i,:) + alpha.*T.*(X(j,:) - X(i,:));
            end

        else  % Exploitation
            alpha = 2*randn(1,Dim) - 1;
            b     = X(i,:) + o^2*randn .* Best_position;
            HuntingAbility = rand * (Fitness(i)) / (Best_fitness + eps);
            X_1(i,:) = X(i,:) + HuntingAbility*o*alpha.*(b - Best_position);
        end
    end

    for i = 1:Popsize
        FU = X_1(i,:) > UB;  FL = X_1(i,:) < LB;
        X_1(i,:) = (X_1(i,:).*(~(FU+FL))) + UB.*FU + LB.*FL;
        fitnessn(i) = Fobj(X_1(i,:));
        if fitnessn(i) < Fitness(i)
            Fitness(i) = fitnessn(i);
            X(i,:)     = X_1(i,:);
        end
        if Fitness(i) < Best_fitness
            Best_fitness  = Fitness(i);
            Best_position = X(i,:);
        end
    end

    % Commensal association phase
    PE = PEmax - (PEmax - PEmin) * (t/Maxiteration);   % Linear decay
    for i = 1:Popsize
        alpha = 2*randn(1,Dim) - 1;
        if rand > (1 - PE)
            X_1(i,:) = X(randi([1,Popsize]),:) + ...
                        o*alpha.*abs(X(i,:) - X(randi([1,Popsize]),:));
        else
            X_1(i,:) = X(i,:);
        end
        FU = X_1(i,:) > UB;  FL = X_1(i,:) < LB;
        X_1(i,:) = (X_1(i,:).*(~(FU+FL))) + UB.*FU + LB.*FL;
        fitnessn(i) = Fobj(X_1(i,:));
        if fitnessn(i) < Fitness(i)
            Fitness(i) = fitnessn(i);
            X(i,:)     = X_1(i,:);
        end
        if Fitness(i) < Best_fitness
            Best_fitness  = Fitness(i);
            Best_position = X(i,:);
        end
    end

    Convergence_curve(t) = Best_fitness;
    t = t + 1;
end
toc;
end
