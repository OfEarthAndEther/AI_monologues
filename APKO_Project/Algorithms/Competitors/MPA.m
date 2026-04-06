%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MPA.m  --  Marine Predators Algorithm
%  Faramarzi et al. (2020), Expert Systems with Applications, 152, 113377
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Best_fitness, Best_position, Convergence_curve] = ...
         MPA(Popsize, Maxiteration, LB, UB, Dim, Fobj)

X = Init(Popsize, Dim, UB, LB);
Fitness = arrayfun(@(i) Fobj(X(i,:)), 1:Popsize);
[Best_fitness, bi] = min(Fitness);
Best_position = X(bi,:);
Convergence_curve = zeros(1, Maxiteration);

FADs = 0.2; P = 0.5;

for t = 1:Maxiteration
    CF = (1 - t/Maxiteration)^(2*t/Maxiteration);
    Elite = repmat(Best_position, Popsize, 1);

    for i = 1:Popsize
        if t < Maxiteration/3
            % Phase 1: High velocity ratio (prey moves faster)
            RB = randn(1,Dim);
            stepsize = RB .* (Elite(i,:) - RB.*X(i,:));
            X(i,:) = X(i,:) + P*rand*stepsize;
        elseif t < 2*Maxiteration/3
            % Phase 2: Unit velocity ratio
            if i <= Popsize/2
                RL = LevyFlight(Dim);
                stepsize = RL .* (Elite(i,:) - RL.*X(i,:));
                X(i,:) = X(i,:) + P*rand*stepsize;
            else
                RB = randn(1,Dim);
                stepsize = RB .* (RB.*Elite(i,:) - X(i,:));
                X(i,:) = Elite(i,:) + P*CF*stepsize;
            end
        else
            % Phase 3: Low velocity ratio (predator moves faster)
            RL = LevyFlight(Dim);
            stepsize = RL .* (RL.*Elite(i,:) - X(i,:));
            X(i,:) = Elite(i,:) + P*CF*stepsize;
        end

        X(i,:) = ClampBounds(X(i,:), LB, UB);
        Fitness(i) = Fobj(X(i,:));
        if Fitness(i) < Best_fitness
            Best_fitness=Fitness(i); Best_position=X(i,:);
        end
    end

    % FADs effect
    if rand < FADs
        U = rand(Popsize, Dim) < FADs;
        X = X + CF*(LB + rand(Popsize,Dim).*(UB-LB)).*U;
        for i=1:Popsize
            X(i,:)=ClampBounds(X(i,:),LB,UB);
            Fitness(i)=Fobj(X(i,:));
            if Fitness(i)<Best_fitness; Best_fitness=Fitness(i); Best_position=X(i,:); end
        end
    end
    Convergence_curve(t) = Best_fitness;
end
end
function step=LevyFlight(d); beta=1.5; sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta); u=randn(1,d)*sigma; v=randn(1,d); step=0.01*u./(abs(v).^(1/beta)); end
function X=Init(N,d,ub,lb); if numel(ub)==1; X=rand(N,d)*(ub-lb)+lb; else; X=zeros(N,d); for j=1:d; X(:,j)=rand(N,1)*(ub(j)-lb(j))+lb(j); end; end; end
function x=ClampBounds(x,lb,ub); if numel(ub)==1; FU=x>ub;FL=x<lb; else; FU=x>ub;FL=x<lb; end; x=(x.*(~(FU+FL)))+ub.*FU+lb.*FL; end
