%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  AOA.m  --  Arithmetic Optimization Algorithm
%  Abualigah et al. (2021), Computer Methods in Applied Mechanics
%  and Engineering, 376, 113609
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Best_fitness, Best_position, Convergence_curve] = ...
         AOA(Popsize, Maxiteration, LB, UB, Dim, Fobj)

X = Init(Popsize, Dim, UB, LB);
Fitness = arrayfun(@(i) Fobj(X(i,:)), 1:Popsize);
[Best_fitness, bi] = min(Fitness);
Best_position = X(bi,:);
Convergence_curve = zeros(1, Maxiteration);

alpha = 5; mu = 0.499;

for t = 1:Maxiteration
    MOA = 0.1 + t*((1-0.1)/Maxiteration);   % Math optimizer accel
    MOP = 1 - (t^(1/alpha))/(Maxiteration^(1/alpha));   % Math optimizer prob

    for i = 1:Popsize
        for d = 1:Dim
            r1 = rand; r2 = rand; r3 = rand;
            if r3 > MOA   % Exploration
                if r1 < 0.5
                    X(i,d) = Best_position(d) / (MOP + eps) * ((UB(min(d,end))-LB(min(d,end)))*mu + LB(min(d,end)));
                else
                    X(i,d) = Best_position(d) * MOP * ((UB(min(d,end))-LB(min(d,end)))*mu + LB(min(d,end)));
                end
            else  % Exploitation
                if r2 < 0.5
                    X(i,d) = Best_position(d) - MOP * ((UB(min(d,end))-LB(min(d,end)))*mu + LB(min(d,end)));
                else
                    X(i,d) = Best_position(d) + MOP * ((UB(min(d,end))-LB(min(d,end)))*mu + LB(min(d,end)));
                end
            end
        end
        X(i,:) = ClampBounds(X(i,:), LB, UB);
        Fitness(i) = Fobj(X(i,:));
        if Fitness(i) < Best_fitness
            Best_fitness=Fitness(i); Best_position=X(i,:);
        end
    end
    Convergence_curve(t) = Best_fitness;
end
end
function X=Init(N,d,ub,lb); if numel(ub)==1; X=rand(N,d)*(ub-lb)+lb; else; X=zeros(N,d); for j=1:d; X(:,j)=rand(N,1)*(ub(j)-lb(j))+lb(j); end; end; end
function x=ClampBounds(x,lb,ub); if numel(ub)==1; FU=x>ub;FL=x<lb; else; FU=bsxfun(@gt,x,ub);FL=bsxfun(@lt,x,lb); end; x=(x.*(~(FU+FL)))+ub.*FU+lb.*FL; end
