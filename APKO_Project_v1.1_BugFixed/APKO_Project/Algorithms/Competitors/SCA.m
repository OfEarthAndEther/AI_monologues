%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SCA.m  --  Sine Cosine Algorithm
%  Mirjalili (2016), Knowledge-Based Systems, 96, 120-133
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Best_fitness, Best_position, Convergence_curve] = ...
         SCA(Popsize, Maxiteration, LB, UB, Dim, Fobj)

X = Init(Popsize, Dim, UB, LB);
Fitness = arrayfun(@(i) Fobj(X(i,:)), 1:Popsize);
[Best_fitness, bi] = min(Fitness);
Best_position = X(bi,:);
Convergence_curve = zeros(1, Maxiteration);

for t = 1:Maxiteration
    a = 2;
    r1 = a - t*(a/Maxiteration);
    for i = 1:Popsize
        r2 = 2*pi*rand; r3 = rand; r4 = rand;
        if r4 < 0.5
            X(i,:) = X(i,:) + r1*sin(r2)*abs(r3*Best_position - X(i,:));
        else
            X(i,:) = X(i,:) + r1*cos(r2)*abs(r3*Best_position - X(i,:));
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
function x=ClampBounds(x,lb,ub); FU=x>ub;FL=x<lb; x=(x.*(~(FU+FL)))+ub.*FU+lb.*FL; end
