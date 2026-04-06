%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WOA.m  --  Whale Optimization Algorithm
%  Mirjalili & Lewis (2016), Advances in Engineering Software, 95, 51-67
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Best_fitness, Best_position, Convergence_curve] = ...
         WOA(Popsize, Maxiteration, LB, UB, Dim, Fobj)

X = Init(Popsize, Dim, UB, LB);
Fitness = arrayfun(@(i) Fobj(X(i,:)), 1:Popsize);
[Best_fitness, bi] = min(Fitness);
Best_position = X(bi,:);
Convergence_curve = zeros(1, Maxiteration);

for t = 1:Maxiteration
    a  = 2 - t*(2/Maxiteration);
    a2 = -1 - t*(-1/Maxiteration);
    for i = 1:Popsize
        r1=rand; r2=rand;
        A=2*a*r1-a; C=2*r2;
        b=1; l=(a2-1)*rand+1;
        p=rand;
        if p < 0.5
            if abs(A) < 1
                D=abs(C.*Best_position - X(i,:));
                X_new = Best_position - A.*D;
            else
                rand_idx = randi(Popsize);
                X_rand = X(rand_idx,:);
                D=abs(C.*X_rand - X(i,:));
                X_new = X_rand - A.*D;
            end
        else
            D2=abs(Best_position - X(i,:));
            X_new = D2.*exp(b*l).*cos(2*pi*l) + Best_position;
        end
        X_new = ClampBounds(X_new, LB, UB);
        f_new = Fobj(X_new);
        if f_new < Fitness(i)
            X(i,:)=X_new; Fitness(i)=f_new;
        end
        if Fitness(i) < Best_fitness
            Best_fitness=Fitness(i); Best_position=X(i,:);
        end
    end
    Convergence_curve(t) = Best_fitness;
end
end
function X=Init(N,d,ub,lb); if numel(ub)==1; X=rand(N,d)*(ub-lb)+lb; else; X=zeros(N,d); for j=1:d; X(:,j)=rand(N,1)*(ub(j)-lb(j))+lb(j); end; end; end
function x=ClampBounds(x,lb,ub); FU=x>ub;FL=x<lb; x=(x.*(~(FU+FL)))+ub.*FU+lb.*FL; end
