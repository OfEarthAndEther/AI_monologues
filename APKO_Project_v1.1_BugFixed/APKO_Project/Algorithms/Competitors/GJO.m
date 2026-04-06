%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  GJO.m  --  Golden Jackal Optimizer
%  Chopra & Ansari (2022), Expert Systems with Applications, 198, 116924
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Best_fitness, Best_position, Convergence_curve] = ...
         GJO(Popsize, Maxiteration, LB, UB, Dim, Fobj)

X = Init(Popsize, Dim, UB, LB);
Fitness = arrayfun(@(i) Fobj(X(i,:)), 1:Popsize);
[~, si] = sort(Fitness);
Male_pos = X(si(1),:); Male_fit = Fitness(si(1));
Female_pos= X(si(2),:); Female_fit= Fitness(si(2));
Best_position = Male_pos; Best_fitness = Male_fit;
Convergence_curve = zeros(1, Maxiteration);

for t = 1:Maxiteration
    E1 = 1.5*(1 - t/Maxiteration);
    for i = 1:Popsize
        E0 = 2*rand - 1;
        E  = E1*E0;
        rl = 0.01*LevyFlight(Dim);

        X1 = Male_pos   - E.*abs(rl.*Male_pos   - X(i,:));
        X2 = Female_pos - E.*abs(rl.*Female_pos - X(i,:));
        X_new = (X1 + X2)/2;
        X_new = ClampBounds(X_new, LB, UB);
        f_new = Fobj(X_new);

        if f_new < Fitness(i); X(i,:)=X_new; Fitness(i)=f_new; end

        if Fitness(i) < Male_fit
            Female_pos=Male_pos; Female_fit=Male_fit;
            Male_pos=X(i,:); Male_fit=Fitness(i);
        elseif Fitness(i) < Female_fit
            Female_pos=X(i,:); Female_fit=Fitness(i);
        end
    end
    Best_position = Male_pos; Best_fitness = Male_fit;
    Convergence_curve(t) = Best_fitness;
end
end
function step=LevyFlight(d); beta=1.5; sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta); u=randn(1,d)*sigma; v=randn(1,d); step=u./(abs(v).^(1/beta)); end
function X=Init(N,d,ub,lb); if numel(ub)==1; X=rand(N,d)*(ub-lb)+lb; else; X=zeros(N,d); for j=1:d; X(:,j)=rand(N,1)*(ub(j)-lb(j))+lb(j); end; end; end
function x=ClampBounds(x,lb,ub); FU=x>ub;FL=x<lb; x=(x.*(~(FU+FL)))+ub.*FU+lb.*FL; end
