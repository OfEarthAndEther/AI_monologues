%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  HHO.m  --  Harris Hawks Optimization
%  Heidari et al. (2019), Future Generation Computer Systems, 97, 849-872
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Best_fitness, Best_position, Convergence_curve] = ...
         HHO(Popsize, Maxiteration, LB, UB, Dim, Fobj)

X = Init(Popsize, Dim, UB, LB);
Fitness = arrayfun(@(i) Fobj(X(i,:)), 1:Popsize);
[Best_fitness, bi] = min(Fitness);
Best_position = X(bi,:);
Convergence_curve = zeros(1, Maxiteration);

for t = 1:Maxiteration
    E1 = 2*(1 - t/Maxiteration);   % Escape energy reduction
    for i = 1:Popsize
        E0 = 2*rand - 1;
        Escaping_E = E1 * E0;
        q = rand; r = rand;
        if abs(Escaping_E) >= 1
            % Exploration
            rand_idx = randi(Popsize);
            X_rand = X(rand_idx,:);
            if q < 0.5
                X(i,:) = X_rand - rand*abs(X_rand - 2*rand*X(i,:));
            else
                X(i,:) = Best_position - mean(X,1) - rand.*(LB + rand*(UB-LB));
            end
        else
            % Exploitation
            J = 2*(1-rand); Delta = Best_position - X(i,:);
            if r >= 0.5 && abs(Escaping_E) < 0.5
                X(i,:) = Delta - Escaping_E*abs(Delta);
            elseif r >= 0.5 && abs(Escaping_E) >= 0.5
                X(i,:) = Best_position - Escaping_E*abs(J*Best_position - X(i,:));
            elseif r < 0.5 && abs(Escaping_E) >= 0.5
                X1 = Best_position - Escaping_E*abs(J*Best_position - X(i,:));
                X1 = ClampBounds(X1, LB, UB);
                if Fobj(X1) < Fitness(i); X(i,:)=X1; end
                LF_D = LevyFlight(Dim);
                X2 = X(i,:) + LF_D.*Delta;
                X2 = ClampBounds(X2, LB, UB);
                if Fobj(X2) < Fitness(i); X(i,:)=X2; end
            else
                X1 = Best_position - Escaping_E*abs(J*Best_position - mean(X,1));
                X1 = ClampBounds(X1, LB, UB);
                if Fobj(X1) < Fitness(i); X(i,:)=X1; end
                LF_D = LevyFlight(Dim);
                X2 = X(i,:) + LF_D.*(Best_position - mean(X,1));
                X2 = ClampBounds(X2, LB, UB);
                if Fobj(X2) < Fitness(i); X(i,:)=X2; end
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
function step=LevyFlight(d); beta=1.5; sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta); u=randn(1,d)*sigma; v=randn(1,d); step=u./(abs(v).^(1/beta)); end
function X=Init(N,d,ub,lb); if numel(ub)==1; X=rand(N,d)*(ub-lb)+lb; else; X=zeros(N,d); for j=1:d; X(:,j)=rand(N,1)*(ub(j)-lb(j))+lb(j); end; end; end
function x=ClampBounds(x,lb,ub); FU=x>ub;FL=x<lb; x=(x.*(~(FU+FL)))+ub.*FU+lb.*FL; end
