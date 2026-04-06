%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  GWO.m  --  Grey Wolf Optimizer
%  Mirjalili et al. (2014), Advances in Engineering Software, 69, 46-61
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Best_fitness, Best_position, Convergence_curve] = ...
         GWO(Popsize, Maxiteration, LB, UB, Dim, Fobj)

X = Init(Popsize, Dim, UB, LB);
Fitness = arrayfun(@(i) Fobj(X(i,:)), 1:Popsize);
[~, si] = sort(Fitness);
Alpha_pos=X(si(1),:); Alpha_score=Fitness(si(1));
Beta_pos =X(si(2),:); Beta_score =Fitness(si(2));
Delta_pos=X(si(3),:); Delta_score=Fitness(si(3));
Convergence_curve=zeros(1,Maxiteration);

for t=1:Maxiteration
    a=2-t*(2/Maxiteration);
    for i=1:Popsize
        for j=1:3
            if j==1; leader=Alpha_pos;
            elseif j==2; leader=Beta_pos;
            else; leader=Delta_pos; end
            r1=rand(1,Dim); r2=rand(1,Dim);
            A=2*a.*r1-a; C=2*r2;
            D=abs(C.*leader-X(i,:));
            X1(j,:)=leader-A.*D;
        end
        X_new=(X1(1,:)+X1(2,:)+X1(3,:))/3;
        X_new=ClampBounds(X_new,LB,UB);
        f_new=Fobj(X_new);
        if f_new<Fitness(i); X(i,:)=X_new; Fitness(i)=f_new; end
        if Fitness(i)<Alpha_score
            Delta_pos=Beta_pos; Delta_score=Beta_score;
            Beta_pos=Alpha_pos; Beta_score=Alpha_score;
            Alpha_pos=X(i,:); Alpha_score=Fitness(i);
        elseif Fitness(i)<Beta_score
            Delta_pos=Beta_pos; Delta_score=Beta_score;
            Beta_pos=X(i,:); Beta_score=Fitness(i);
        elseif Fitness(i)<Delta_score
            Delta_pos=X(i,:); Delta_score=Fitness(i);
        end
    end
    Convergence_curve(t)=Alpha_score;
end
Best_fitness=Alpha_score; Best_position=Alpha_pos;
end
function X=Init(N,d,ub,lb); if numel(ub)==1; X=rand(N,d)*(ub-lb)+lb; else; X=zeros(N,d); for j=1:d; X(:,j)=rand(N,1)*(ub(j)-lb(j))+lb(j); end; end; end
function x=ClampBounds(x,lb,ub); FU=x>ub;FL=x<lb; x=(x.*(~(FU+FL)))+ub.*FU+lb.*FL; end
