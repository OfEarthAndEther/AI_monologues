%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Initialization.m  --  Standard Uniform Random Initialization
%  (Used by base PKO and competitor algorithms)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X, new_lb, new_ub] = Initialization(N, dim, ub, lb)

Boundary = size(ub, 2);
new_lb = lb;
new_ub = ub;

if Boundary == 1
    X = rand(N, dim) .* (ub - lb) + lb;
    new_lb = lb * ones(1, dim);
    new_ub = ub * ones(1, dim);
end

if Boundary > 1
    X = zeros(N, dim);
    for i = 1:dim
        X(:, i) = rand(N, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end
end
