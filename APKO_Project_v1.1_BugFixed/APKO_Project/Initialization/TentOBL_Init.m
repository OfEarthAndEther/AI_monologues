%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  TentOBL_Init.m
%
%  Tent-Chaos + Opposition-Based Learning Initialization
%
%  Addresses GAP-M3 from SOTA analysis:
%    Standard tent map has a fixed-point singularity at x = 0.5.
%    If any particle starts near 0.5, the chaotic sequence stalls.
%
%  FIX: After each tent iteration, if |x - 0.5| < epsilon, a small
%       uniform random perturbation pushes x away from the fixed point.
%       This preserves ergodicity without changing the overall
%       chaotic properties of the tent map.
%
%  Inputs:
%    N   - population size
%    dim - problem dimension
%    ub  - upper bounds (scalar or 1xdim vector)
%    lb  - lower bounds (scalar or 1xdim vector)
%
%  Output:
%    X   - initialised population (N x dim matrix)
%
%  Mathematical Reference:
%    Tent map:  x(n+1) = 2*x(n)         if x(n) < 0.5
%               x(n+1) = 2*(1 - x(n))   if x(n) >= 0.5
%    OBL:       X'_j = lb_j + ub_j - X_j
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = TentOBL_Init(N, dim, ub, lb)

%% Handle scalar vs vector bounds
if numel(ub) == 1
    ub = ub * ones(1, dim);
end
if numel(lb) == 1
    lb = lb * ones(1, dim);
end

eps_singularity = 0.02;   % Singularity avoidance radius around x=0.5

%% --- Step 1: Tent-Chaotic Initialisation ----------------------------
X_tent = zeros(N, dim);

for d = 1:dim
    x0 = rand;   % Random seed in (0,1)
    % Warm-up: discard first 50 tent iterations to enter chaotic regime
    for w = 1:50
        x0 = TentStep(x0, eps_singularity);
    end
    % Generate N chaotic values
    for i = 1:N
        x0 = TentStep(x0, eps_singularity);
        X_tent(i, d) = lb(d) + x0 * (ub(d) - lb(d));
    end
end

%% --- Step 2: OBL Opposite Population --------------------------------
%  X'_j = lb_j + ub_j - X_j  (standard OBL formula)
X_obl = zeros(N, dim);
for i = 1:N
    for d = 1:dim
        X_obl(i, d) = lb(d) + ub(d) - X_tent(i, d);
        % Clamp to bounds (OBL may produce out-of-bound values for
        % variable-bound problems, though not for symmetric bounds)
        X_obl(i, d) = min(max(X_obl(i, d), lb(d)), ub(d));
    end
end

%% --- Step 3: Selection ----------------------------------------------
%  Combine tent population and OBL population (2N individuals).
%  Keep the N individuals with the most spread (maximum diversity).
%
%  Diversity metric: we select N individuals such that the
%  minimum pairwise Euclidean distance among selected agents is
%  maximised. For computational efficiency, we use a greedy
%  max-min-distance selection (farthest-point sampling).
%
X_combined = [X_tent; X_obl];  % 2N x dim
X = FarthestPointSample(X_combined, N);

end


%% =====================================================================
%% Helper: Single tent-map step with singularity fix
%% =====================================================================
function x_next = TentStep(x, eps_sing)
    if abs(x - 0.5) < eps_sing
        % Perturb away from fixed point
        x = x + eps_sing * (2*rand - 1);
        x = min(max(x, 0), 1);   % Keep in (0,1)
    end
    if x < 0.5
        x_next = 2 * x;
    else
        x_next = 2 * (1 - x);
    end
    % Final safety clamp
    x_next = min(max(x_next, 1e-10), 1-1e-10);
end


%% =====================================================================
%% Helper: Farthest-Point Sampling (Greedy Max-Min Distance)
%%
%%  Selects N points from X_pool (M x dim) such that the minimum
%%  pairwise distance among the selected set is maximised.
%%  Time: O(N * M * dim)
%% =====================================================================
function X_sel = FarthestPointSample(X_pool, N)
    M = size(X_pool, 1);
    if M <= N
        X_sel = X_pool;
        return;
    end

    selected = false(M, 1);
    % Start with the individual closest to the centroid
    centroid = mean(X_pool, 1);
    dists_to_centroid = sum((X_pool - centroid).^2, 2);
    [~, first] = min(dists_to_centroid);
    selected(first) = true;
    sel_idx = first;

    % Incrementally add the point farthest from the current set
    min_dists = sum((X_pool - X_pool(first,:)).^2, 2);  % dist to set
    for k = 2:N
        [~, next_idx] = max(min_dists);
        selected(next_idx) = true;
        sel_idx(end+1) = next_idx; %#ok<AGROW>
        % Update minimum distances
        d_new = sum((X_pool - X_pool(next_idx,:)).^2, 2);
        min_dists = min(min_dists, d_new);
    end

    X_sel = X_pool(sel_idx, :);
end
