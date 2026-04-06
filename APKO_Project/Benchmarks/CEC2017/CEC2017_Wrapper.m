%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  CEC2017_Wrapper.m
%
%  Returns function handle and bounds for CEC-2017 test functions.
%
%  IMPORTANT: This file wraps the official CEC2017 MATLAB/C library.
%  Before running:
%    1. Download the official CEC2017 code from:
%       http://www.ntu.edu.sg/home/epnsugan/index_files/CEC2017/CEC2017.htm
%    2. Compile the MEX file:
%       mex cec17_func.cpp -DWINDOWS (or -DLINUX)
%    3. Ensure cec17_func.mexw64 (or .mexa64) is on the MATLAB path.
%
%  If CEC2017 is not available, this wrapper falls back to
%  standard shifted/rotated benchmark functions (F1-F29 approximations).
%
%  Usage:
%    [Fobj, LB, UB] = CEC2017_Wrapper(func_num, dim)
%    % func_num: 1-29 (CEC2017 functions)
%    % dim: 10, 20, or 30
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Fobj, LB, UB] = CEC2017_Wrapper(func_num, dim)

LB = -100 * ones(1, dim);
UB =  100 * ones(1, dim);

% Check if CEC2017 MEX is available
if exist('cec17_func', 'file') == 3
    % Official MEX available
    Fobj = @(x) cec17_func(x', func_num);
else
    % Fallback: shifted standard functions (approximate behavior)
    warning('CEC2017 MEX not found. Using fallback implementations.');
    Fobj = @(x) CEC17_Fallback(x, func_num, dim);
end
end


%% =====================================================================
%% Fallback implementations (shifted standard functions)
%% These preserve general unimodal/multimodal behavior characteristics
%% but are NOT identical to official CEC2017 functions.
%% For official results, always use the compiled CEC2017 MEX.
%% =====================================================================
function f = CEC17_Fallback(x, func_num, dim)
persistent shift_data rotation_data
if isempty(shift_data)
    shift_data  = -100 + 200*rand(29, 100);   % Random shifts
    rotation_data = eye(100);   % Identity rotation (no rotation in fallback)
end

x = x(:)';   % Ensure row vector
shift = shift_data(func_num, 1:dim);
z = x - shift;   % Shift

switch func_num
    case 1   % Shifted Sphere (unimodal)
        f = sum(z.^2) + 100;
    case 2   % Shifted Schwefel 1.2 (unimodal)
        f = sum(cumsum(z).^2) + 200;
    case 3   % Shifted Rotated High-Conditioned Elliptic
        f = sum((1e6).^((0:dim-1)/(dim-1)) .* z.^2) + 300;
    case 4   % Shifted Bent Cigar
        f = z(1)^2 + 1e6*sum(z(2:end).^2) + 400;
    case 5   % Shifted Rastrigin (simple multimodal)
        f = 10*dim + sum(z.^2 - 10*cos(2*pi*z)) + 500;
    case 6   % Shifted Weierstrass
        a=0.5; b=3; k_max=20;
        f = sum(arrayfun(@(xi) sum(a.^(0:k_max).*cos(2*pi*b.^(0:k_max).*(xi+0.5))), z)) ...
            - dim*sum(a.^(0:k_max).*cos(2*pi*b.^(0:k_max)*0.5)) + 600;
    case 7   % Shifted Griewank
        f = sum(z.^2)/4000 - prod(cos(z./sqrt(1:dim))) + 1 + 700;
    case 8   % Shifted Rastrigin (rotated version in CEC)
        f = 10*dim + sum(z.^2 - 10*cos(2*pi*z)) + 800;
    case 9   % Shifted Rastrigin variant
        f = 10*dim + sum(z.^2 - 10*cos(2*pi*z)) + 900;
    case 10  % Schwefel variant
        f = 418.9829*dim - sum(z.*sin(sqrt(abs(z)))) + 1000;
    case {11,12,13,14,15}  % Hybrid functions (simplified: sum of components)
        f = sum(z.^2 - 10*cos(2*pi*z)) + 10*dim + func_num*100;
    case {16,17,18,19,20}  % Hybrid functions variant 2
        f = sum(z.^2)/4000 - prod(cos(z./sqrt(1:dim))) + 1 + func_num*100;
    case {21,22,23,24,25}  % Composition functions (simplified)
        f = sum((z.^2 - 10*cos(2*pi*z))) + 10*dim + func_num*100;
    case {26,27,28,29}     % Composition functions variant 2
        f = sum(cumsum(z).^2) + func_num*100;
    otherwise
        f = sum(z.^2) + func_num*100;
end
end
