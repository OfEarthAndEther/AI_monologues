%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  CEC2014_Wrapper.m
%
%  Returns function handle and bounds for CEC-2014 test functions.
%
%  IMPORTANT: This wraps the official CEC2014 MATLAB/C library.
%  Before running:
%    1. Download from:
%       http://www.ntu.edu.sg/home/epnsugan/index_files/CEC2014/CEC2014.htm
%    2. Compile MEX: mex cec14_func.cpp input_data/
%    3. Ensure cec14_func.mex* is on the MATLAB path.
%
%  Usage:
%    [Fobj, LB, UB] = CEC2014_Wrapper(func_num, dim)
%    % func_num: 1-30 (CEC2014 functions)
%    % dim: 10, 20, 30, or 50
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Fobj, LB, UB] = CEC2014_Wrapper(func_num, dim)

LB = -100 * ones(1, dim);
UB =  100 * ones(1, dim);

if exist('cec14_func', 'file') == 3
    Fobj = @(x) cec14_func(x', func_num);
else
    warning('CEC2014 MEX not found. Using fallback implementations.');
    Fobj = @(x) CEC14_Fallback(x, func_num, dim);
end
end


function f = CEC14_Fallback(x, func_num, dim)
persistent shift14
if isempty(shift14)
    shift14 = -100 + 200*rand(30, 100);
end

x = x(:)';
shift = shift14(func_num, 1:dim);
z = x - shift;

switch func_num
    % Unimodal functions (F1-F3)
    case 1   % Rotated High-Conditioned Elliptic
        f = sum((1e6).^((0:dim-1)/(dim-1)) .* z.^2) + 100;
    case 2   % Rotated Bent Cigar
        f = z(1)^2 + 1e6*sum(z(2:end).^2) + 200;
    case 3   % Rotated Discus
        f = 1e6*z(1)^2 + sum(z(2:end).^2) + 300;

    % Simple multimodal (F4-F16)
    case 4   % Shifted/Rotated Rosenbrock
        f = sum(100*(z(2:end)-z(1:end-1).^2).^2 + (z(1:end-1)-1).^2) + 400;
    case 5   % Shifted/Rotated Ackley
        f = -20*exp(-0.2*sqrt(mean(z.^2))) - exp(mean(cos(2*pi*z))) + 20 + exp(1) + 500;
    case 6   % Shifted/Rotated Weierstrass
        a=0.5; b=3; k_max=20;
        f = sum(arrayfun(@(xi) sum(a.^(0:k_max).*cos(2*pi*b.^(0:k_max).*(xi+0.5))),z)) ...
            - dim*sum(a.^(0:k_max).*cos(pi*b.^(0:k_max))) + 600;
    case 7   % Shifted/Rotated Griewank
        f = sum(z.^2)/4000 - prod(cos(z./sqrt(1:dim))) + 1 + 700;
    case 8   % Shifted Rastrigin
        f = 10*dim + sum(z.^2 - 10*cos(2*pi*z)) + 800;
    case 9   % Shifted/Rotated Rastrigin
        f = 10*dim + sum(z.^2 - 10*cos(2*pi*z)) + 900;
    case 10  % Shifted Schwefel
        f = 418.9829*dim - sum(z.*sin(sqrt(abs(z)))) + 1000;
    case 11  % Shifted/Rotated Schwefel
        f = 418.9829*dim - sum(z.*sin(sqrt(abs(z)))) + 1100;
    case 12  % Shifted/Rotated Katsuura
        f = (10/dim^2)*prod((1+((1:dim).*abs(z)).^(10/(dim^1.2))).^(10/dim^2)) + 1200;
    case 13  % Shifted/Rotated HappyCat
        f = abs(sum(z.^2)-dim)^0.25 + (0.5*sum(z.^2)+sum(z))/dim + 0.5 + 1300;
    case 14  % Shifted/Rotated HGBat
        f = abs(sum(z.^2)^2 - sum(z)^2)^0.5 + (0.5*sum(z.^2)+sum(z))/dim + 0.5 + 1400;
    case 15  % Shifted/Rotated Expanded Griewank's + Rosenbrock
        g = @(a,b) (a^2-b)^2 + (a-1)^2;
        f = sum(arrayfun(@(i) g(z(i),z(mod(i,dim)+1)), 1:dim)) + 1500;
    case 16  % Shifted/Rotated Expanded Scaffer's F6
        sf = @(a,b) 0.5 + (sin(sqrt(a^2+b^2))^2-0.5)/(1+0.001*(a^2+b^2))^2;
        f = sum(arrayfun(@(i) sf(z(i),z(mod(i,dim)+1)), 1:dim)) + 1600;

    % Hybrid functions (F17-F22) - simplified as weighted sums
    case {17,18,19,20,21,22}
        p = func_num - 16;
        components = [sum(z.^2), sum(z.^2-10*cos(2*pi*z))+10*dim, ...
                      -20*exp(-0.2*sqrt(mean(z.^2)))-exp(mean(cos(2*pi*z)))+20+exp(1), ...
                      sum(z.^2)/4000-prod(cos(z./sqrt(1:dim)))+1, ...
                      sum(100*(z(2:end)-z(1:end-1).^2).^2+(z(1:end-1)-1).^2), ...
                      418.9829*dim-sum(z.*sin(sqrt(abs(z))))];
        w = ones(1,6)/6;
        f = sum(w .* components) + func_num*100;

    % Composition functions (F23-F30)
    case {23,24,25,26,27,28,29,30}
        % Composition of Rastrigin + Weierstrass + Griewank
        f1 = 10*dim + sum(z.^2 - 10*cos(2*pi*z));
        f2 = sum(z.^2)/4000 - prod(cos(z./sqrt(1:dim))) + 1;
        f3 = sum(cumsum(z).^2);
        w  = [0.3 0.4 0.3];
        f  = w(1)*f1 + w(2)*f2 + w(3)*f3 + func_num*100;

    otherwise
        f = sum(z.^2) + func_num*100;
end
end
