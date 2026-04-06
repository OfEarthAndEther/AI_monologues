%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Engineering_Problems.m
%
%  Returns problem definition struct for a given engineering problem.
%  Problems from SNS paper (Ayyarao et al., 2022) and standard NIA
%  benchmark literature.
%
%  Usage:
%    P = Engineering_Problems('TCS');   % Tension/Compression Spring
%    [f, g] = P.Fobj(x);               % f=cost, g=constraints (<=0 OK)
%
%  Problems:
%    'TCS'  - Tension/Compression Spring Design
%    'PVD'  - Pressure Vessel Design
%    'WBD'  - Welded Beam Design
%    'SRD'  - Speed Reducer Design
%    'TBT'  - Three-Bar Truss Design
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function P = Engineering_Problems(name)
switch upper(name)

    %% ================================================================
    %  TCS: Tension/Compression Spring Design
    %  Variables: x = [d (wire dia), D (mean coil dia), N (coils)]
    %  Source: Coello (2000), Mezura-Montes & Coello (2008)
    %% ================================================================
    case 'TCS'
        P.name = 'Tension/Compression Spring';
        P.Dim  = 3;
        P.LB   = [0.05, 0.25, 2];
        P.UB   = [2.00, 1.30, 15];
        P.Fobj = @TCS_Fobj;

    %% ================================================================
    %  PVD: Pressure Vessel Design
    %  Variables: x = [Ts (shell thick), Th (head thick), R (radius), L (length)]
    %  Source: Kannan & Kramer (1994)
    %% ================================================================
    case 'PVD'
        P.name = 'Pressure Vessel Design';
        P.Dim  = 4;
        P.LB   = [0, 0, 10, 10];
        P.UB   = [99, 99, 200, 200];
        P.Fobj = @PVD_Fobj;

    %% ================================================================
    %  WBD: Welded Beam Design
    %  Variables: x = [h (weld height), l (attach length), t (bar height), b (bar width)]
    %  Source: Ragsdell & Phillips (1976)
    %% ================================================================
    case 'WBD'
        P.name = 'Welded Beam Design';
        P.Dim  = 4;
        P.LB   = [0.1, 0.1, 0.1, 0.1];
        P.UB   = [2.0, 10.0, 10.0, 2.0];
        P.Fobj = @WBD_Fobj;

    %% ================================================================
    %  SRD: Speed Reducer Design
    %  Variables: x = [b, m, z, l1, l2, d1, d2]
    %             (face width, module, teeth, shaft lengths, shaft diameters)
    %  Source: Golinski (1973)
    %% ================================================================
    case 'SRD'
        P.name = 'Speed Reducer Design';
        P.Dim  = 7;
        P.LB   = [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0];
        P.UB   = [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5];
        P.Fobj = @SRD_Fobj;

    %% ================================================================
    %  TBT: Three-Bar Truss Design (from SNS problem set)
    %  Variables: x = [A1 (cross section 1), A2 (cross section 2)]
    %  Source: Deb & Goyal (1996)
    %% ================================================================
    case 'TBT'
        P.name = 'Three-Bar Truss Design';
        P.Dim  = 2;
        P.LB   = [0, 0];
        P.UB   = [1, 1];
        P.Fobj = @TBT_Fobj;

    otherwise
        error('Unknown engineering problem: %s', name);
end
end


%% ================================================================
%%  Problem Implementations
%% ================================================================

function f = TCS_Fobj(x)
    d = x(1); D = x(2); N = x(3);
    % Objective: minimize total spring weight
    f = (N + 2)*D*d^2;

    % Constraints (penalty method: P = sum(max(0,g)^2) * 1e6)
    g(1) = 1 - (D^3*N)/(71785*d^4);
    g(2) = (4*D^2 - d*D)/(12566*(D*d^3 - d^4)) + 1/(5108*d^2) - 1;
    g(3) = 1 - 140.45*d/(D^2*N);
    g(4) = (D+d)/1.5 - 1;

    penalty = 1e6 * sum(max(0, g).^2);
    f = f + penalty;
end

function f = PVD_Fobj(x)
    Ts = x(1); Th = x(2); R = x(3); L = x(4);
    % Discrete variable approximation (continuous relaxation)
    % Objective: minimize total manufacturing cost
    f = 0.6224*Ts*R*L + 1.7781*Th*R^2 + 3.1661*Ts^2*L + 19.84*Ts^2*R;

    % Constraints
    g(1) = -Ts + 0.0193*R;
    g(2) = -Th + 0.00954*R;
    g(3) = -pi*R^2*L - (4/3)*pi*R^3 + 1296000;
    g(4) = L - 240;

    penalty = 1e6 * sum(max(0, g).^2);
    f = f + penalty;
end

function f = WBD_Fobj(x)
    h = x(1); l = x(2); t = x(3); b = x(4);
    % Objective: minimize fabrication cost
    f = 1.10471*h^2*l + 0.04811*t*b*(14+l);

    % Material and geometry constants
    P   = 6000;    % Applied load (lb)
    L   = 14;      % Length of bar (in)
    del = 0.25;    % Max deflection (in)
    E   = 30e6;    % Elastic modulus (psi)
    G   = 12e6;    % Shear modulus (psi)
    tau_max = 13600;   % Max shear stress (psi)
    sig_max = 30000;   % Max bending stress (psi)
    P_c_min = 6000;    % Min buckling load

    M   = P*(L + l/2);
    R   = sqrt(l^2/4 + ((h+t)/2)^2);
    J   = 2*(sqrt(2)*h*l*(l^2/12 + ((h+t)/2)^2));
    tau1 = P/(sqrt(2)*h*l);
    tau2 = M*R/J;
    tau  = sqrt(tau1^2 + 2*tau1*tau2*(l/(2*R)) + tau2^2);
    sig  = 6*P*L/(b*t^2);
    delta_val = 4*P*L^3/(E*b*t^3);
    Pc  = (4.013*E*sqrt(t^2*b^6/36)/L^2)*(1 - t/(2*L)*sqrt(E/(4*G)));

    g(1) = tau - tau_max;
    g(2) = sig - sig_max;
    g(3) = h - b;
    g(4) = 0.10471*h^2 + 0.04811*t*b*(14+l) - 5;
    g(5) = 0.125 - h;
    g(6) = delta_val - del;
    g(7) = P - Pc;

    penalty = 1e6 * sum(max(0, g).^2);
    f = f + penalty;
end

function f = SRD_Fobj(x)
    b=x(1); m=x(2); z=x(3); l1=x(4); l2=x(5); d1=x(6); d2=x(7);
    % Objective: minimize weight of speed reducer
    f = 0.7854*b*m^2*(3.3333*z^2 + 14.9334*z - 43.0934) - ...
        1.508*b*(d1^2 + d2^2) + ...
        7.4777*(d1^3 + d2^3) + ...
        0.7854*(l1*d1^2 + l2*d2^2);

    % Constraints
    g(1)  = 27/(b*m^2*z) - 1;
    g(2)  = 397.5/(b*m^2*z^2) - 1;
    g(3)  = 1.93*l1^3/(m*z*d1^4) - 1;
    g(4)  = 1.93*l2^3/(m*z*d2^4) - 1;
    A1    = sqrt((745*l1/(m*z))^2 + 16.9e6);
    A2    = sqrt((745*l2/(m*z))^2 + 157.5e6);
    g(5)  = A1/(0.1*d1^3) - 1100;
    g(6)  = A2/(0.1*d2^3) - 850;
    g(7)  = m*z - 40;
    g(8)  = 5*m - b;
    g(9)  = b - 12*m;
    g(10) = 1.5*d1 + 1.9 - l1;
    g(11) = 1.1*d2 + 1.9 - l2;

    penalty = 1e6 * sum(max(0, g).^2);
    f = f + penalty;
end

function f = TBT_Fobj(x)
    % Three-Bar Truss: minimize weight
    A1 = x(1); A2 = x(2);
    l  = 100;   % Length (cm)
    P  = 2;     % Load (kN/cm^2)
    sig= 2;     % Stress limit (kN/cm^2)
    rho= 0.1;   % Density

    f = rho*l*(2*sqrt(2)*A1 + A2);

    g(1) = (sqrt(2)*A1 + A2)/(sqrt(2)*A1^2 + 2*A1*A2)*P - sig;
    g(2) = A2/(sqrt(2)*A1^2 + 2*A1*A2)*P - sig;
    g(3) = 1/(sqrt(2)*A2 + A1)*P - sig;

    penalty = 1e6 * sum(max(0, g).^2);
    f = f + penalty;
end
