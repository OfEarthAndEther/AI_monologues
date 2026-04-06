# APKO Project — Step-by-Step Execution Guide
**Version 1.1 (Bug-Fixed) | All three critical bugs resolved**

---

## What was fixed in v1.1

| Bug | Problem | Fix |
|-----|---------|-----|
| BUG-1 | Entropy-restart called `Fobj()` without counting against FE budget → APKO got a silent free advantage | Added `FE_count` throughout; main loop exits when `FE_count >= Popsize*Maxiteration` |
| BUG-2 | Wilcoxon **Signed-Rank** (paired test) was used on **independent** runs — wrong test | Replaced with **Mann-Whitney U** (Rank-Sum) for independent groups |
| BUG-3 | Crest angle called `rand` every iteration — claimed to be "deterministic" but wasn't | Replaced with a pure function of `t`: `θ(t) = π/2·sin(πt/T) + π/4·cos(2πt/T)` |

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| MATLAB | R2019b or newer | Run everything |
| C compiler | GCC / MSVC | Compile CEC MEX files (one-time) |
| CEC2014 source | Downloaded separately | Official benchmark landscapes |
| CEC2017 source | Downloaded separately | Official benchmark landscapes |

> **No external MATLAB toolboxes required.**
> All statistical functions (Mann-Whitney U, Friedman, Holm) are implemented
> natively inside `Analysis/Statistical_Tests.m`.

---

## PART A — One-Time Setup (do this once, never again)

### Step 1 — Download CEC benchmark source code

**CEC-2014:**
1. Go to: http://www.ntu.edu.sg/home/epnsugan/index_files/CEC2014/CEC2014.htm
2. Download the MATLAB/C source package
3. Copy **all files** from that package into:
   ```
   APKO_Project/Benchmarks/CEC2014/
   ```
   You should have these files there after copying:
   - `cec14_func.cpp`
   - `input_data/` folder (rotation matrices and shift vectors)
   - `cec14_test_func.m` (optional reference)

**CEC-2017:**
1. Go to: http://www.ntu.edu.sg/home/epnsugan/index_files/CEC2017/CEC2017.htm
2. Download the MATLAB/C source package
3. Copy **all files** into:
   ```
   APKO_Project/Benchmarks/CEC2017/
   ```
   You need:
   - `cec17_func.cpp`
   - `input_data/` folder
   - `cec17_test_func.m` (optional reference)

---

### Step 2 — Compile the CEC MEX files

Open MATLAB. In the Command Window:

```matlab
% --- Compile CEC2014 ---
cd('APKO_Project/Benchmarks/CEC2014')

% Windows:
mex cec14_func.cpp

% Linux/Mac:
mex cec14_func.cpp CFLAGS='$CFLAGS -std=c99'

% Verify:  a file called cec14_func.mexw64 (Win) or cec14_func.mexa64 (Linux) appears
dir('cec14_func.*')


% --- Compile CEC2017 ---
cd('../CEC2017')

% Windows:
mex cec17_func.cpp

% Linux/Mac:
mex cec17_func.cpp CFLAGS='$CFLAGS -std=c99'

% Verify:
dir('cec17_func.*')

% Return to project root
cd('../../..')
```

> **If MEX compilation fails:**
> MATLAB will show an error. Common fixes:
> - Run `mex -setup` to select your C compiler
> - On Windows: install "MinGW-w64 C/C++ Compiler" from MATLAB Add-Ons
> - On Linux: `sudo apt install build-essential`
> - If you cannot compile: the wrappers fall back to approximate functions
>   (results will run but won't match official CEC benchmarks exactly)

---

### Step 3 — Configure MATLAB path

In the MATLAB Command Window, navigate to the project root and run:

```matlab
cd('path/to/APKO_Project')   % Replace with your actual path
setup_paths()
```

Expected output:
```
[setup_paths] MATLAB path configured for APKO project.
[setup_paths] Root: /your/path/APKO_Project
[setup_paths] IMPORTANT: Compile CEC2014 and CEC2017 MEX files:
  cd Benchmarks/CEC2014 && mex cec14_func.cpp
  cd Benchmarks/CEC2017 && mex cec17_func.cpp
[setup_paths] Ready.
```

> Run `setup_paths()` every time you open a new MATLAB session before
> running any experiment.

---

## PART B — Validate the Setup (Run This First)

Before committing to the full 50-run experiment (~12–24 hours), run a
quick 5-run test to confirm everything is wired up correctly.

### Step 4 — Quick test

Open `main_experiment.m` in the MATLAB editor and change line ~43:
```matlab
QUICK_TEST = true;   % Changed from false
```

Then run:
```matlab
main_experiment
```

**Expected output (within ~5 minutes):**
```
[QUICK TEST MODE] Runs=5, MaxFEs=5000
============================================
  APKO: Track 2 NIA Comparison Experiment
  Algorithms  : 8
  Runs/func   : 5
  Max FEs     : 5000
  Dimension   : 30
============================================

--- CEC-2014 Benchmark (D=30) ---
  CEC14-F01 : APKO (Proposed) WOA GWO HHO SCA MPA AOA GJO
  CEC14-F02 : APKO (Proposed) WOA GWO ...
  ...
CEC2014 complete. Results saved.
...
============================================
  EXPERIMENT COMPLETE
  All results saved to: Results/Track2_NIA/
============================================
```

If this runs without errors, the setup is complete. Set `QUICK_TEST = false`
before the real experiment.

---

## PART C — Run the Full Experiments

### Step 5 — Track 2: APKO vs 7 NIA competitors

```matlab
% In main_experiment.m, ensure:
%   QUICK_TEST = false
%   N_RUNS = 50
%   MAX_FES = 60000
%   DIM = 30

main_experiment
```

**Runtime:** approximately 12–24 hours on a modern laptop (Intel i7, 16GB RAM).
MATLAB will print progress after each function. Safe to leave running overnight.

**Outputs saved to `Results/Track2_NIA/`:**
```
Results/Track2_NIA/
├── CEC2014/
│   ├── Results_Mean.csv          ← Mean ± Std per function per algo
│   ├── Results_LaTeX.tex         ← Copy-paste ready LaTeX table
│   ├── workspace_CEC14.mat       ← All raw data (50 runs × 30 funcs × 8 algos)
│   └── Figures/
│       ├── Conv_F01_CEC14.png    ← Convergence plot for F01
│       ├── Conv_F01_CEC14.pdf    ← PDF version for paper
│       └── ... (30 plots)
├── CEC2017/
│   ├── Results_Mean.csv
│   ├── Results_LaTeX.tex
│   ├── workspace_CEC17.mat
│   └── Figures/ (29 plots)
└── Engineering/
    ├── Results_Mean.csv
    ├── Results_LaTeX.tex
    └── Figures/ (5 plots)
```

---

### Step 6 — Track 1: APKO vs PKO variants (SOTA)

```matlab
main_sota_comparison
```

This also runs the **ablation study** automatically — disabling each of the
7 improvements one at a time and testing on 6 CEC2017 functions.

**Outputs saved to `Results/Track1_SOTA/`:**
```
Results/Track1_SOTA/
├── CEC2017/
│   ├── Results_Mean.csv
│   ├── Results_LaTeX.tex
│   └── Figures/
├── CEC2014/
│   └── ...
├── Engineering/
│   └── ...
├── Ablation/
│   ├── Results_Mean.csv          ← Shows contribution of each improvement
│   └── Results_LaTeX.tex
└── workspace_SOTA.mat
```

---

## PART D — Inspect Results

### Step 7 — Load and view results

```matlab
% Load Track 2 CEC2017 workspace
load('Results/Track2_NIA/CEC2017/workspace_CEC17.mat')

% Print statistical test results
disp(stat_results_17)

% View mean Friedman ranks (lower = better; column 1 = APKO)
fprintf('Mean Friedman Ranks:\n')
for a = 1:length(algo_names)
    fprintf('  %-20s : %.3f\n', algo_names{a}, stat_results_17.friedman_ranks(a))
end

% Show convergence curve for a specific function
% Function index 1 = F01, 10 = F10, etc.
f_idx = 10;
curves_f10 = conv_17(f_idx,:)';
Plot_Convergence(curves_f10, algo_names, func_names_17{f_idx}, 'my_plots/')
```

---

### Step 8 — Generate tables for report

The `.tex` files are ready to paste directly. For example:

```matlab
% View LaTeX table
type('Results/Track2_NIA/CEC2017/Results_LaTeX.tex')
```

The table uses `\textbf{}` for the best result on each row and `\toprule /
\midrule / \bottomrule` from the `booktabs` package. Add to your LaTeX
preamble:
```latex
\usepackage{booktabs}
\usepackage{resizebox}   % for \resizebox
```

---

## PART E — Run APKO on a Custom Problem

```matlab
setup_paths()

% Define your problem
myFobj = @(x) sum(x.^2);   % Sphere function (minimise)
Dim = 30;
LB  = -100;
UB  =  100;

% Run APKO
Popsize      = 30;
Maxiteration = 2000;   % = 60,000 FEs / 30 agents

[best_f, best_x, conv] = APKO(Popsize, Maxiteration, LB, UB, Dim, myFobj);

fprintf('Best fitness found: %.6e\n', best_f)
fprintf('Best position: %s\n', mat2str(best_x, 4))

% Plot convergence
figure
semilogy(conv, 'b-', 'LineWidth', 2)
xlabel('Iteration'); ylabel('Best Fitness (log)')
title('APKO Convergence on Custom Problem')
grid on
```

---

## Quick Reference: Key Parameters

| Parameter | Location | Default | Notes |
|-----------|----------|---------|-------|
| `N_RUNS` | `main_experiment.m` | 50 | Independent runs per function |
| `MAX_FES` | `main_experiment.m` | 60000 | FE budget per run |
| `DIM` | `main_experiment.m` | 30 | Problem dimension |
| `POPSIZE` | `main_experiment.m` | 30 | Population size |
| `QUICK_TEST` | `main_experiment.m` | false | Set true for 5-run debug |
| `BF0` | `APKO.m` line 53 | 8 | Biological baseline BF |
| `gamma_BF` | `APKO.m` line 54 | 1.5 | BF decay exponent |
| `H_entr` | `APKO.m` line 58 | 0.15 | Entropy restart threshold |
| `k_restart` | `APKO.m` line 59 | 0.2 | Fraction of agents restarted |

---

## Troubleshooting

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `Undefined function 'APKO'` | Path not set | Run `setup_paths()` |
| `Undefined function 'cec17_func'` | MEX not compiled | Repeat Step 2 |
| `Undefined function 'TentOBL_Init'` | Path not set | Run `setup_paths()` |
| `Warning: CEC2017 MEX not found` | MEX missing | Run Step 2; results use fallback approximations |
| `mex: compile error` | No C compiler | Run `mex -setup` in MATLAB |
| Results are all `Inf` | Fobj error or bounds issue | Check problem bounds; set `QUICK_TEST=true` and add `try/catch` breakpoints |
| Very slow (>24h for 50 runs) | Expensive Fobj | Reduce `N_RUNS=20` temporarily for a preview |

---

*APKO Project v1.1 | Bug-Fixed Release | April 2026*
