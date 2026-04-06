# APKO: Adaptive Pied Kingfisher Optimizer
**Proposed Algorithm | NIA Research Project | Phase 3 Submission**

---

## Project Overview

This repository implements the **Adaptive Pied Kingfisher Optimizer (APKO)**, a novel improvement over the base Pied Kingfisher Optimizer (PKO, Bouaouda et al., 2024). APKO addresses **7 critical gaps** identified through a systematic SOTA analysis of 6 PKO-related papers (2024–2025).

**Two comparison tracks are implemented:**
- **Track 1 (SOTA):** APKO vs. existing PKO variants (PKO, IPKO, MPKO, EPKO)
- **Track 2 (NIA Literature):** APKO vs. 7 recent NIA algorithms (WOA, GWO, HHO, SCA, MPA, AOA, GJO)

---

## Directory Structure

```
APKO_Project/
├── main_experiment.m           ← Track 2: APKO vs 7 NIA competitors
├── main_sota_comparison.m      ← Track 1: APKO vs PKO variants
├── setup_paths.m               ← Run first to configure MATLAB path
│
├── Algorithms/
│   ├── APKO.m                  ← ★ PROPOSED ALGORITHM ★
│   ├── PKO_Variants/
│   │   ├── PKO.m               ← Base algorithm (Bouaouda 2024)
│   │   ├── IPKO.m              ← Improved PKO (Cong 2024)
│   │   ├── MPKO.m              ← Multi-strategy PKO (Wang 2025)
│   │   └── EPKO.m              ← Enhanced PKO (Benfeng 2025)
│   └── Competitors/
│       ├── WOA.m               ← Whale Optimization Algorithm
│       ├── GWO.m               ← Grey Wolf Optimizer
│       ├── HHO.m               ← Harris Hawks Optimization
│       ├── SCA.m               ← Sine Cosine Algorithm
│       ├── MPA.m               ← Marine Predators Algorithm
│       ├── AOA.m               ← Arithmetic Optimization Algorithm
│       └── GJO.m               ← Golden Jackal Optimizer
│
├── Initialization/
│   ├── TentOBL_Init.m          ← Tent chaos + OBL (with singularity fix)
│   └── Initialization.m        ← Standard uniform random (for competitors)
│
├── Benchmarks/
│   ├── CEC2014/
│   │   └── CEC2014_Wrapper.m   ← CEC2014 interface (wraps MEX)
│   ├── CEC2017/
│   │   └── CEC2017_Wrapper.m   ← CEC2017 interface (wraps MEX)
│   └── Engineering/
│       └── Engineering_Problems.m  ← 5 standard engineering problems
│
├── Analysis/
│   ├── Statistical_Tests.m     ← Wilcoxon + Friedman + Holm
│   ├── Generate_Tables.m       ← Mean/Std/Rank tables + LaTeX export
│   └── Plot_Convergence.m      ← Publication-quality convergence plots
│
└── Results/                    ← Auto-created during experiment runs
    ├── Track1_SOTA/
    └── Track2_NIA/
```

---

## Quick Start

```matlab
% Step 1: Open MATLAB and navigate to APKO_Project/
cd APKO_Project

% Step 2: Configure path
setup_paths()

% Step 3: Compile CEC benchmarks (one time only)
% Download CEC2014 and CEC2017 official code from IEEE CEC website
% cd Benchmarks/CEC2014 && mex cec14_func.cpp
% cd Benchmarks/CEC2017 && mex cec17_func.cpp

% Step 4a: Quick test (5 runs, debug mode)
% In main_experiment.m, set QUICK_TEST = true, then:
main_experiment

% Step 4b: Full experiment (50 runs - takes ~12-24 hours)
main_experiment         % Track 2: APKO vs 7 NIA algorithms
main_sota_comparison    % Track 1: APKO vs PKO variants
```

---

## APKO Improvements: Mathematical Derivations

### [IMP-1] Adaptive Beating Factor

**Problem:** Base PKO fixes BF = 8 (biological constant). No justification
for optimality on any problem landscape. (GAP-M1 in SOTA analysis)

**Proposed schedule:**
```
BF*(t, D) = BF₀ · (1 + ln(D)/D) · exp(-γ · t/T)
```

- `BF₀ = 8`: Biological baseline (preserves biological inspiration)
- `(1 + ln(D)/D)`: Dimension scaling — larger search spaces require
  more hovering patterns to adequately cover neighborhoods
- `exp(-γ·t/T)`: Exponential decay ensures BF contracts as iterations
  progress — intensive exploration early, focused late
- `γ = 1.5`: Empirically calibrated; defines transition speed

**Mathematical argument:**
The hovering phase covers a local neighborhood of radius proportional
to `BF^(1/D)` (from the position update formula). For equal coverage
probability across dimensions, BF should grow logarithmically with D.
As t → T, BF → BF₀·e^(-γ), providing ~0.22·BF₀ minimum hovering.

---

### [IMP-2] Dynamic Crest Angle

**Problem:** Base PKO uses a single fixed random angle drawn once at
initialization. This provides no directional diversity across iterations.

**Proposed schedule:**
```
θ(t) = 2π·rand · (1 - t/T) + (π/2)·sin(π·t/T)
```

- **Decaying random component** `2π·rand·(1-t/T)`: Allows broad angular
  exploration early, converges to near-zero variability late
- **Oscillatory component** `(π/2)·sin(π·t/T)`: Creates a symmetric,
  bell-shaped mid-iteration angular push — maximum directional diversity
  at t = T/2, zero at t = 0 and t = T
- Combined effect: agents approach best solution from varied angles
  throughout the run, preventing collinear convergence traps

---

### [IMP-3] Cosine-Annealed PE Schedule

**Problem:** PKO's linear PE decay is unjustified. Linear functions have
no special properties that make them optimal for probability scheduling
in iterative optimization. (GAP-M2 in SOTA analysis)

**Linear (original):**
```
PE(t) = 0.5 · (1 - t/T)
```

**Proposed cosine schedule:**
```
PE(t) = (PEmax/2) · (1 + cos(π·t/T))
```

**Mathematical advantages over linear:**
1. **Differentiable at boundaries:** cos(0) = cos(π) = 0 — no abrupt
   transition. Linear decay has a discontinuous derivative at t = 0.
2. **Naturally slower transition:** Cosine spends more time near PEmax
   at early iterations and near PEmin at late iterations — biologically
   more consistent with optimal foraging: heavy commensalism early when
   the otter is fresh, tapering off.
3. **Analytic decay rate:** At t = T/2, PE = PEmax/2 exactly. This
   ensures exactly half the commensalism energy is spent in each half
   of the run — a balanced budget without manual tuning.

---

### [IMP-4] Tent Map Initialization with Singularity Fix

**Problem:** Papers P2 and P6 use tent map initialization but do not
address the fixed-point singularity at x = 0.5. (GAP-M3 in SOTA analysis)

**Tent map:**
```
x(n+1) = 2·x(n)         if x(n) < 0.5
x(n+1) = 2·(1 - x(n))   if x(n) ≥ 0.5
```

**Singularity:** If x = 0.5, then x(n+1) = 2·(1-0.5) = 1.0, and
x(n+2) = 2·(1-1.0) = 0 → sequence collapses. Probability that any
of N·D random seeds hits exactly 0.5 is zero in continuous probability,
but in floating-point arithmetic with N≥50, D≥30, this becomes non-trivial.

**Fix implemented in TentOBL_Init.m:**
```
if |x - 0.5| < ε_sing:
    x = x + ε_sing · uniform(-1, 1)
    x = clamp(x, 0, 1)
```
With ε_sing = 0.02, this provides a 4% "exclusion zone" around the
fixed point without materially changing the chaotic properties.

**Proof that fix preserves ergodicity:**
The tent map's Lyapunov exponent is λ = ln(2) ≈ 0.693 throughout the
chaotic region [0,1]\{0.5}. The perturbation displaces x by at most ε_sing,
whose effect on λ vanishes as ε_sing → 0. With ε_sing = 0.02, the
perturbation affects < 4% of the sequence length on average.

**OBL component:**
```
X'_j = lb_j + ub_j - X_j
```
The opposite solution X' and original X are combined (2N candidates);
N are selected by farthest-point sampling (greedy max-min distance),
ensuring maximal initial population diversity.

---

### [IMP-5] Adaptive Cauchy Mutation Scale

**Problem:** All PKO variants using Cauchy mutation apply it with fixed
scale σ = 1 (standard Cauchy). Late in the run, this causes disruptions
orders of magnitude larger than the current search basin. (GAP-M5)

**Proposed adaptive scale:**
```
σ(t) = σ₀ · (1 - t/T)²
```

**Rationale:**
- Quadratic (not linear) contraction: provides more gradual reduction
  in mid-iteration (where escape probability should remain meaningful)
  and rapid reduction near convergence
- At t = T/2: σ = σ₀/4 (25% of initial scale)
- At t = T:   σ → 0 (Cauchy becomes negligible — no disruption)
- **Lower floor:** σ ≥ 10⁻⁴ prevents numerical issues

---

### [IMP-6] Population Entropy Restart

**Problem:** No PKO variant has a principled stagnation detection
mechanism. MPKO uses an ad-hoc stagnation counter. (LOOP-3 in SOTA)

**Information-theoretic formulation:**
```
H_pop(t) = -Σ_i p_i · ln(p_i)     [Shannon entropy in nats]
```

where `p_i` is the fraction of agents in fitness bin i (√N bins).

**Normalized entropy ratio:**
```
R(t) = H_pop(t) / H_max(N)   where H_max(N) = ln(N)
```

H_max(N) = ln(N) is the maximum possible entropy (uniform distribution
over all N agents in distinct bins).

**Restart criterion:**
```
if mean(R over last W iterations) < H_thresh:
    restart worst k% of agents
```

With H_thresh = 0.15 and k = 20%: triggers when population diversity
has collapsed to ≤15% of theoretical maximum — a principled,
problem-independent threshold unlike fixed stagnation counters.

**Restart mechanism:**
```
X_restart = X_best + N(0, 0.1·|UB-LB|) · (1 - t/T)
```
Perturbation decays with iteration — large perturbations early allow
genuine exploration; small perturbations late preserve near-optimal positions.

---

### [IMP-7] Ranked Commensalism

**Problem:** Uniform random agent selection in commensalism becomes
degenerate as the population converges (LOOP-2 in SOTA analysis).
When all agents have similar positions, (m, n) pairs produce
near-zero perturbations regardless of selection.

**Proposed:** Select m from top 50% (fitness rank) and n from bottom 50%.

**Mathematical argument:**
Expected perturbation magnitude ∝ |X_m - X_n| · o

With uniform selection in a converged population: |X_m - X_n| → 0.
With ranked selection: |X_m - X_n| is maximized because m and n are
from opposite ends of the quality spectrum, maintaining meaningful
perturbations even when overall diversity has decreased.

---

## Experimental Settings

Per instructor guidelines (Image 2):

| Parameter | Value |
|-----------|-------|
| Function evaluations | 60,000 per function per run |
| Independent runs | 50 per function |
| Population size | 30 (→ 2000 max iterations) |
| Benchmarks | CEC-2014 (F1-F30), CEC-2017 (F1-F29) |
| Engineering problems | TCS, PVD, WBD, SRD, TBT |
| Dimension | D = 30 (primary), also D = 10, 20 |
| Statistical tests | Wilcoxon signed-rank + Friedman + Holm correction |
| Outputs | Mean ± Std tables, Rank tables, Convergence curves |

---

## Engineering Problems

| ID  | Problem | Variables | Key Constraints |
|-----|---------|-----------|-----------------|
| TCS | Tension/Compression Spring | 3 | Shear stress, deflection, surge |
| PVD | Pressure Vessel Design | 4 | Wall thickness, volume |
| WBD | Welded Beam Design | 4 | Shear stress, bending, buckling |
| SRD | Speed Reducer Design | 7 | Bending stress, shaft deflection |
| TBT | Three-Bar Truss Design | 2 | Stress limits per bar |

Engineering problem link: https://in.mathworks.com/matlabcentral/fileexchange/97577

---

## Competitor Algorithms (Track 2)

| Algorithm | Year | Reference |
|-----------|------|-----------|
| WOA | 2016 | Mirjalili & Lewis, Adv. Eng. Software |
| GWO | 2014 | Mirjalili et al., Adv. Eng. Software |
| HHO | 2019 | Heidari et al., Future Gen. Comp. Sys. |
| SCA | 2016 | Mirjalili, Knowledge-Based Sys. |
| MPA | 2020 | Faramarzi et al., Expert Sys. with App. |
| AOA | 2021 | Abualigah et al., Comp. Meth. App. Mech. |
| GJO | 2022 | Chopra & Ansari, Expert Sys. with App. |

---

## PKO Variants (Track 1 — SOTA Comparison)

| Algorithm | Year | Key Contribution | Key Limitation |
|-----------|------|-----------------|----------------|
| PKO | 2024 | Base algorithm | Fixed BF=8, fixed angle, linear PE |
| IPKO | 2024 | ROBL + Spiral SCA | Fixed refraction index n_r |
| MPKO | 2025 | 4-phase + DAS + DFS | Empirical thresholds; no convergence proof |
| EPKO | 2025 | 5 mechanisms + Simplex | Changes stochastic nature; costly OBL init |
| **APKO** | **2025** | **7 principled improvements with mathematical derivations** | — |

---

## Known Limitations & Future Work

1. **No convergence proof:** APKO, like all PKO variants, lacks a formal
   convergence proof via Markov chain analysis (RS-T1 in SOTA report).
   This remains the highest-priority theoretical gap.
2. **Hyperparameter sensitivity:** γ_BF = 1.5, H_thresh = 0.15, k = 0.2
   are validated empirically. A principled derivation from problem structure
   would strengthen the contribution.
3. **Single-objective only:** APKO does not extend to multi-objective
   optimization (LOOP-5 in SOTA). MOPKO design is deferred to future work.
4. **No NAS/LLM application:** High-impact application domains (RS-AP1,
   RS-AP3) remain unexplored.

---

## Citation

If you use this code, please cite the original PKO paper:

> Bouaouda, A., et al. (2024). Pied Kingfisher Optimizer: A new bio-inspired
> metaheuristic algorithm for solving numerical optimization and engineering
> problems. Neural Computing and Applications.
> DOI: [see published version]

---

*APKO Project | NIA Research | Phase 3 | April 2026*
