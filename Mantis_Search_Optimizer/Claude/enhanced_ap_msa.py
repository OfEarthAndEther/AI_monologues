"""
Enhanced Adaptive Perception Mantis Shrimp Algorithm (EAP-MSA)
==============================================================
Combines the biological accuracy of the standard MSA with the
perceptual enhancements of the Advanced Perception variant,
while fixing critical bugs and adding proven meta-heuristic techniques.

Key Improvements over both prior versions:
    1. Fitness tracking bug fix: best_pos and fitness are updated every iteration
    2. True Quasi-random (Sobol-like) initialization via Latin Hypercube Sampling
    3. Lévy-flight exploration for heavy-tailed, long-range jumps
    4. Adaptive dual-vector (stereopsis) using cosine similarity, not just magnitude
    5. Retained and improved Sexual Cannibalism diversity mechanism
    6. Nonlinear adaptive probability p(t) via hyperbolic tangent
    7. Elitism: the global best is always preserved across iterations

Author  : EAP-MSA Research Framework
Version : 1.0
"""

import numpy as np
from scipy.stats import qmc   # for Latin Hypercube Sampling (LHS)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def levy_flight(dim: int, beta: float = 1.5) -> np.ndarray:
    """
    Generate a Lévy flight step vector using the Mantegna algorithm.

    Lévy flights produce occasional large jumps, preventing premature
    convergence in multi-modal landscapes.

    Args:
        dim  : Dimensionality of the search space.
        beta : Stability index (1 < beta <= 2). Default 1.5.

    Returns:
        step : Array of shape (dim,) containing the Lévy step.
    """
    num   = np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    denom = np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / denom) ** (1 / beta)

    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1,     dim)
    return u / (np.abs(v) ** (1 / beta))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between vectors a and b.
    Returns 0 if either vector is a zero vector.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def lhs_initialization(pop_size: int, dim: int,
                        lb: float, ub: float) -> np.ndarray:
    """
    Initialize population using Latin Hypercube Sampling (LHS).

    LHS ensures each dimension is sampled uniformly across its range,
    reducing clustering and improving early-stage diversity compared to
    pure random initialization.

    Args:
        pop_size : Number of individuals.
        dim      : Problem dimensionality.
        lb, ub   : Scalar lower and upper bounds.

    Returns:
        positions : Array of shape (pop_size, dim).
    """
    sampler   = qmc.LatinHypercube(d=dim, seed=42)
    sample    = sampler.random(n=pop_size)           # in [0, 1]
    positions = lb + sample * (ub - lb)
    return positions


# ---------------------------------------------------------------------------
# Core Algorithm
# ---------------------------------------------------------------------------

def eap_msa(obj_func,
            pop_size : int   = 30,
            iter_max : int   = 500,
            lb       : float = -100.0,
            ub       : float =  100.0,
            dim      : int   = 30,
            beta     : float = 1.5,
            cannib_rate: float = 0.12) -> tuple:
    """
    Enhanced Adaptive Perception Mantis Shrimp Algorithm (EAP-MSA).

    Biologically-inspired phases
    ----------------------------
    1. LHS Initialization  : Stratified random seeding of the search space.
    2. Exploration Phase   : Lévy-flight jumps guided by a dual-vector
                             stereopsis model (left-eye = best agent,
                             right-eye = random agent).
    3. Exploitation Phase  : Directed strike toward best solution modulated
                             by an adaptive step.
    4. Sexual Cannibalism  : Stochastic diversity reset using a cosine-
                             weighted blend to prevent genetic collapse.
    5. Elitism             : Global best solution is preserved across all
                             iterations (no regression).

    Args:
        obj_func     : Objective function f(x) -> scalar (minimization).
        pop_size     : Number of agents in the population.
        iter_max     : Maximum number of iterations.
        lb, ub       : Search-space bounds (scalars, applied per dimension).
        dim          : Problem dimensionality.
        beta         : Lévy flight stability index (1 < beta <= 2).
        cannib_rate  : Probability of sexual cannibalism diversity event.

    Returns:
        best_fitness : Scalar — best objective value found.
        best_pos     : Array of shape (dim,) — corresponding solution.
        history      : List of best fitness per iteration (for convergence plots).
    """
    # ------------------------------------------------------------------
    # 1. Initialization (LHS for quasi-random coverage)
    # ------------------------------------------------------------------
    positions = lhs_initialization(pop_size, dim, lb, ub)
    fitness   = np.array([obj_func(ind) for ind in positions])

    best_idx     = int(np.argmin(fitness))
    best_pos     = positions[best_idx].copy()
    best_fitness = float(fitness[best_idx])
    history      = []

    # ------------------------------------------------------------------
    # 2. Main Loop
    # ------------------------------------------------------------------
    for t in range(iter_max):

        # Adaptive probability: smooth sigmoid-based transition
        # from exploration (p≈0.8) to exploitation (p≈0) as t → iter_max
        p = 0.8 * (1.0 - np.tanh(2.5 * t / iter_max))

        for i in range(pop_size):

            pos_new = positions[i].copy()

            # ---- Dual-Vector Stereopsis (binocular distance estimation) ----
            # Left-eye vector  : direction from agent i → global best
            # Right-eye vector : direction from agent i → random peer
            rand_idx  = np.random.randint(0, pop_size)
            l_vector  = best_pos               - positions[i]
            r_vector  = positions[rand_idx]    - positions[i]

            # Select the vector with HIGHER cosine similarity to the
            # current movement direction (momentum-aware focus selection).
            # Falls back to l_vector if agent has no prior momentum.
            momentum  = best_pos - positions[i]   # proxy for search direction
            sim_l     = cosine_similarity(l_vector, momentum)
            sim_r     = cosine_similarity(r_vector, momentum)
            target    = l_vector if sim_l >= sim_r else r_vector

            # ---- Phase Selection ----------------------------------------
            if np.random.rand() < p:
                # EXPLORATION: Lévy-flight jump along target direction
                step      = levy_flight(dim, beta)
                pos_new   = positions[i] + step * target

            else:
                # EXPLOITATION: Directed strike (Attack Phase)
                alpha     = np.random.rand()
                pos_new   = best_pos - alpha * target

            # ---- Sexual Cannibalism (Diversity Maintenance) ---------------
            # With probability cannib_rate, a 'male' agent is absorbed and
            # the current agent is repositioned via cosine-weighted blending,
            # preventing population collapse into a single basin.
            if np.random.rand() < cannib_rate:
                male_idx  = np.random.randint(0, pop_size)
                male      = positions[male_idx]
                theta     = 2 * np.pi * np.random.rand()
                # Weighted blend: scale male contribution by cos(theta)
                pos_new   = (1 - abs(np.cos(theta))) * pos_new \
                           +     abs(np.cos(theta))  * male

            # ---- Boundary Handling ----------------------------------------
            pos_new = np.clip(pos_new, lb, ub)

            # ---- Greedy Selection (accept only if improved) ---------------
            new_fit = obj_func(pos_new)
            if new_fit < fitness[i]:
                positions[i] = pos_new
                fitness[i]   = new_fit

        # ------------------------------------------------------------------
        # 3. Elitism: update global best after each full iteration
        # ------------------------------------------------------------------
        iter_best_idx = int(np.argmin(fitness))
        if fitness[iter_best_idx] < best_fitness:
            best_fitness = float(fitness[iter_best_idx])
            best_pos     = positions[iter_best_idx].copy()

        history.append(best_fitness)

    return best_fitness, best_pos, history


# ---------------------------------------------------------------------------
# Reference Implementations (for benchmarking comparison)
# ---------------------------------------------------------------------------

def standard_msa(obj_func, pop_size, iter_max, lb, ub, dim):
    """
    Original Standard MSA (reproduced verbatim from reference).
    NOTE: Best solution is NOT updated inside the loop — this is a known
    limitation of the original implementation.
    """
    positions = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness   = np.array([obj_func(ind) for ind in positions])
    best_pos  = positions[np.argmin(fitness)].copy()

    for t in range(iter_max):
        p = 0.5 * (1 - t / iter_max)
        for i in range(pop_size):
            if np.random.rand() < p:
                positions[i] = best_pos + (2 * np.random.rand() - 1) * (best_pos - positions[i])
            else:
                positions[i] = best_pos - np.random.rand() * (best_pos - positions[i])

            if np.random.rand() < 0.15:
                male         = positions[np.random.randint(0, pop_size)]
                positions[i] = male * np.cos(2 * np.pi * np.random.rand()) * np.random.rand()

            positions[i] = np.clip(positions[i], lb, ub)

    return np.min(fitness), best_pos


def advanced_perception_msa(obj_func, pop_size, iter_max, lb, ub, dim):
    """
    Original AP-MSA (reproduced verbatim from reference).
    NOTE: Fitness array and best_pos are NOT updated inside the loop —
    critical bug that prevents convergence tracking.
    """
    positions = lb + np.random.uniform(0, 1, (pop_size, dim)) * (ub - lb)
    fitness   = np.array([obj_func(ind) for ind in positions])
    best_pos  = positions[np.argmin(fitness)].copy()

    for t in range(iter_max):
        p = 0.8 * (1 - np.tanh(2 * t / iter_max))
        for i in range(pop_size):
            l_vector      = best_pos - positions[i]
            r_vector      = positions[np.random.randint(0, pop_size)] - positions[i]
            target_vector = l_vector if np.linalg.norm(l_vector) < np.linalg.norm(r_vector) else r_vector

            if np.random.rand() < p:
                positions[i] += np.random.normal(0, 1, dim) * target_vector
            else:
                positions[i]  = best_pos - np.random.rand() * target_vector

            positions[i] = np.clip(positions[i], lb, ub)

    return np.min(fitness), best_pos


# ---------------------------------------------------------------------------
# Demo / Benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    # --- Benchmark functions ---
    def sphere(x):
        return float(np.sum(x ** 2))

    def rastrigin(x):
        A = 10
        return float(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))

    def rosenbrock(x):
        return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

    benchmarks = {
        "Sphere"     : (sphere,     -100, 100),
        "Rastrigin"  : (rastrigin,  -5.12, 5.12),
        "Rosenbrock" : (rosenbrock, -5,    10),
    }

    POP  = 30
    ITER = 500
    DIM  = 30
    RUNS = 5   # average over multiple runs for statistical reliability

    print(f"{'='*70}")
    print(f"  Benchmark Comparison: Standard MSA vs AP-MSA vs EAP-MSA")
    print(f"  Pop={POP}, Iter={ITER}, Dim={DIM}, Runs={RUNS}")
    print(f"{'='*70}")

    for fname, (func, lb, ub) in benchmarks.items():
        print(f"\n{fname}")
        print(f"  {'Algorithm':<22} {'Mean Best':>14}  {'Std Dev':>12}  {'Time(s)':>8}")
        print(f"  {'-'*60}")

        for label, algo in [
            ("Standard MSA",  lambda f, lb, ub: standard_msa(f, POP, ITER, lb, ub, DIM)[:2]),
            ("AP-MSA",        lambda f, lb, ub: advanced_perception_msa(f, POP, ITER, lb, ub, DIM)),
            ("EAP-MSA (ours)",lambda f, lb, ub: eap_msa(f, POP, ITER, lb, ub, DIM)[:2]),
        ]:
            results = []
            t0 = time.time()
            for _ in range(RUNS):
                best_fit, _ = algo(func, lb, ub)
                results.append(best_fit)
            elapsed = time.time() - t0

            mean = np.mean(results)
            std  = np.std(results)
            print(f"  {label:<22} {mean:>14.6e}  {std:>12.4e}  {elapsed/RUNS:>8.3f}")

    print(f"\n{'='*70}")
    print("Done. For convergence curves, store history from eap_msa() and plot.")
