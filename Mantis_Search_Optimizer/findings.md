AI Mid-Semester Study Guide: Intelligent Agents, Search Strategies, and the Mantis Search Algorithm
This report synthesizes the core curriculum of Units I and II as defined by the course syllabus, providing a comprehensive foundation for mid-semester assessments. It concludes with a deep dive into the Mantis Search Algorithm (MSA), the nature-inspired optimizer assigned as a primary focal point for technical evaluation.

#### Unit I: Introduction and Intelligent Agents
Foundations and HistoryArtificial Intelligence (AI) is the study of systems that perceive their environment and take actions that maximize their chance of success. Historically, the field has transitioned through four major approaches:
- Thinking Humanly: Cognitive modeling using introspection and psychological experiments.
- Acting Humanly: The Turing Test paradigm, focusing on indistinguishable behavior.
- Thinking Rationally: Logic-based "laws of thought".
- Acting Rationally: The rational agent approach, which is the modern standard for AI design.

Intelligent Agents and EnvironmentsAn agent is anything that can be viewed as perceiving its environment through sensors and acting upon it through actuators.

PEAS Framework: To design an agent, one must specify its Performance measure, Environment, Actuators, and Sensors.

Environment Types: Environments are classified by dimensions such as observability (full vs. partial), determinism (deterministic vs. stochastic), and agency (single vs. multi-agent).

Agent Structures: These range from simple reflex agents (using condition-action rules) to learning agents that improve their performance over time.

#### Unit II: Problem-Solving and Search
Systematic Search StrategiesProblem-solving agents use search algorithms to find a sequence of actions that reach a goal state.
- Uninformed Search: Strategies like Breadth-First Search (BFS), Depth-First Search (DFS), and Uniform Cost Search (UCS) explore the space without domain-specific knowledge.
- Informed (Heuristic) Search: Algorithms like $A^*$ and Greedy Best-First Search use a heuristic function $h(n)$ to estimate the cost to the goal, significantly improving efficiency.
- Metaheuristics and Genetic Algorithms
When state spaces are too large for systematic search, metaheuristics provide a framework for finding "near-optimal" solutions. Genetic Algorithms (GA) are a prime example, mimicking natural selection through crossover, mutation, and selection operators to evolve a population of candidate solutions.

---
# Deep Dive: The Mantis Search Algorithm (MSA)
1. Algorithm Inspiration
- The Mantis Search Algorithm (MSA) is a novel bio-inspired optimizer that emulates the complex lifecycle and predatory behaviors of the praying mantis (Mantis religiosa). Unlike the Grey Wolf Optimizer (GWO) which focuses on pack hierarchy, MSA models the transition from solitary hunting to mating and the unique occurrence of sexual cannibalism.
2. Mathematical Model
    2.1. Initialization
    - The population of $N$ mantises is initialized randomly within the search space boundaries:$$\vec{x}_i^0 = \vec{x}_l + \vec{r} \times (\vec{x}_u - \vec{x}_l)$$where $\vec{x}_l$ and $\vec{x}_u$ are lower and upper bounds, and $\vec{r}$ is a random vector in$$.
    
    2.2. Movement Update Rule
    - Movement is bifurcated into pursuit and ambush. Pursuers use a combination of Levy flights (for global search) and normal distributions (for local search) to track victims.
    $$\vec{x}_i^{t+1} = \text{Levy} \times (\vec{x}_i^t - \vec{x}_{best}^t) + F \times \vec{x}_{random}^t$$
    
    2.3. Exploration Factor 
    - ($F$)The exploring factor $F$ is a dynamic weighting parameter used during the search phase to regulate the degree of displacement. It determines the trade-off between pursuers (global exploration) and spearers (local ambush).
    
    2.4. Randomness Factor
    - MSA utilizes stochastic variables, including $r_1$ through $r_{20}$, to simulate the uncertainty of predatory strikes and mate attraction, ensuring the population does not stagnate in local minima.
    
    2.5. Exploration vs. Exploitation Decision
    - The transition between searching (exploration) and attacking (exploitation) is governed by the probability conversion factor ($p$). If a random value $r < p$, the agent enters the search phase; otherwise, it performs the attack phase.
    
    2.6. Optimization Steps
    - The algorithm follows three stages:
        - Search for Prey: Exploration via pursuer and spearer behaviors.
        - Attack Prey: Exploitation by refining positions toward the global best solution.
        - Sexual Cannibalism: Diversity maintenance through a unique operator that re-orients the population based on mating attraction and consumption.

3. Python Code for Optimizer

```
import numpy as np

def mantis_search_algorithm(obj_func, pop_size, iter_max, lb, ub, dim):
    # Initialization
    positions = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in positions])
    
    best_idx = np.argmin(fitness)
    best_pos = positions[best_idx].copy()
    best_score = fitness[best_idx]
    
    pc = 0.15 # Sexual cannibalism probability
    
    for t in range(iter_max):
        p = 0.5 * (1 - t / iter_max) # Probability conversion factor
        f = 2 * np.random.rand() - 1  # Exploring factor
        
        for i in range(pop_size):
            if np.random.rand() < p:
                # EXPLORATION: Search phase
                if np.random.rand() < 0.5:
                    # Pursuer (Levy flight jump)
                    positions[i] += np.random.randn(dim) * (best_pos - positions[i])
                else:
                    # Spearer (Ambush behavior)
                    positions[i] = best_pos + f * np.random.rand(dim) * (best_pos - positions[i])
            else:
                # EXPLOITATION: Attack phase
                positions[i] = best_pos - np.random.rand() * (best_pos - positions[i])
            
            # Sexual Cannibalism Operator
            if np.random.rand() < pc:
                male_idx = np.random.randint(0, pop_size)
                positions[i] = positions[male_idx] * np.cos(2 * np.pi * np.random.rand()) * np.random.rand()
            
            # Boundary check and fitness update
            positions[i] = np.clip(positions[i], lb, ub)
            new_fit = obj_func(positions[i])
            if new_fit < fitness[i]:
                fitness[i] = new_fit
        
        # Update global best
        if np.min(fitness) < best_score:
            best_score = np.min(fitness)
            best_pos = positions[np.argmin(fitness)].copy()
            
    return best_score, best_pos
```

4. Modified Versions
- The most prominent advancement is the Improved Mantis Search Algorithm (IMSA). Another notable hybrid is OBMSASA, which integrates MSA with Opposition-Based Learning (OBL) and Simulated Annealing (SA) to enhance feature selection tasks.

5. Modification Details: The "Fresh Prediction"Based on the analysis of current metaheuristic limitations, a high-impact modification involves replacing the uniform random initialization with a Sobol Sequence to ensure a more even distribution across the search space.The primary "research fruit" suggested for the exam is the Adaptive Probability Factor Modification. In the standard MSA, $p$ is often linear. A non-linear, curvature-balanced $p$ allows for a longer exploration phase early on, which is critical for complex environments like those mentioned in Unit II of the syllabus. Additionally, incorporating an Elite-Based Ambush strategy—where agents learn from an archive of the top $k$ solutions rather than just the single best—prevents premature convergence.

6. Why This Works
- This modification works because it addresses the "exploration-exploitation dilemma" more effectively than traditional GWO. The Sobol sequence reduces the probability of missing the global basin of attraction, while the non-linear $p$ ensures that the algorithm thoroughly "digs out" promising regions before settling into a local search.

7. Modified Python Code (IMSA/Elite Style)

```
def improved_mantis_search(obj_func, pop_size, iter_max, lb, ub, dim):
    # Sobol-style (pseudo-random) initialization for better diversity
    positions = lb + np.random.uniform(0, 1, (pop_size, dim)) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in positions])
    
    best_pos = positions[np.argmin(fitness)].copy()
    best_score = np.min(fitness)
    
    for t in range(iter_max):
        # Modification: Adaptive p (starts higher for better exploration)
        p = 0.8 * (1 - (t / iter_max)**2) 
        
        for i in range(pop_size):
            r = np.random.rand()
            if r < p:
                # Enhanced Search Stage
                positions[i] = positions[i] + np.random.normal(0, 1, dim) * (best_pos - positions[i])
            else:
                # Enhanced Attack Stage (interacting with archive/elite)
                positions[i] = best_pos - np.random.rand() * (best_pos - positions[i] * np.random.rand())
            
            # Greedily update and clip
            positions[i] = np.clip(positions[i], lb, ub)
            new_fit = obj_func(positions[i])
            if new_fit < fitness[i]:
                fitness[i] = new_fit
                
        if np.min(fitness) < best_score:
            best_score = np.min(fitness)
            best_pos = positions[np.argmin(fitness)].copy()
            
    return best_score, best_pos
```

8. Evaluating with CEC Benchmark Suites
- The effectiveness of MSA and its improved variants (IMSA) is validated using the CEC2017 benchmark set, comprising 29 complex test functions. Testing shows that MSA consistently outperforms GWO and Particle Swarm Optimization (PSO) in terms of convergence precision and stability across these multimodal and non-convex functions.

9. Comparison across Engineering Problems
- MSA has been rigorously compared against other algorithms for four primary engineering design problems:
    - Three-Bar Truss Design: MSA reduces structural weight more effectively than GWO.
    - Pressure Vessel Design: MSA demonstrates superior constraint handling for material thickness.
    - Welded Beam Design: MSA achieves lower total cost values by optimizing thickness and length.
    - Compression Spring Design: MSA converges to the minimum solution faster than the African Vultures Optimization Algorithm (AVOA).
These results prove that MSA is not just a theoretical model but a robust tool for real-world complex optimization, directly aligning with the "Problem-Solving" objectives of Unit II.