# 🧠 UNINFORMED SEARCH — COMPLETE EXAM PREP GUIDE
### Based Entirely on Your Professor's Slides

---

## PART 0: FOUNDATIONS (Don't Skip — Prof Loves This)

### What is a State?
> "All information about the environment necessary to make a decision for the task at hand."

### Three Types of State Representation (Vacuum World Example)

| Type | Description | Scalability |
|------|-------------|-------------|
| **Atomic** | State = indivisible snapshot (S1, S2...S8); Actions = S×S matrices | Add 2nd roomba → state space *doubles* |
| **Propositional/Factored** | State = assignment of values to variables (Dirt-L: T/F, Dirt-R: T/F, Room: L/R); 2³ = 8 states | Add 2nd roomba → +1 variable only |
| **Relational** | State = objects + relations: In(robot, room), dirty(room) | Add rooms/roombas → only objects change |

**Key insight for paper:** Propositional is *logarithmic* in the size of the state space — extremely compact.

---

### Problem Formulation (Always 4 Components)

| Component | Definition | 8-Puzzle Example | Robotic Assembly Example |
|-----------|------------|------------------|--------------------------|
| **States** | Configurations of world | Locations of tiles | Real-valued joint angles |
| **Actions** | Legal moves | Move blank: L/R/Up/Down | Continuous joint motions |
| **Goal Test** | Is this the goal? | = goal state (given) | Complete assembly |
| **Path Cost** | Cost of solution | 1 per move | Time to execute |

---

### Search Node vs. State (A Common Exam Trap!)

- **State** = physical configuration of the world
- **Node** = data structure in the search tree containing:
  - State
  - Parent node
  - Action taken to reach this state
  - Path cost g(x)
  - Depth

The `Expand()` function uses `SuccessorFn` to create child nodes from a node.

---

## PART 1: HOW TO EVALUATE ANY SEARCH ALGORITHM

Your professor explicitly lists these — every answer must address them:

| Criterion | Definition | Variable |
|-----------|------------|----------|
| **Completeness** | Does it always find a solution if one exists? | — |
| **Time Complexity** | Number of nodes generated | b, d, m |
| **Space Complexity** | Maximum nodes in memory at once | b, d, m |
| **Optimality** | Does it find the *least-cost* solution? | — |
| **Systematicity** | Does it visit each state at most once? | — |

### Notation (Use This in Answers):
- **b** = maximum branching factor (max children per node)
- **d** = depth of the shallowest goal node (least-cost solution)
- **m** = maximum depth of the state space (can be ∞)
- **C\*** = cost of the optimal solution
- **ε** = minimum step cost (ε > 0)

---

## PART 2: ALL SEARCH ALGORITHMS — DEEP DIVE

---

### 🔵 BREADTH-FIRST SEARCH (BFS)

**Core Idea:** Expand shallowest unexpanded node first.  
**Data Structure:** FIFO Queue

#### Pseudocode:
```
function BFS(problem):
    node ← make_node(problem.initial_state)
    if GOAL_TEST(node.state): return SOLUTION(node)
    frontier ← QUEUE with node
    explored ← empty set
    
    while frontier is not empty:
        node ← DEQUEUE(frontier)          // remove from front
        explored.add(node.state)
        
        for each action in ACTIONS(node.state):
            child ← CHILD_NODE(problem, node, action)
            if child.state ∉ explored AND child.state ∉ frontier:
                if GOAL_TEST(child.state): return SOLUTION(child)
                ENQUEUE(frontier, child)   // add to back
    
    return FAILURE
```

#### Analysis:
| Metric | Value | Why |
|--------|-------|-----|
| Complete? | **Yes** (if b is finite) | Explores all nodes level by level |
| Time | **O(b^d)** | Generates all nodes up to depth d |
| Space | **O(b^d)** | Keeps ALL nodes in memory (the big problem) |
| Optimal? | **Yes, if step cost = 1** | Finds shallowest = fewest steps |

**Exam Note:** BFS is optimal ONLY when all step costs are equal. If costs vary → use Uniform Cost Search.

**The Space Problem:** If b=10, d=12, each node takes 1000 bytes → **1 TERABYTE** of memory. BFS is practically unusable for deep problems.

---

### 🔴 DEPTH-FIRST SEARCH (DFS)

**Core Idea:** Expand deepest unexpanded node first.  
**Data Structure:** LIFO Stack (or recursion)

#### Pseudocode:
```
function DFS(problem):
    frontier ← STACK with make_node(problem.initial_state)
    explored ← empty set
    
    while frontier is not empty:
        node ← POP(frontier)              // remove from top
        if GOAL_TEST(node.state): return SOLUTION(node)
        explored.add(node.state)
        
        for each action in ACTIONS(node.state):
            child ← CHILD_NODE(problem, node, action)
            if child.state ∉ explored AND child.state ∉ frontier:
                PUSH(frontier, child)     // push to top
    
    return FAILURE
```

#### Analysis:
| Metric | Value | Why |
|--------|-------|-----|
| Complete? | **No** | Can get stuck in infinite loops/branches |
| Time | **O(b^m)** | May explore entire tree (m can be ∞) |
| Space | **O(b·m)** | Only stores current path + siblings |
| Optimal? | **No** | May find a deep non-optimal solution first |

**DFS's Superpower:** Space is only O(bm) — linear in depth! This is why DFS is used when memory is limited.

**Repeated States Warning:** Failure to detect repeated states converts a LINEAR problem to an EXPONENTIAL one.

---

### 🟠 DEPTH-LIMITED SEARCH (DLS)

**Core Idea:** DFS with a predetermined depth limit ℓ. Nodes at depth ℓ are treated as if they have no successors.

#### Pseudocode:
```
function DLS(problem, limit ℓ):
    return RECURSIVE_DLS(make_node(initial_state), problem, ℓ)

function RECURSIVE_DLS(node, problem, limit):
    if GOAL_TEST(node.state): return SOLUTION(node)
    else if node.depth == limit: return CUTOFF
    else:
        cutoff_occurred ← false
        for each action in ACTIONS(node.state):
            child ← CHILD_NODE(problem, node, action)
            result ← RECURSIVE_DLS(child, problem, limit)
            if result == CUTOFF: cutoff_occurred ← true
            else if result ≠ FAILURE: return result
        
        if cutoff_occurred: return CUTOFF
        else: return FAILURE
```

#### Analysis:
| Metric | Value |
|--------|-------|
| Complete? | **Yes** (if goal is within depth ℓ) |
| Time | **O(b^ℓ)** |
| Space | **O(b·ℓ)** |
| Optimal? | **No** |

**Problem:** If ℓ < d (limit set too shallow), we miss the solution. If ℓ > d, we waste time. How do we pick ℓ? → Enter Iterative Deepening.

---

### 🟢 ITERATIVE DEEPENING SEARCH (IDS / IDDFS)

**Core Idea:** Run DLS repeatedly, increasing the limit by 1 each iteration: ℓ = 0, 1, 2, 3, ...  
**Best of both worlds:** BFS's completeness + optimality, DFS's space efficiency.

#### Pseudocode:
```
function IDS(problem):
    for depth = 0 to ∞:
        result ← DLS(problem, depth)
        if result ≠ CUTOFF: return result
```

#### Mathematical Analysis of Repeated Node Generation:
Nodes at depth d are generated once.  
Nodes at depth d-1 are generated twice (once in iteration d-1, once in d).  
Nodes at depth 1 are generated d times.

**Total nodes generated:**
```
N(IDS) = (d)·b¹ + (d-1)·b² + ... + (1)·b^d
       = Σᵢ₌₁ᵈ (d - i + 1) · bⁱ
```

For b=10, d=5:
- BFS: 1,111,111 nodes
- IDS: 123,456 nodes  
→ IDS generates only ~11% more nodes than BFS but uses far less memory!

#### Analysis:
| Metric | Value | Why |
|--------|-------|-----|
| Complete? | **Yes** | Guaranteed to find goal at finite depth |
| Time | **O(b^d)** | Same asymptotic as BFS |
| Space | **O(b·d)** | Same as DFS — linear! |
| Optimal? | **Yes** (if step cost = 1) | Finds shallowest first |

**IDS is generally preferred for large search spaces with unknown depth.**

---

### 🟡 UNIFORM COST SEARCH (UCS / Dijkstra's)

**Core Idea:** Expand the node with the LOWEST cumulative path cost g(n) first.  
**Data Structure:** Priority Queue ordered by g(n)

#### Pseudocode:
```
function UCS(problem):
    node ← make_node(problem.initial_state, g=0)
    frontier ← PRIORITY_QUEUE ordered by g, insert node
    explored ← empty set
    
    while frontier is not empty:
        node ← POP(frontier)              // lowest g(n)
        if GOAL_TEST(node.state): return SOLUTION(node)
        explored.add(node.state)
        
        for each action in ACTIONS(node.state):
            child ← CHILD_NODE(problem, node, action)
            if child.state ∉ explored AND child.state ∉ frontier:
                INSERT(frontier, child)
            elif child.state ∈ frontier with higher g:
                REPLACE old node with child  // path improvement
    
    return FAILURE
```

#### Mathematical Analysis:
Let C* = optimal solution cost, ε = minimum step cost.

The maximum depth explored ≈ ⌊C*/ε⌋

| Metric | Value | Why |
|--------|-------|-----|
| Complete? | **Yes** (if ε > 0, b finite) | Won't get trapped |
| Time | **O(b^(C*/ε))** | Can be worse than BFS when costs uniform |
| Space | **O(b^(C*/ε))** | Keeps all frontier nodes |
| Optimal? | **Yes** | Always expands cheapest first |

**Key difference from BFS:** UCS doesn't care about depth — it cares about COST. When all costs = 1, UCS ≡ BFS.

**Exam Trap:** UCS tests goal when DEQUEUING (not when generating). This ensures optimality!

---

### 🔵 BEST-FIRST SEARCH (Greedy)

**Core Idea:** Use a heuristic function h(n) to expand the node that *seems closest* to the goal.  
**Data Structure:** Priority Queue ordered by h(n)

```
h(n) = estimated cost from node n to goal
```

#### Pseudocode:
```
function BEST_FIRST_SEARCH(problem, h):
    node ← make_node(initial_state)
    frontier ← PRIORITY_QUEUE ordered by h, insert node
    explored ← empty set
    
    while frontier is not empty:
        node ← POP(frontier)
        if GOAL_TEST(node.state): return SOLUTION(node)
        explored.add(node.state)
        
        for each action in ACTIONS(node.state):
            child ← CHILD_NODE(problem, node, action)
            if child.state ∉ explored:
                INSERT(frontier, child)
    
    return FAILURE
```

#### Analysis:
| Metric | Value | Why |
|--------|-------|-----|
| Complete? | **No** (tree) / Yes (graph with explored set) | Can get stuck in loops |
| Time | **O(b^m)** worst case | Heuristic may be misleading |
| Space | **O(b^m)** | Keeps all nodes |
| Optimal? | **No** | Greedy choices don't guarantee optimality |

**Analogy:** Like navigating by "as the crow flies" distance — fast but can go down dead ends.

---

### 🔷 BRANCH AND BOUND (B&B)

**Core Idea:** A systematic method that prunes the search tree by maintaining an upper bound on the optimal solution cost. Any branch whose lower bound exceeds the current best solution is pruned ("bounded").

**Origin:** More of a general optimization framework than a pure graph search — used heavily in operations research, but appears in AI as a precursor to A*.

#### How it works:
1. Maintain a **global upper bound U** (cost of best complete solution found so far, initially ∞)
2. For each node, compute a **lower bound L(n)** on the cost of any solution reachable through n
3. If L(n) ≥ U → **prune** this branch (it can't beat the best known solution)
4. If a complete solution is found with cost < U → **update U**
5. Repeat until no nodes remain

#### Pseudocode:
```
function BRANCH_AND_BOUND(problem):
    best_solution ← NULL
    best_cost U ← ∞
    frontier ← PRIORITY_QUEUE with make_node(initial_state)
    
    while frontier is not empty:
        node ← POP(frontier)          // usually by lower bound
        
        // Pruning: if lower bound ≥ current best, skip
        if lower_bound(node) >= U:
            continue                  // PRUNE this branch
        
        if GOAL_TEST(node.state):
            if cost(node) < U:
                best_solution ← node
                U ← cost(node)        // update upper bound
            continue
        
        for each child in EXPAND(node):
            if lower_bound(child) < U:
                INSERT(frontier, child)
    
    return best_solution
```

#### Variants:
- **B&B with DFS:** Uses a stack; memory efficient but may find bad solutions first
- **B&B with BFS:** Uses a queue; finds optimal sooner but more memory
- **B&B with Best-First (LC Branch & Bound):** Uses lower bound as priority → this is essentially **UCS** / **A\*** with pruning

#### Analysis:
| Metric | Value | Notes |
|--------|-------|-------|
| Complete? | **Yes** (finite space) | Explores all non-pruned branches |
| Time | **O(b^d)** worst case | Pruning reduces this significantly in practice |
| Space | Depends on traversal order | DFS variant: O(bd) |
| Optimal? | **Yes** | Never prunes the optimal solution (if lower bound is tight) |

**Key insight for exam:** Branch and Bound is the *general framework*. UCS is B&B without a heuristic lower bound. A* is B&B with an admissible heuristic as the lower bound. They are all the same family!

**Relationship diagram:**
```
Branch & Bound (framework)
    ├── No heuristic (g only) → Uniform Cost Search (UCS)
    ├── Admissible heuristic (f = g + h) → A*
    └── Iterative cost threshold → IDA*
```

---

### ⭐ A* SEARCH — MOST IMPORTANT (Asked Every Exam)

**Core Idea:** Avoid expanding paths already expensive.
- **g(n)** = actual cost from start to n (like UCS)
- **h(n)** = heuristic estimated cost from n to goal (like Best-First)
- **f(n) = g(n) + h(n)** = estimated total cost of path through n

**Data Structure:** Priority Queue ordered by f(n) (lowest first)

#### Pseudocode (Prof's General Tree-Search Paradigm):
```
function A_STAR(root_node, h):
    fringe ← {make_node(initial_state, g=0, f=h(initial_state))}
    explored ← empty set

    while not empty(fringe):
        node ← remove_first(fringe)      // remove lowest f(n)
        state ← state(node)

        if goal_test(state): return solution(node)
        explored.add(state)

        for each child in successors(node):
            child.g ← node.g + step_cost(node, child)
            child.f ← child.g + h(child.state)
            if child.state not in explored:
                INSERT(fringe, child)   // re-insert if better path found

    return failure
```

---

#### FULL A* TRACE — Romania Problem (Prof's Slides — Learn This Cold!)

Using **h_SLD** = straight-line distance to Bucharest (from table: Arad=366, Sibiu=253, etc.)

```
Initial: fringe = [Arad: f=0+366=366]

Step 1 — Expand Arad (f=366):
  Sibiu:      g=140, h=253, f=393  ← LOWEST → expand next
  Timisoara:  g=118, h=329, f=447
  Zerind:     g=75,  h=374, f=449

Step 2 — Expand Sibiu (f=393):
  Arad:           g=280, h=366, f=646
  Fagaras:        g=239, h=176, f=415
  Oradea:         g=291, h=380, f=671
  Rimnicu Vilcea: g=220, h=193, f=413  ← LOWEST in whole fringe

Step 3 — Expand Rimnicu Vilcea (f=413):
  Craiova: g=366, h=160, f=526
  Pitesti: g=317, h=100, f=417
  Sibiu:   g=300, h=253, f=553
  Fringe lowest: Fagaras at 415

Step 4 — Expand Fagaras (f=415):
  Sibiu:     g=338, h=253, f=591
  Bucharest: g=450, h=0,   f=450  ← on fringe, NOT selected yet! (450 > 417)
  Fringe lowest: Pitesti at 417

Step 5 — Expand Pitesti (f=417):
  Bucharest: g=418, h=0, f=418  ← cheaper than existing Bucharest (450)!
  Craiova:   g=455, h=160, f=615
  Rimnicu:   g=414, h=193, f=607
  Replace old Bucharest(450) with Bucharest(418). Now lowest in fringe.

Step 6 — Expand Bucharest (f=418) → GOAL! ✓

OPTIMAL PATH: Arad → Sibiu → Rimnicu Vilcea → Pitesti → Bucharest = 418 km
```

**KEY LESSON:** A* first found Bucharest via Fagaras (f=450) but didn't stop — it found a cheaper path via Pitesti (f=418). This proves goal-testing happens at **DEQUEUE**, not generation.

**BFS/Greedy would have stopped early and returned a suboptimal answer.**

---

#### HEURISTIC THEORY — Complete Coverage

##### 1. Admissibility
> h(n) is **admissible** if: **h(n) ≤ h*(n)** for every node n

- h*(n) = true optimal cost from n to goal
- Admissible = never overestimates = **optimistic**
- **Theorem:** Admissible h + TREE-SEARCH → A* is **optimal**

##### 2. Consistency (Monotonicity)
> h(n) is **consistent** if: **h(n) ≤ c(n, a, n') + h(n')** for every successor n'

- This is the heuristic triangle inequality
- **Theorem:** Consistent h + GRAPH-SEARCH → A* is **optimal**
- Every consistent heuristic is also admissible (but not vice versa)

---

#### PROOF OF A* OPTIMALITY (Prof's Full Proof — Write This in Exams)

**Setup:** h is admissible. G₂ = suboptimal goal on frontier. n = unexpanded node on optimal path to G.

**Step 1 — Analyse G₂:**
```
f(G₂) = g(G₂)          [since h(G₂) = 0, it's a goal]
g(G₂) > g(G)            [G₂ is suboptimal by assumption]
```

**Step 2 — Analyse G:**
```
f(G) = g(G)              [since h(G) = 0]
∴ f(G₂) > f(G)           [substitution]
```

**Step 3 — Analyse n:**
```
h(n) ≤ h*(n)             [admissibility]
f(n) = g(n) + h(n) ≤ g(n) + h*(n) = f(G)
∴ f(n) ≤ f(G)            [n on optimal path, so g(n)+h*(n) = f(G)]
```

**Conclusion:**
```
f(G₂) > f(G) ≥ f(n)
∴ A* always expands n before G₂ → never returns suboptimal solution ∎
```

---

#### ADMISSIBLE HEURISTICS — 8-PUZZLE (Prof's Examples)

**h₁(n)** = number of misplaced tiles
**h₂(n)** = total Manhattan distance = Σ (|row_curr - row_goal| + |col_curr - col_goal|)

For start [7,2,4 / 5,_,6 / 8,3,1] → goal [_,1,2 / 3,4,5 / 6,7,8]:
- **h₁(S) = 8** (every numbered tile is out of place)
- **h₂(S) = 3+1+2+2+2+3+3+2 = 18** (per-tile grid distances)

Both ≤ true optimal cost → both admissible ✓

---

#### DOMINANCE

> h₂ **dominates** h₁ if h₂(n) ≥ h₁(n) for all n (both admissible)

A dominant heuristic is strictly better — same optimality guarantee, fewer nodes expanded.

**From Prof's data:**

| Algorithm | d=12 | d=24 |
|-----------|------|------|
| IDS | 3,644,035 | way too many |
| A*(h₁ misplaced) | 227 | 39,135 |
| **A*(h₂ Manhattan)** | **73** | **1,641** |

Manhattan dominates misplaced tiles → 3× fewer expansions at d=12, 24× fewer at d=24!

---

#### RELAXED PROBLEMS — Inventing Admissible Heuristics

> Remove restrictions from the problem → solve relaxed version → that cost is an admissible heuristic.

**8-puzzle derivations:**
- Relax "must slide to empty adjacent" → tile moves to ANY adjacent square → **h₂ Manhattan**
- Relax "must be adjacent too" → tile teleports anywhere → **h₁ Misplaced tiles**

Since relaxed optimal ≤ original optimal → automatically admissible!

---

#### WEIGHTED A* (Speed/Quality Tradeoff)

```
f(n) = g(n) + w·h(n)    w > 1 (typically w = 5)
```
- w=1 → standard A* (optimal)
- w>1 → faster but suboptimal; cost ≤ w × optimal (when h admissible)
- Useful when approximate answers are acceptable

---

#### DFS Branch and Bound vs IDA* (Prof's Direct Comparison)

| Property | DFS B&B | IDA* |
|----------|---------|------|
| Optimal? | Yes | Yes |
| Systematic? | **Yes** — never expands same node twice | No — re-expands nodes |
| Expands suboptimal nodes? | Yes | No (never expands f > optimal) |
| Works on infinite trees? | Risky | Yes |
| Memory | O(bd) | O(bd) |

**When to prefer:** IDA* when finding suboptimal solution first is OK; DFS B&B when systematicity matters.

---

#### A* Properties Summary:
| Metric | Value | Condition |
|--------|-------|-----------|
| Complete? | Yes | Unless ∞ nodes with f ≤ f(G) |
| Time | O(b^d) worst case | Exponential |
| Space | **O(b^d) — keeps ALL nodes** | Main weakness → use IDA* instead |
| Optimal? | Yes (tree: admissible h) | Yes (graph: consistent h) |

---

### 🌿 RECURSIVE BEST-FIRST SEARCH (RBFS)

**Core Idea:** Like A* but uses O(bd) linear memory by simulating recursion. It tracks the **best alternative f-value** available at each level and backtracks when the current path's f exceeds that alternative.

**Motivation:** A* runs out of memory (O(b^d)). IDA* wastes time re-generating nodes. RBFS is a middle ground — linear memory, less re-generation than IDA*.

#### Pseudocode:
```
function RBFS(problem, node, f_limit):
    if GOAL_TEST(node.state): return SOLUTION(node)
    
    successors ← EXPAND(node)
    if successors is empty: return FAILURE, ∞
    
    // Inherit parent's f if it's higher (monotone update)
    for each s in successors:
        s.f ← max(s.g + h(s.state), node.f)
    
    loop:
        best ← successor with lowest f-value
        if best.f > f_limit: return FAILURE, best.f  // backtrack!
        
        alternative ← second-lowest f among successors
        
        result, best.f ← RBFS(problem, best, min(f_limit, alternative))
        
        if result ≠ FAILURE: return result

// Initial call:
function RBFS_SEARCH(problem):
    return RBFS(problem, make_node(initial_state), ∞)
```

#### How it works:
- Explores best path, tracking what the **next-best alternative** costs
- When current best path's f exceeds f_limit → **backtrack** and update f for that subtree
- Re-explores paths only when needed (less waste than IDA*, more than A*)

#### Analysis:
| Metric | Value | Notes |
|--------|-------|-------|
| Complete? | Yes (admissible h) | |
| Time | Potentially exponential | Better than IDA* in practice |
| Space | **O(bd) — linear!** | Big advantage over A* |
| Optimal? | Yes (admissible h) | |

**Problem with RBFS:** Excessive node re-generation still occurs when f-values are similar across branches. Each backtrack may re-explore a large subtree.

---

### 🧠 SMA* — SIMPLIFIED MEMORY-BOUNDED A*

**Core Idea:** A* but with a **fixed memory limit**. When memory is full, drop the worst node (highest f on the frontier) and store its f-value in its parent so it can be regenerated if needed.

**Motivation:** A* uses too much memory. IDA*/RBFS use too little (re-expand too often). SMA* uses ALL available memory efficiently.

#### How it works:
1. Run A* normally
2. When memory is full (all slots used):
   - **Drop** the frontier node with highest f (worst node)
   - **Record** its f-value in its parent node
   - If the parent later becomes the best node and all its remembered children need expansion, regenerate the dropped child
3. Continue until goal found or memory provably insufficient

#### Pseudocode (sketch):
```
function SMA_STAR(problem, h, memory_limit):
    fringe ← {make_node(initial_state)}
    
    while true:
        if fringe is empty: return FAILURE
        node ← best_leaf(fringe)             // lowest f
        
        if GOAL_TEST(node.state): return SOLUTION(node)
        
        successor ← next_ungenerated_child(node)
        if no successor:
            node.f ← ∞                       // fully expanded, mark forgotten
            update parent's f accordingly
        else:
            s.f ← max(node.f, s.g + h(s.state))
            if memory full:
                forgotten ← worst_leaf(fringe)   // highest f
                forgotten.parent.forgotten_f ← forgotten.f  // save for later
                remove forgotten from fringe
            INSERT(fringe, s)
    
```

#### Analysis:
| Metric | Value | Notes |
|--------|-------|-------|
| Complete? | **Yes** (if solution reachable in available memory) | |
| Time | Better than IDA* when memory sufficient | |
| Space | **Uses ALL available memory** — configurable | The key feature |
| Optimal? | Yes (admissible h, sufficient memory) | |

**SMA* vs IDA* vs A*:**

| | A* | IDA* | RBFS | SMA* |
|---|---|---|---|---|
| Memory | O(b^d) all | O(bd) linear | O(bd) linear | Configurable |
| Re-expansion | None | Heavy | Moderate | Minimal |
| Best when | Small space | Large space, simple | Medium space | Limited but known memory |

---

### ↔️ BIDIRECTIONAL SEARCH

**Core Idea:** Run two simultaneous searches — one **forward** from start, one **backward** from goal. Stop when the two frontiers **meet in the middle**.

**Motivation:** If BFS from start alone takes O(b^d), two half-searches each take O(b^(d/2)). Total = 2·O(b^(d/2)) << O(b^d).

**Example:** b=10, d=6:
- Unidirectional BFS: 10^6 = 1,000,000 nodes
- Bidirectional BFS: 2 × 10^3 = 2,000 nodes → **500× reduction!**

#### Pseudocode:
```
function BIDIRECTIONAL_SEARCH(problem):
    forward_frontier  ← {make_node(initial_state)}
    backward_frontier ← {make_node(goal_state)}
    forward_explored  ← empty set
    backward_explored ← empty set
    
    while true:
        // Expand smaller frontier first (for efficiency)
        if forward_frontier is empty OR backward_frontier is empty:
            return FAILURE
        
        // Forward step
        node ← remove_first(forward_frontier)
        forward_explored.add(node.state)
        for each child in EXPAND(node):
            if child.state ∈ backward_explored:
                return JOIN_PATHS(child, corresponding_backward_node)
            INSERT(forward_frontier, child)
        
        // Backward step
        node ← remove_first(backward_frontier)
        backward_explored.add(node.state)
        for each pred in PREDECESSORS(node):
            if pred.state ∈ forward_explored:
                return JOIN_PATHS(corresponding_forward_node, pred)
            INSERT(backward_frontier, pred)
```

#### Analysis:
| Metric | Value | Notes |
|--------|-------|-------|
| Complete? | **Yes** (BFS variant) | |
| Time | **O(b^(d/2))** | Exponential savings! |
| Space | **O(b^(d/2))** | Must store both frontiers |
| Optimal? | **Yes** (uniform cost) | If both directions use UCS |

#### Challenges with Bidirectional Search:
1. **Goal must be explicit** — can't use with general goal tests (e.g., "is this checkmate?")
2. **Predecessor function required** — must be able to search backwards (not always easy)
3. **Meeting condition** — just because frontiers intersect doesn't mean optimal path found; need careful handling
4. **Cost-weighted graphs** — harder to ensure optimality; use bidirectional UCS/A*

#### Bidirectional A*:
Run A* in both directions with h(n) pointing toward the respective goals. Optimal but meeting-point detection is tricky — current best solution = min over all meeting nodes of g_f(n) + g_b(n).

---

### 🌟 IDA* (Iterative Deepening A*)

**Core Idea:** Apply the IDS idea to A*. Instead of iterating on depth, iterate on the **f-cost threshold**. At each iteration, explore all nodes with f(n) ≤ threshold. If goal not found, increase threshold to the minimum f-value that exceeded the previous threshold.

**Motivation:** A* uses O(b^d) space (exponential). IDA* uses only O(bd) space — same as DFS — while remaining optimal.

#### Pseudocode:
```
function IDA_STAR(problem, h):
    root ← make_node(initial_state)
    threshold ← h(root.state)           // initial f-cost limit
    
    while true:
        result ← IDA_SEARCH(root, 0, threshold, problem, h)
        
        if result == FOUND: return SOLUTION
        if result == ∞: return FAILURE   // no solution exists
        threshold ← result               // next threshold = min exceeded f
    

function IDA_SEARCH(node, g, threshold, problem, h):
    f ← g + h(node.state)
    
    if f > threshold: return f           // exceeded: return f for next iteration
    if GOAL_TEST(node.state): return FOUND
    
    min_exceeded ← ∞
    
    for each action in ACTIONS(node.state):
        child ← CHILD_NODE(problem, node, action)
        child_g ← g + step_cost(node, action, child)
        result ← IDA_SEARCH(child, child_g, threshold, problem, h)
        
        if result == FOUND: return FOUND
        if result < min_exceeded: min_exceeded ← result
    
    return min_exceeded
```

#### How threshold evolves:
```
Iteration 1: threshold = h(start)        e.g., threshold = 10
Iteration 2: threshold = min f exceeded  e.g., threshold = 12
Iteration 3: threshold = next min        e.g., threshold = 15
...until goal found
```

#### Analysis:
| Metric | Value | Notes |
|--------|-------|-------|
| Complete? | **Yes** | Finds solution if one exists |
| Time | **O(b^d)** | Same as A* asymptotically |
| Space | **O(b·d)** | Linear! This is the big win over A* |
| Optimal? | **Yes** (admissible h) | Inherits A*'s optimality guarantee |

**IDA* vs A* comparison (exam favourite):**

| Property | A* | IDA* |
|----------|-----|------|
| Space | O(b^d) — exponential | **O(bd) — linear** |
| Time | O(b^d) | O(b^d) — slightly more due to repeated work |
| Optimal? | Yes | Yes |
| Repeated nodes | Handled by explored set | **Re-generates** nodes (no memory of visited) |
| Best for | Small/medium spaces | **Large spaces, memory limited** |

**The catch:** IDA* can re-expand nodes many times (no explored set). In graphs with many paths to the same state, this is very expensive. A* handles this via its explored set.

---

### 🔶 BEAM SEARCH

**Core Idea:** A memory-bounded variant of BFS. At each level, keep only the **k best nodes** (the "beam width").

#### Pseudocode:
```
function BEAM_SEARCH(problem, k, h):
    beam ← {make_node(initial_state)}
    
    while beam is not empty:
        successors ← all children of nodes in beam
        if any successor is GOAL: return SOLUTION
        
        // Keep only top-k by heuristic h
        beam ← TOP_K(successors, k, key=h)
    
    return FAILURE
```

#### Analysis:
| Metric | Value | Notes |
|--------|-------|-------|
| Complete? | **No** | May discard the path to the goal |
| Time | **O(k·b·d)** | Much faster than BFS |
| Space | **O(k·b)** | Only k nodes per level |
| Optimal? | **No** | Approximate |

**Tradeoff:** Larger k → more complete, more memory. k=∞ → BFS. k=1 → Hill Climbing.

**Key insight:** Beam Search is a *heuristic*, not guaranteed to find the best solution. Used in NLP (machine translation), speech recognition.

---

## PART 3: MASTER COMPARISON TABLE

| Algorithm | Complete? | Time | Space | Optimal? | Data Structure | Key Formula |
|-----------|-----------|------|-------|----------|----------------|-------------|
| **BFS** | Yes (finite b) | O(b^d) | O(b^d) | Yes (uniform cost) | Queue | depth-level order |
| **DFS** | No | O(b^m) | O(b·m) | No | Stack | deepest first |
| **DLS** | Yes (d ≤ ℓ) | O(b^ℓ) | O(b·ℓ) | No | Stack | DFS + limit ℓ |
| **IDS** | Yes | O(b^d) | O(b·d) | Yes (uniform cost) | Stack (iterative) | DLS × iterations |
| **UCS** | Yes (ε > 0) | O(b^(C*/ε)) | O(b^(C*/ε)) | Yes | Priority Queue | sort by g(n) |
| **Best-First** | No | O(b^m) | O(b^m) | No | Priority Queue | sort by h(n) |
| **B&B** | Yes (finite) | O(b^d) pruned | O(b·d) DFS | Yes | Priority Queue | prune if L(n) ≥ U |
| **A\*** | Yes (admissible h) | O(b^d) | O(b^d) | Yes | Priority Queue | f(n)=g(n)+h(n) |
| **IDA\*** | Yes | O(b^d) | O(b·d) | Yes (admissible h) | Implicit stack | f-threshold iterations |
| **Beam** | No | O(k·b·d) | O(k·b) | No | k-best list | keep top-k by h |

---

## PART 4: REPEATED STATES — CRITICAL CONCEPT

**Why it matters:** Failure to detect repeated states can turn a LINEAR problem into an EXPONENTIAL one.

**Three strategies (increasing cost):**
1. **Don't revisit the parent** — cheap, catches immediate cycles
2. **Don't revisit any ancestor** — catches loops on current path
3. **Don't revisit any previously explored state** — complete, expensive (requires hash set)

Graph search (with explored set) >> Tree search (without) for avoiding repeated states.

---

## PART 5: PSEUDOCODE QUESTIONS — HOW PROFS FRAME THEM

### Pattern 1: "Write pseudocode for X"
Always include: initialization, loop condition, expansion step, goal check placement, return values.

### Pattern 2: "Trace algorithm X on this graph"
Write the frontier/queue state after each step:
```
Step 0: Frontier = [A]
Step 1: Expand A → Frontier = [B, C]  (BFS: queue; DFS: stack)
Step 2: Expand B → Frontier = [C, D, E]
...
```

### Pattern 3: "Compare X and Y on these criteria"
Use the table format above. Always mention WHY each property holds.

---

## PART 6: 5 PREDICTED EXAM QUESTIONS

### Q1 — Conceptual + Numerical (High Probability)
**"Given a search tree with branching factor b=3 and solution depth d=4, calculate: (a) the number of nodes generated by BFS, (b) nodes by IDS, and (c) which is preferred and why."**

**Answer:**
- BFS nodes: b + b² + b³ + b⁴ = 3 + 9 + 27 + 81 = **120 nodes** (plus root = 121)
- IDS nodes: d·b + (d-1)·b² + (d-2)·b³ + (d-3)·b⁴  
  = 4(3) + 3(9) + 2(27) + 1(81) = 12 + 27 + 54 + 81 = **174 nodes**
- IDS generates ~45% more nodes but uses O(bd) = O(12) space vs BFS's O(81).
- **IDS preferred** when memory is constrained.

---

### Q2 — Algorithm Trace (High Probability)
**"Apply BFS and DFS to the Romania map problem starting from Arad. Show the frontier queue/stack at each step. Which finds the goal first?"**

**Method:** Draw the expansion order. BFS finds shortest path in hops; DFS finds first path via stack ordering (may not be optimal).

---

### Q3 — Problem Formulation (High Probability)
**"Formulate the N-Queens problem as a search problem. Define: states, initial state, actions, goal test, and path cost."**

**Answer:**
- States: Any arrangement of 0 to N queens on the board
- Initial state: Empty board
- Actions: Add a queen to any non-attacked square
- Goal test: N queens placed, none attacking each other
- Path cost: 0 (we only care about the final state, not how we got there)

---

### Q4 — Comparative Analysis (Medium Probability)
**"Why is DFS not complete? Under what conditions does it become complete? Why is BFS impractical for large problems despite being complete?"**

**Answer:**
- DFS not complete: Can follow infinite branches without ever finding goal or returning
- DFS becomes complete: In finite state spaces with loop detection (graph search)
- BFS impractical: Space complexity O(b^d) — must store ALL generated nodes. With b=10, d=10: 10 billion nodes in memory

---

### Q5 — Design/Coding Question (Medium-High Probability)
**"Implement Iterative Deepening Search in Python. Show that it combines BFS completeness with DFS space efficiency."**

```python
def depth_limited_search(problem, node, limit):
    if problem.is_goal(node.state):
        return node
    if limit == 0:
        return 'CUTOFF'
    
    cutoff_occurred = False
    for child in problem.expand(node):
        result = depth_limited_search(problem, child, limit - 1)
        if result == 'CUTOFF':
            cutoff_occurred = True
        elif result != 'FAILURE':
            return result
    
    return 'CUTOFF' if cutoff_occurred else 'FAILURE'

def iterative_deepening_search(problem):
    depth = 0
    while True:
        root = Node(problem.initial_state)
        result = depth_limited_search(problem, root, depth)
        if result != 'CUTOFF':
            return result
        depth += 1

class Node:
    def __init__(self, state, parent=None, action=None, g=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g  # path cost
        self.depth = 0 if parent is None else parent.depth + 1
```

---

## PART 7: MATHEMATICAL NOTATION & NOMENCLATURE

Use these in answers to sound rigorous:

| Symbol | Meaning |
|--------|---------|
| b | Branching factor (max successors per node) |
| d | Depth of shallowest goal node |
| m | Maximum depth of search tree (m → ∞ for infinite spaces) |
| C* | Optimal path cost |
| ε | Minimum edge cost (ε > 0 required for UCS completeness) |
| g(n) | Actual cost from start to node n |
| h(n) | Heuristic: estimated cost from n to goal |
| f(n) | A*: f(n) = g(n) + h(n), estimated total cost |
| O(b^d) | Exponential in depth — the curse of search |
| ℓ | Depth limit in DLS |

**Phrases that score marks:**
- "The time complexity is exponential in the depth of the solution, O(b^d), making it intractable for large search spaces."
- "UCS guarantees optimality provided ε > 0, i.e., no zero-cost cycles exist."
- "IDS asymptotically matches BFS in time — O(b^d) — while maintaining DFS's linear space complexity O(bd)."
- "Repeated state detection transforms the search from tree search to graph search, preventing exponential blowup."

---

## PART 8: ABOUT "MANTIS" ALGORITHM

⚠️ "Mantis" is **not a standard search algorithm name** in the classical AI curriculum (Russell & Norvig). Your professor may be referring to:

1. **A custom/course-specific algorithm** — Check your lecture notes for its definition
2. **A metaheuristic** (like Ant Colony, Simulated Annealing, Genetic Algorithms) — possible if your prof covers local search
3. **A mnemonic** for remembering algorithm properties

**On Metaheuristic Derivation Questions:**
If your prof asks about metaheuristics, key things to know:
- They are *approximate*, not guaranteed optimal
- Examples: Simulated Annealing, Genetic Algorithms, Ant Colony Optimization
- They escape local optima (unlike pure hill climbing)
- Derivation usually involves: defining energy/fitness function, neighborhood structure, acceptance probability

**If you mean Minimax** (for games): f(n) = minimax value, backed up from leaves using MAX and MIN operators.

**Recommendation:** Ask your professor or classmate to clarify "Mantis" — it's important to get this right before the exam!

---

## PART 9: QUICK REVISION CHEAT SHEET

```
BFS:  Queue    | O(b^d) time    | O(b^d) space   | Complete✓ | Optimal✓(uniform)
DFS:  Stack    | O(b^m) time    | O(bm) space    | Complete✗ | Optimal✗
DLS:  Stack+ℓ  | O(b^ℓ) time    | O(bℓ) space    | Complete? | Optimal✗
IDS:  DLS×     | O(b^d) time    | O(bd) space    | Complete✓ | Optimal✓(uniform)
UCS:  PQ(g)    | O(b^C*/ε)      | O(b^C*/ε)      | Complete✓ | Optimal✓
BestF:PQ(h)    | O(b^m) time    | O(b^m) space   | Complete✗ | Optimal✗
B&B:  PQ+prune | O(b^d) pruned  | O(bd) DFS-mode | Complete✓ | Optimal✓
A*:   PQ(f=g+h)| O(b^d) time    | O(b^d) space   | Complete✓ | Optimal✓(admissible h)
IDA*: DLS(f-thresh)| O(b^d)     | O(bd) LINEAR!  | Complete✓ | Optimal✓(admissible h)
Beam: k-best   | O(kbd) time    | O(kb) space    | Complete✗ | Optimal✗
```

### Goal Check Placement (Exam Trap):
- **BFS:** Check when GENERATING (safe because shallowest = optimal)
- **UCS:** Check when DEQUEUING (must be cheapest before declaring goal)

### When to use what:
- Memory unlimited, uniform cost → **BFS**
- Memory limited → **IDS**
- Varying costs, need optimal → **UCS**
- Have a good heuristic, need speed → **Best-First / A***
- Massive space, approximate ok → **Beam Search**

---

*Good luck tomorrow! You've got this 🎯*
