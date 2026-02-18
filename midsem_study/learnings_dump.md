## UNIT II: Problem Solving & Search

#### 1. Defining the Search Variables
When analyzing search algorithms, we use specific variables to measure how much time and memory (space) an algorithm will consume:
    - $b$ (Branching Factor): The maximum number of successors (children) any node can have. If $b=2$, every node splits into two.
    - $d$ (Depth of Solution): The depth of the shallowest goal node (how many steps from the start to the first solution).
    - $m$ (Maximum Depth): The maximum length of any path in the state space. This can be infinite ($\infty$) if there are loops.
    - $C^*$ (Cost of Optimal Solution): The total path cost of the best possible solution.
    - $\epsilon$ (Minimum Step Cost): The smallest possible cost of a single action. This is used in Uniform Cost Search to ensure the algorithm doesn't get stuck in an infinite loop of tiny costs.

#### 2. Uninformed Search Strategies (Blind Search)
These algorithms have no information about how far a node is from the goal; they only know how to generate successors.

###### A. Breadth-First Search (BFS)
    - Mechanism: Expands the shallowest unexpanded node first using a FIFO (First-In-First-Out) queue. It explores the tree layer by layer.
    - Why $O(b^d)$? In the worst case, you must generate every node up to depth $d$. The number of nodes is $1 + b + b^2 + ... + b^d$.
    - The Problem: Space complexity. It keeps every generated node in memory. For a branching factor of 10 and depth 10, this could require terabytes of RAM.

###### B. Depth-First Search (DFS)
    - Mechanism: Expands the deepest unexpanded node first using a LIFO (Last-In-First-Out) stack.
    - Why $O(bm)$? DFS only needs to store the path from the root to the current node, plus the siblings of nodes on that path. This is much more memory-efficient than BFS.
    - The Risk: It is not complete if the tree is infinite or has loops, as it might follow a "dead end" forever.

###### Uniform Cost Search (UCS)
    - Mechanism: Instead of depth, it expands the node $n$ with the lowest path cost $g(n)$ using a Priority Queue.
    - Optimality: Guaranteed to find the cheapest path because it always expands the "cheapest" node currently known.

###### Iterative Deepening Search (IDS)
    - Mechanism: It runs a DFS with a depth limit of 0, then 1, then 2, and so on.
    - The "Magic": It feels wasteful because it re-generates top-level nodes multiple times. However, since the number of nodes at depth $d$ is so much larger than all previous levels combined, the overhead is actually quite small (only about 11% for $b=10$).
    - Conclusion: It is the "Best of Both Worlds"—it has the linear space of DFS ($O(bd)$) and the optimality/completeness of BFS.

#### 3. Informed Search (A Search)*
Informed search uses "hints" called heuristics to find the goal faster.

###### The Evaluation Function: $f(n) = g(n) + h(n)
    - $$g(n)$: The actual cost reached so far from the start node to node $n$.
    - $h(n)$: The estimated cost from node $n$ to the goal. This is the "heuristic."
    - $f(n)$: The estimated total path cost through node $n$.

###### Admissibility and Consistency
For A* to be optimal, the heuristic $h(n)$ must be "well-behaved":
    - Admissible: $h(n)$ must never overestimate the cost. It must be optimistic. If the real cost to the goal is 10, $h(n)$ can be 7 or 10, but never 11.
    - Consistent (Monotonic): The estimate $h(n)$ should not decrease more than the actual cost of moving to a neighbor. Formally: $h(n) \le cost(n, a, n') + h(n')$.

###### Dominance: Comparing Heuristics
If you have two admissible heuristics, $h_1$ and $h_2$, and $h_2(n) \ge h_1(n)$ for all nodes, then $h_2$ dominates $h_1$.
    - Why it matters: A dominating heuristic is "tighter" (closer to the real cost). It provides better guidance and forces A* to expand fewer nodes, making the search more efficient.

