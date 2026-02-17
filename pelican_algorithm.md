## 3rd Feb 2026 Lab
__Pelican Optimization Algorithm__

---
#### Google Classroom Form (topic submission)
- __Pelican Optimization Algorithm (POA)__ is a nature-inspired, population-based metaheuristic introduced in 2022, modeled on the cooperative hunting behavior of pelicans. 
- It operates in two main phases: exploration, where pelicans randomly search the environment to __locate prey__, and exploitation, where they perform a __rapid diving movement toward the best-identified solution__. This helps balance global search and local refinement while maintaining fast convergence. POA is especially effective for continuous optimization problems where quick convergence is desirable. 
- However, like many swarm algorithms and also since it has __fast convergence__, it may suffer from __premature convergence in highly multimodal landscapes__, which we will try to optimize. 
- In real world applications, POA is used in engineering design to choose the best values for dimensions or materials so a system performs well. In ML, it helps in __feature selection__, meaning it picks the most important input features to improve __accuracy__ and __reduce complexity__. In power systems, POA is applied to decide how much electricity different power plants should generate to reduce cost while meeting demand, and so on.

---
#### Pareto Optimal


---
#### learnings
- Endoscopic image analysis plays a crucial role in diagnosis Gastrointestinal Diseases (GD) by allowing visualization of the inner tissues of gastrointestinal tract. 
- However, quality of gastrointestinal images is often suboptimal and GD classification are complex and require multiple parameters to training, affecting their accuracy. 
- In this research, proposed Levy Flight Pelican Optimization Algorithm (LFPOA) and Ensemble Machine Learning (ML) method for classification using methods like Multi-Support Vector Machine (MSVM), K-Nearest Neighbour (KNN), Random Forest (RF), Decision Tree (DT) and Naïve Bayes (NB). 
- The LFPOA reduces dimensionality, balances exploration and exploitation, leads to a global optimum, helping to select relevant features. 
- The EL combines five machine learning techniques to handle multiple classifiers and produce a single output, often leading to higher accuracy.
- 

---
## Base Paper
- To validate its efficiency, the POA was tested against twenty-three benchmark objective functions and four distinct real-world engineering design challenges.
- Comparative analysis demonstrates that the POA maintains a superior balance between global search and local convergence when compared to eight other established algorithms.
- The algorithm models the hunting strategy of pelicans, specifically:
    1. Locating prey: Identifying the target area.
    2. Attacking: Diving toward the prey from a height.
    3. Foraging: Spreading wings on the water surface to guide fish into their throat pouch.
#### 1. __Phase 1__
- Moving Towards Prey (__Exploration__) In this phase, the pelican (search agent) identifies the location of the prey (randomly generated in the search space) and moves toward it. This facilitates the scanning of different areas of the search space.
    - _Mechanism_: If the __prey's location__ is __better__ (has a better objective function value) than the __pelican's current position__, the pelican __moves toward__ the prey. If not, it __moves away__ to search elsewhere.
    - _Randomization_: A parameter I (randomly 1 or 2) is used to __alter the displacement magnitude__, enhancing __exploration__ power.
#### 2. __Phase 2__
- Winging on the Water Surface (__Exploitation__) After reaching the prey's area, the pelican spreads its wings on the water surface to catch fish. This models the local search in the neighborhood of the pelican's current position to converge to a better solution.
    - Mechanism: The algorithm examines points in the neighborhood of the pelican. The __radius of this neighborhood decreases linearly__ as the iteration count (t) increases, allowing for __finer local searching__ as the algorithm progresses.
    - Equation: The position is updated based on a neighborhood radius factor ```R⋅(1−t/T)```, where T is the maximum number of iterations. Sensitivity analysis suggests __R=0.2__ is often optimal.
#### Improvements from Previous Methods
- Superior Convergence: In testing on unimodal functions, POA demonstrated high exploitation ability, effectively converging to the global optimum compared to competitors
- 