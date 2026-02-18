# The Mantis Search Algorithm (MSA)

### 1. Algorithm Inspiration
The __Mantis Search Algorithm__ (MSA) is inspired by the unique **predatory tactics** and lifecycle of the praying mantis (Mantis religiosa). Unlike standard swarm optimizers, MSA models specific biological innovations:
    - __Camouflage__: Mantises mimic objects like ants or wood (crypsis/masquerade) to minimize the predator-prey gap and remain undetected.
    - __UV Ray Absorption__: Orchid mantises (Hymenopus coronatus) absorb UV light to appear flower-like to UV-sensitive pollinators, creating a high-contrast "allure" against foliage.
    - __Aggressive Chemical Mimicry__: Juveniles secrete pheromones (3HOA and 10HDA) to actively lure specific prey like honeybees.
    - __Advanced Perception__: Mantises use "head peering" (motion parallax) and stereopsis (3D vision) to accurately estimate distances for ballistic strikes.
    - __Asymmetric Vision__: Inspired by the Mantis Shrimp variant (MShOA), specialized eyes (red/purple functionality) allow for "multi-perspective perception" of polarized light signals to decide between attack or defense.

### 2. Mathematical Model

#### 2.1. Initialization
Agents are distributed using a uniform distribution or, in improved versions, a Sobol sequence for better spatial coverage:
![Sobol Sequence](Sobol_Sequence.png)

#### 2.2. Movement Update Rule (Pursuit vs. Ambush)
The movement mimics the transition from a "pursuer" (global search) to a "spearer" (local ambush):
    - **Pursuer (Levy Flight)**: Uses small steps with occasional large jumps to escape local optima.
    - **Spearer (Fixed Ambush)**: Adjusts position based on the global best ($x_{best}$) and the head-position parameter $\alpha$.