# Democratic Hammurabi: AI Benchmark & Environment

This repository contains the **Democratic Hammurabi** environment—a highly difficult, multi-objective, text-based resource management game designed to test the limits of AI algorithms.

## History & Motivation
What started as a simple experiment to see if a Genetic Algorithm (GA) could beat the classic 1968 text game *Hammurabi* evolved into a complex research benchmark. By introducing **Democracy** (three opposing political factions and quadrennial elections), the environment became a zero-sum trap.

We tested multiple algorithms against it:
1. **Genetic Algorithms (MLP & RNN):** Failed to solve temporal credit assignment.
2. **Hebbian Plasticity (Evolved RNN):** Showed dynamic learning but fell into local minima (reward hacking by selling land).
3. **Deep Reinforcement Learning (PPO):** Mastered short-term resource management (zero starvation) but failed long-horizon planning, dying in Year 6.
4. **Agentic SLMs (Gemma 4):** Demonstrated superior algebraic reasoning but struggled with multi-turn political foresight.

The environment is now being ported as an official **LLM Reasoning Benchmark** (e.g., via `kaggle-benchmarks`), testing if frontier models have the long-horizon context window to survive a 12-year democratic term.

## The Environment (`hammurabi_env.py`)
The environment is built on standard reinforcement learning principles but is fiercely unforgiving:
- **State Space (9 features):** Year, Population, Grain, Land, Land Price, Farmer Approval, Worker Approval, Elite Approval, Years to Election.
- **Action Space:** Continuous fractions for buying/selling land, feeding, and planting.
- **The Trap:** The reward function utilizes `min(factions)`, forcing the agent to balance all three opposing groups rather than pandering to a majority.
- **The Minority Coalition:** The election threshold requires an average approval of **45.0%** to stay in power. This softens the mathematical impossibility of absolute dictatorships while preserving the vicious multi-objective tension of democracy.

## How to Run the Baselines

### 1. Genetic Algorithms & CMA-ES
No external libraries (other than `numpy` and `cmaes`) are required.
```bash
python train.py         # Trains the MLP GA
python train_cmaes.py   # Trains the CMA-ES optimizer
python benchmark.py     # Evaluates the saved models
```

### 2. Deep RL (PPO)
Requires `gymnasium` and `stable-baselines3`.
```bash
python train_ppo.py     # Trains a PPO agent for 1M timesteps
python benchmark_ppo.py # Evaluates the PPO policy
```

### 3. LLM / SLM Testing
The `llm_dem_hammurabi.py` script wraps the environment into an agentic tool-calling framework. It parses the complex state into a text report and provides a `SYSTEM_PROMPT` tailored for LLMs.
