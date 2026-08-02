import numpy as np
from ga_agent import GAPolicy, evaluate_policy, crossover
import copy
import time
import os

def train_ga(population_size=200, generations=150, mutation_rate=0.2, mutation_scale=0.2):
    print(f"Starting GA Training: Pop={population_size}, Gens={generations}")

    # Initialize population
    population = [GAPolicy() for _ in range(population_size)]


    best_overall_policy = None
    best_overall_fitness = -float('inf')

    for gen in range(generations):
        # Evaluate all policies (increased num_episodes to 10 to smooth out RNG)
        fitnesses = [evaluate_policy(p, num_episodes=10) for p in population]

        # Track best
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]

        if best_fitness > best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_policy = copy.deepcopy(population[best_idx])

        print(f"Generation {gen+1}/{generations} | Best Fitness: {best_fitness:.2f} | Avg Fitness: {np.mean(fitnesses):.2f}")

        # Selection: Tournament selection
        new_population = []

        # Elitism: keep best 2 policies unchanged
        sorted_indices = np.argsort(fitnesses)[::-1]
        new_population.append(copy.deepcopy(population[sorted_indices[0]]))
        new_population.append(copy.deepcopy(population[sorted_indices[1]]))

        while len(new_population) < population_size:
            # Tournament selection (size 3)
            tournament1 = np.random.choice(population_size, 3, replace=False)
            parent1 = population[tournament1[np.argmax([fitnesses[i] for i in tournament1])]]

            tournament2 = np.random.choice(population_size, 3, replace=False)
            parent2 = population[tournament2[np.argmax([fitnesses[i] for i in tournament2])]]

            # Crossover
            child = crossover(parent1, parent2)

            # Mutate
            child.mutate(mutation_rate, mutation_scale)

            new_population.append(child)

        population = new_population

    print("Training complete!")
    print(f"Best overall fitness: {best_overall_fitness:.2f}")
    return best_overall_policy

if __name__ == "__main__":
    start_time = time.time()
    best_policy = train_ga(population_size=100, generations=100, mutation_rate=0.2, mutation_scale=0.2)
    print(f"Time taken: {time.time() - start_time:.2f}s")

    # Save best policy weights for MLP
    os.makedirs('models', exist_ok=True)
    np.save('models/W1.npy', best_policy.W1)
    np.save('models/b1.npy', best_policy.b1)
    np.save('models/W2.npy', best_policy.W2)
    np.save('models/b2.npy', best_policy.b2)
    print("Saved best policy to models/")
