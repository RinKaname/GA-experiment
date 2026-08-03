import numpy as np
from cmaes import CMA
from ga_agent import GAPolicy, evaluate_policy
import os
import time

def flatten_weights(policy):
    return np.concatenate([
        policy.W1.flatten(),
        policy.b1.flatten(),
        policy.W2.flatten(),
        policy.b2.flatten()
    ])

def unflatten_weights(policy, flat_weights):
    w1_end = policy.W1.size
    b1_end = w1_end + policy.b1.size
    w2_end = b1_end + policy.W2.size

    policy.W1 = flat_weights[:w1_end].reshape(policy.W1.shape)
    policy.b1 = flat_weights[w1_end:b1_end].reshape(policy.b1.shape)
    policy.W2 = flat_weights[b1_end:w2_end].reshape(policy.W2.shape)
    policy.b2 = flat_weights[w2_end:].reshape(policy.b2.shape)

def train_cmaes(generations=150, population_size=200):
    print(f"Starting CMA-ES Training: Pop={population_size}, Gens={generations}")

    dummy_policy = GAPolicy()
    initial_weights = flatten_weights(dummy_policy)

    optimizer = CMA(mean=initial_weights, sigma=0.5, population_size=population_size)

    best_overall_weights = None
    best_overall_fitness = -float('inf')

    start_time = time.time()

    for generation in range(generations):
        solutions = []
        best_gen_fitness = -float('inf')
        gen_fitnesses = []

        for _ in range(optimizer.population_size):
            x = optimizer.ask()

            test_policy = GAPolicy()
            unflatten_weights(test_policy, x)

            fitness = evaluate_policy(test_policy, num_episodes=5)

            solutions.append((x, -fitness))
            gen_fitnesses.append(fitness)

            if fitness > best_gen_fitness:
                best_gen_fitness = fitness
                if fitness > best_overall_fitness:
                    best_overall_fitness = fitness
                    best_overall_weights = np.copy(x)

        optimizer.tell(solutions)

        avg_fitness = np.mean(gen_fitnesses)
        print(f"Generation {generation+1}/{generations} | Best Gen Fitness: {best_gen_fitness:.2f} | Avg Fitness: {avg_fitness:.2f}")

        if optimizer.should_stop():
            print("CMA-ES optimization converged early!")
            break

    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    print(f"Best overall fitness: {best_overall_fitness:.2f}")

    best_policy = GAPolicy()
    unflatten_weights(best_policy, best_overall_weights)

    os.makedirs('models', exist_ok=True)
    np.save('models/W1_cmaes.npy', best_policy.W1)
    np.save('models/b1_cmaes.npy', best_policy.b1)
    np.save('models/W2_cmaes.npy', best_policy.W2)
    np.save('models/b2_cmaes.npy', best_policy.b2)
    print("Saved best CMA-ES policy to models/")

if __name__ == "__main__":
    train_cmaes(generations=150, population_size=200)
