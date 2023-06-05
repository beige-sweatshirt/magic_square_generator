import random
import numpy as np


def init_square(n):
    values = np.arange(1, n*n+1) 
    np.random.shuffle(values)
    square = values.reshape((n,n))
    return square

def eval(square):
    n = len(square)
    sums = np.zeros(2 * n + 2)

    # calculate row / column sums
    for i in range(n): sums[i] = np.sum(square[i,:])
    for i in range(n): sums[i + n] = np.sum(square[:,i])

    # calculate diagonal sums
    sums[2 * n] = np.sum(square.diagonal())
    sums[2 * n + 1] = np.sum(np.fliplr(square).diagonal())

    # Return standard deviation of all sums
    std = np.std(sums)
    return std

def fitness_prop_selection(pop, pop_fitness):
    # Calculate the normalized fitness scores
    total_fitness = np.sum(pop_fitness)
    norm_fitness = pop_fitness / total_fitness
    
    # Select parents with probabilities proportional to their fitness scores.
    parent_idxs = np.random.choice(len(pop), size=2, p=norm_fitness, replace=False)
    parents = [pop[idx] for idx in parent_idxs]
    
    return parents

def mutate(square):
    n = len(square)
    # Choose two random indices
    i1, j1, i2, j2 = np.random.randint(0, n, size=4)
    # Swap the cells at these indices
    square[i1, j1], square[i2, j2] = square[i2, j2], square[i1, j1]
    return square

def tournament_selection(pop, pop_fitness, tournament_size):
    # Randomly select a number of individuals from the population
    tournament = np.random.choice(len(pop), tournament_size, replace=False)
    # Get the fitnesses of these individuals
    tournament_fitnesses = [pop_fitness[i] for i in tournament]
    # Select the individual with the best fitness
    winner = tournament[np.argmin(tournament_fitnesses)]
    return pop[winner]

def crossover(parent1, parent2):
    n = len(parent1)
    child = np.empty((n, n), dtype=int)

    # Convert 2D parents to 1D
    parent1_1D = parent1.flatten()
    parent2_1D = parent2.flatten()

    # Pick a crossover point
    cross_point = np.random.randint(0, n*n)

    # Combine data from both parents
    child_1D = np.concatenate((parent1_1D[:cross_point], parent2_1D[cross_point:]), axis=None)
    child = child_1D.reshape(n, n)

    # Replace duplicates
    numbers = list(range(1, n*n+1))
    random.shuffle(numbers)
    count = np.zeros(n*n+1, dtype=int)
    for i in range(n):
        for j in range(n):
            count[child[i, j]] += 1
            if count[child[i, j]] > 1:
                while numbers and count[numbers[-1]] != 0:
                    numbers.pop()
                if numbers:
                    new_num = numbers.pop()
                    child[i, j] = new_num
                    count[new_num] += 1

    return child

def acceptance_probability(old_fitness, new_fitness, T):
    if new_fitness < old_fitness:
        return 1.0
    else:
        return np.exp((old_fitness - new_fitness) / T)

def simulated_annealing(pop, pop_fitness, T):
    for i in range(len(pop)):
        old_square = pop[i]
        old_fitness = pop_fitness[i]
        new_square = mutate(old_square.copy())
        new_fitness = eval(new_square)
        if acceptance_probability(old_fitness, new_fitness, T) > np.random.rand():
            pop[i] = new_square
            pop_fitness[i] = new_fitness
    return pop, pop_fitness

square_size = 6
num_elites = 75
pop_size = 100
max_generations = 500
mutation_rate = 0.5
tournament_size = 5
print(f"Square Size: {square_size}")
print(f"Population Size: {pop_size}")

pop = [init_square(square_size) for i in range(pop_size)]
pop_fitness = [eval(pop[i]) for i in range(pop_size)]

while(True):
    for gen in range(max_generations):
        new_pop = []

        # Sort the current population and fitness by fitness
        sorted_pop_with_fitness = sorted(zip(pop, pop_fitness), key=lambda x: x[1])
        sorted_pop = [x[0] for x in sorted_pop_with_fitness]
        sorted_fitness = [x[1] for x in sorted_pop_with_fitness]

        # Add the elites to the new population
        for i in range(num_elites):
            new_pop.append(sorted_pop[i])

        for _ in range(pop_size - num_elites):
            mom = tournament_selection(sorted_pop, sorted_fitness, tournament_size)
            dad = tournament_selection(sorted_pop, sorted_fitness, tournament_size)
            child = crossover(mom,dad)
            if np.random.random() < mutation_rate:
                child = mutate(child)
            new_pop.append(child)
        pop = new_pop
        
        pop_fitness = [eval(pop[i]) for i in range(pop_size)]
        fittest = np.min(pop_fitness)
        print("Generation", gen, "Best Fitness", fittest, "Average Fitness", np.average(pop_fitness))
        if (fittest == 0):
            print(f"Found a {square_size}x{square_size} magic square at generation {gen}!")
            print()
            break
    max_generations = 0
    pop = [init_square(square_size) for i in range(pop_size)]
    pop_fitness = [eval(pop[i]) for i in range(pop_size)]
