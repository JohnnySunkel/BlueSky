# Bit counter

# import packages
import random
from deap import base, creator, tools

# define an evaluation function
def eval_func(individual):
    target_sum = 45
    return len(individual) - abs(sum(individual) - target_sum),

# define a fucntion to create the toolbox
def create_toolbox(num_bits):
    creator.create("FitnessMax", base.Fitness, weights = (1.0, ))
    creator.create("Individual", list, fitness = creator.FitnessMax)
    
    # initialize the toolbox
    toolbox = base.Toolbox()
    
    # generate attributes
    toolbox.register("attr_bool", random.randint, 0, 1)
    
    # initialize structures
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual,
                     toolbox.attr_bool, num_bits)
    
    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    
    # register the evaluation operator
    toolbox.register("evaluate", eval_func)
    
    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)
    
    # register the mutation operator
    toolbox.register("mutate", tools.mutFlipBit, indpb = 0.05)
    
    # register the selection operator
    toolbox.register("select", tools.selTournament, tournsize = 3)
    
    return toolbox
    

# define the main function
if __name__ == "__main__":
    # define the number of bits
    num_bits = 75
    
    # create a toolbox
    toolbox = create_toolbox(num_bits)
    
    # seed the random number generator
    random.seed(7)
    
    # create an initial population of 500 individuals
    population = toolbox.population(n = 500)
    
    # define the probabilities of crossing and mutating
    probab_crossing, probab_mutating = 0.5, 0.2
    
    # define the number of generations
    num_generations = 60
    
    # evaluate all the individuals in the population using
    # the fitness functions
    print('\nStarting the evolution process')
    
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
        
    # start iterating through the generations
    print('\nEvaluated', len(population), 'individuals')
    
    for g in range(num_generations):
        print("\n===== Generation", g)
        
        # select the next generation individuals
        offspring = toolbox.select(population, len(population))
        
        # clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals
            if random.random() < probab_crossing:
                toolbox.mate(child1, child2)
                
                # reset the fitness values of the children
                del child1.fitness.values
                del child2.fitness.values
                
        # apply mutation
        for mutant in offspring:
            # mutate an individual
            if random.random() < probab_mutating:
                toolbox.mutate(mutant)
                
                # reset the fitness values
                del mutant.fitness.values
                
        # evaluate the individuals with invalid fitness values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print('Evaluated', len(invalid_ind), 'individuals')
        
        # replace the entire population with the offspring
        population[:] = offspring

        # gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]

        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        
        print('Min =', min(fits), ', Max =', max(fits))
        print('Average =', round(mean, 2), ', Standard deviation =', 
              round(std, 2))
    
    print("\n==== End of evolution")
    
    # print the final output
    best_ind = tools.selBest(population, 1)[0]
    print('\nBest individual:\n', best_ind)
    print('\nNumber of ones:', sum(best_ind))
