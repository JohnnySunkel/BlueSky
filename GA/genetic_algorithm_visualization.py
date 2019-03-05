# Visualizing an evolutionary algorithm

# import packages
import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms, base, benchmarks, cma, creator, tools

# define a function to create the toolbox
def create_toolbox(strategy):
    creator.create("FitnessMin", base.Fitness, weights = (-1.0, ))
    creator.create("Individual", list, fitness = creator.FitnessMin)
    
    # create the toolbox and register the evaluation function
    toolbox = base.Toolbox()
    toolbox.register("evaluate", benchmarks.rastrigin)
    
    # seed the random number generator
    np.random.seed(7)
    
    # register the generate and update methods
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    
    return toolbox
    
    
# define the main function
if __name__ == "__main__":
    # problem size
    num_individuals = 10
    num_generations = 125
    
    # create a strategy using the CMA-ES algorithm
    strategy = cma.Strategy(centroid = [5.0] * num_individuals,
                            sigma = 5.0,
                            lambda_ = 20 * num_individuals)
    
    # create the toolbox based on the strategy
    toolbox = create_toolbox(strategy)
    
    # create Hall of Fame object
    hall_of_fame = tools.HallOfFame(1)
    
    # register the relevant statistics
    stats = tools.Statistics(lambda x: x.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # define the logbook to keep track of the evolution records
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    # define the objects that will compile the data
    sigma = np.ndarray((num_generations, 1))
    axis_ratio = np.ndarray((num_generations, 1))
    diagD = np.ndarray((num_generations, num_individuals))
    fbest = np.ndarray((num_generations, 1))
    best = np.ndarray((num_generations, num_individuals))
    std = np.ndarray((num_generations, num_individuals))
    
    # iterate through the generations
    for gen in range(num_generations):
        # generate a new population
        population = toolbox.generate()
        
        # evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
            
        # update the strategy with the evaluated individuals
        toolbox.update(population)
        
        # update the Hall of Fame and the statistics with the
        # currently evaluated population
        hall_of_fame.update(population)
        record = stats.compile(population)
        logbook.record(evals = len(population), gen = gen, ** record)
        
        print(logbook.stream)
        
        # save more data along the evolution for plotting
        sigma[gen] = strategy.sigma
        axis_ratio[gen] = max(strategy.diagD) ** 2 / min(strategy.diagD) ** 2
        diagD[gen, :num_individuals] = strategy.diagD ** 2
        fbest[gen] = hall_of_fame[0].fitness.values
        best[gen, :num_individuals] = hall_of_fame[0]
        std[gen, :num_individuals] = np.std(population, axis = 0)
        
    # the x-axis will be the number of evaluations
    x = list(range(0, strategy.lambda_ * num_generations,
                   strategy.lambda_))
    avg, max_, min_ = logbook.select("avg", "max", "min")
    plt.figure()
    plt.semilogy(x, avg, "--b")
    plt.semilogy(x, max_, "--b")
    plt.semilogy(x, min_, "-b")
    plt.semilogy(x, fbest, "-c")
    plt.semilogy(x, sigma, "-g")
    plt.semilogy(x, axis_ratio, "-r")
    plt.grid(True)
    plt.title("blue: f-values, green: sigma, red: axis ratio")
    
    # plot the progress
    plt.figure()
    plt.plot(x, best)
    plt.grid(True)
    plt.title("Object Variables")
    
    plt.figure()
    plt.semilogy(x, diagD)
    plt.grid(True)
    plt.title("Scaling (All Main Axes)")
    
    plt.figure()
    plt.semilogy(x, std)
    plt.grid(True)
    plt.title("Standard Deviations in All Coordinates")
    plt.show()
