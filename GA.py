import random
from deap import creator, base, tools, algorithms
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

#READING INPUT FROM TXT FILE
f = open('input.txt','r')
arr = f.read().split('\n',1)
N=arr[0]
N = int(N,10)
rating = arr[1]
rating = rating.split(' ')
f.close()
rating = map(int,rating)


#CHROMOSOME ENCODING
toolbox.register("attr_int", random.randint, 1, 100)

#CREATING INITIAL POPULATION
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=N)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


#DEFINING THE FITNESS FUNCTION
def fitness(individual):
	sum=0
	for i in range(N): sum+=individual[i]*rating[i]*rating[i]*rating[i]
	return sum,

    #return sum(individual),

#FITNESS FUNCTION PASSED TO TOOLBOX
toolbox.register("evaluate", fitness)

#MATING TWO INDIVIDUALS
toolbox.register("mate", tools.cxTwoPoint)

#MUTATION
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

#TOURNAMENT SELECTION USED
toolbox.register("select", tools.selTournament, tournsize=3)


#SETTING INITIAL POPULATION AS 300
population = toolbox.population(n=300)


#CALCULATING FOR 'NGEN' NUMBER OF GENERATIONS
NGEN=40
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
top1 = tools.selBest(population, k=1)
#print top1[0]

plt.plot(rating,top1[0],'ro')
#plt.axis([0, 6, 0, 20])
plt.show()