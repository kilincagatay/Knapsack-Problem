# coding=utf-8
from simpleai.search.utils import BoundedPriorityQueue, InverseTransformSampler
from simpleai.search.models import SearchNodeValueOrdered
import math
import random
import math
import numpy as np
import random as rd
from random import randint
from random import Random  
import numpy as np






def _all_expander(fringe, iteration, viewer):
    '''
    Expander that expands all nodes on the fringe.
    '''
    expanded_neighbors = [node.expand(local_search=True)
                          for node in fringe]

    if viewer:
        viewer.event('expanded', list(fringe), expanded_neighbors)

    list(map(fringe.extend, expanded_neighbors))


def beam(problem, beam_size=100, iterations_limit=0, viewer=None):
    '''
    Beam search.

    beam_size is the size of the beam.
    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.actions, SearchProblem.result, SearchProblem.value,
    and SearchProblem.generate_random_state.
    '''
    return _local_search(problem,
                         _all_expander,
                         iterations_limit=iterations_limit,
                         fringe_size=beam_size,
                         random_initial_states=True,
                         stop_when_no_better=iterations_limit==0,
                         viewer=viewer)


def _first_expander(fringe, iteration, viewer):
    '''
    Expander that expands only the first node on the fringe.
    '''
    current = fringe[0]
    neighbors = current.expand(local_search=True)

    if viewer:
        viewer.event('expanded', [current], [neighbors])

    fringe.extend(neighbors)



def beam_best_first(problem, beam_size=100, iterations_limit=0, viewer=None):
    '''
    Beam search best first.

    beam_size is the size of the beam.
    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.actions, SearchProblem.result, and
    SearchProblem.value.
    '''
    return _local_search(problem,
                         _first_expander,
                         iterations_limit=iterations_limit,
                         fringe_size=beam_size,
                         random_initial_states=True,
                         stop_when_no_better=iterations_limit==0,
                         viewer=viewer)


def hill_climbing(problem, iterations_limit=0, viewer=None):
    '''
    Hill climbing search.

    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.actions, SearchProblem.result, and
    SearchProblem.value.
    '''
    return _local_search(problem,
                         _first_expander,
                         iterations_limit=iterations_limit,
                         fringe_size=1,
                         stop_when_no_better=True,
                         viewer=viewer)


def _random_best_expander(fringe, iteration, viewer):
    '''
    Expander that expands one randomly chosen nodes on the fringe that
    is better than the current (first) node.
    '''
    current = fringe[0]
    neighbors = current.expand(local_search=True)
    if viewer:
        viewer.event('expanded', [current], [neighbors])

    betters = [n for n in neighbors
               if n.value > current.value]
    if betters:
        chosen = random.choice(betters)
        if viewer:
            viewer.event('chosen_node', chosen)
        fringe.append(chosen)


def hill_climbing_stochastic(problem, iterations_limit=0, viewer=None):
    '''
    Stochastic hill climbing.

    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.actions, SearchProblem.result, and
    SearchProblem.value.
    '''
    return _local_search(problem,
                         _random_best_expander,
                         iterations_limit=iterations_limit,
                         fringe_size=1,
                         stop_when_no_better=iterations_limit==0,
                         viewer=viewer)


def hill_climbing_random_restarts(problem, restarts_limit, iterations_limit=0, viewer=None):
    '''
    Hill climbing with random restarts.

    restarts_limit specifies the number of times hill_climbing will be runned.
    If iterations_limit is specified, each hill_climbing will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.actions, SearchProblem.result, SearchProblem.value,
    and SearchProblem.generate_random_state.
    '''
    restarts = 0
    best = None

    while restarts < restarts_limit:
        new = _local_search(problem,
                            _first_expander,
                            iterations_limit=iterations_limit,
                            fringe_size=1,
                            random_initial_states=True,
                            stop_when_no_better=True,
                            viewer=viewer)

        if not best or best.value < new.value:
            best = new

        restarts += 1

    if viewer:
        viewer.event('no_more_runs', best, 'returned after %i runs' % restarts_limit)

    return best


# Math literally copied from aima-python
def _exp_schedule(iteration, k=20, lam=0.005, limit=100):
    '''
    Possible scheduler for simulated_annealing, based on the aima example.
    '''
    return k * math.exp(-lam * iteration)


def _create_simulated_annealing_expander(schedule):
    '''
    Creates an expander that has a random chance to choose a node that is worse
    than the current (first) node, but that chance decreases with time.
    '''
    def _expander(fringe, iteration, viewer):
        T = schedule(iteration)
        current = fringe[0]
        neighbors = current.expand(local_search=True)

        if viewer:
            viewer.event('expanded', [current], [neighbors])

        if neighbors:
            succ = random.choice(neighbors)
            delta_e = succ.value - current.value
            if delta_e > 0 or random.random() < math.exp(delta_e / T):
                fringe.pop()
                fringe.append(succ)

                if viewer:
                    viewer.event('chosen_node', succ)

    return _expander


def simulated_annealing(problem, schedule=_exp_schedule, iterations_limit=0, viewer=None):
    '''
    Simulated annealing.

    schedule is the scheduling function that decides the chance to choose worst
    nodes depending on the time.
    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.actions, SearchProblem.result, and
    SearchProblem.value.
    '''
    return _local_search(problem,
                         _create_simulated_annealing_expander(schedule),
                         iterations_limit=iterations_limit,
                         fringe_size=1,
                         stop_when_no_better=iterations_limit==0,
                         viewer=viewer)


def _create_genetic_expander(problem, mutation_chance):
    '''
    Creates an expander that expands the bests nodes of the population,
    crossing over them.
    '''
    def _expander(fringe, iteration, viewer):
        fitness = [x.value for x in fringe]
        sampler = InverseTransformSampler(fitness, fringe)
        new_generation = []

        expanded_nodes = []
        expanded_neighbors = []

        for _ in fringe:
            node1 = sampler.sample()
            node2 = sampler.sample()
            child = problem.crossover(node1.state, node2.state)
            action = 'crossover'
            if random.random() < mutation_chance:
                # Noooouuu! she is... he is... *IT* is a mutant!
                child = problem.mutate(child)
                action += '+mutation'

            child_node = SearchNodeValueOrdered(state=child, problem=problem, action=action)
            new_generation.append(child_node)

            expanded_nodes.append(node1)
            expanded_neighbors.append([child_node])
            expanded_nodes.append(node2)
            expanded_neighbors.append([child_node])

        if viewer:
            viewer.event('expanded', expanded_nodes, expanded_neighbors)

        fringe.clear()
        for node in new_generation:
            fringe.append(node)

    return _expander


def genetic(problem, population_size=100, mutation_chance=0.1,
            iterations_limit=0, viewer=None):
    '''
    Genetic search.

    population_size specifies the size of the population (ORLY).
    mutation_chance specifies the probability of a mutation on a child,
    varying from 0 to 1.
    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.generate_random_state, SearchProblem.crossover,
    SearchProblem.mutate and SearchProblem.value.
    '''
    return _local_search(problem,
                         _create_genetic_expander(problem, mutation_chance),
                         iterations_limit=iterations_limit,
                         fringe_size=population_size,
                         random_initial_states=True,
                         stop_when_no_better=iterations_limit==0,
                         viewer=viewer)


def _local_search(problem, fringe_expander, iterations_limit=0, fringe_size=1,
                  random_initial_states=False, stop_when_no_better=True,
                  viewer=None):
    '''
    Basic algorithm for all local search algorithms.
    '''
    if viewer:
        viewer.event('started')

    fringe = BoundedPriorityQueue(fringe_size)
    if random_initial_states:
        for _ in range(fringe_size):
            s = problem.generate_random_state()
            fringe.append(SearchNodeValueOrdered(state=s, problem=problem))
    else:
        fringe.append(SearchNodeValueOrdered(state=problem.initial_state,
                                             problem=problem))

    finish_reason = ''
    iteration = 0
    run = True
    best = None

    while run:
        if viewer:
            viewer.event('new_iteration', list(fringe))

        old_best = fringe[0]
        fringe_expander(fringe, iteration, viewer)
        best = fringe[0]

        iteration += 1

        if iterations_limit and iteration >= iterations_limit:
            run = False
            finish_reason = 'reaching iteration limit'
        elif old_best.value >= best.value and stop_when_no_better:
            run = False
            finish_reason = 'not being able to improve solution'

    if viewer:
        viewer.event('finished', fringe, best, 'returned after %s' % finish_reason)

    return best


##########################################



print("Please enter the weight value of the item between the numbers 1-15.")
print("Please enter the value of the item between the numbers 10-750.")

item_number = np.arange(1,11)
weight=[]
value=[]


weight2=int(input("Enter the weight of the item."))
value2=int(input("Enter the value of the item."))
weight3=int(input("Enter the weight of the item."))
value3=int(input("Enter the value of the item."))
weight4=int(input("Enter the weight of the item."))
value4=int(input("Enter the value of the item."))
weight5=int(input("Enter the weight of the item."))
value5=int(input("Enter the value of the item."))
weight6=int(input("Enter the weight of the item."))
value6=int(input("Enter the value of the item."))
weight7=int(input("Enter the weight of the item."))
value7=int(input("Enter the value of the item."))
weight8=int(input("Enter the weight of the item."))
value8=int(input("Enter the value of the item."))
weight9=int(input("Enter the weight of the item."))
value9=int(input("Enter the value of the item."))
weight10=int(input("Enter the weight of the item."))
value10=int(input("Enter the value of the item."))
weight11=int(input("Enter the weight of the item."))
value11=int(input("Enter the value of the item."))

weight.append(weight2)
value.append(value2)
weight.append(weight3)
value.append(value3)
weight.append(weight4)
value.append(value4)
weight.append(weight5)
value.append(value5)
weight.append(weight6)
value.append(value6)
weight.append(weight7)
value.append(value7)
weight.append(weight8)
value.append(value8)
weight.append(weight9)
value.append(value9)
weight.append(weight10)
value.append(value10)
weight.append(weight11)
value.append(value11)



knapsack_threshold = 35    
print('The list is as follows:')
print('Item No.   Weight   Value')
for i in range(item_number.shape[0]):
    print('{0}          {1}         {2}\n'.format(item_number[i], weight[i], value[i]))
    



solutions_per_pop = 8
pop_size = (solutions_per_pop, item_number.shape[0])
print('Population size = {}'.format(pop_size))
initial_population = np.random.randint(2, size = pop_size)
initial_population = initial_population.astype(int)
num_generations = 50
print('Initial population: \n{}'.format(initial_population))



def cal_fitness(weight, value, population, threshold):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * value)
        S2 = np.sum(population[i] * weight)
        if S2 <= threshold:
            fitness[i] = S1
        else :
            fitness[i] = 0 
    return fitness.astype(int)     



def selection(fitness, num_parents, population):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i,:] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents




def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1]/2)
    crossover_rate = 0.8
    i=0
    while (parents.shape[0] < num_offsprings):
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        x = rd.random()
        if x > crossover_rate:
            continue
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        offsprings[i,0:crossover_point] = parents[parent1_index,0:crossover_point]
        offsprings[i,crossover_point:] = parents[parent2_index,crossover_point:]
        i=+1
    return offsprings   



    
def mutation(offsprings):
    mutants = np.empty((offsprings.shape))
    mutation_rate = 0.4
    for i in range(mutants.shape[0]):
        random_value = rd.random()
        mutants[i,:] = offsprings[i,:]
        if random_value > mutation_rate:
            continue
        int_random_value = randint(0,offsprings.shape[1]-1)    
        if mutants[i,int_random_value] == 0 :
            mutants[i,int_random_value] = 1
        else :
            mutants[i,int_random_value] = 0
    return mutants    




def optimize(weight, value, population, pop_size, num_generations, threshold):
    parameters, fitness_history = [], []
    num_parents = int(pop_size[0]/2)
    num_offsprings = pop_size[0] - num_parents 
    for i in range(num_generations):
        fitness = cal_fitness(weight, value, population, threshold)
        fitness_history.append(fitness)
        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants
        
    print('Last generation: \n{}\n'.format(population)) 
    fitness_last_gen = cal_fitness(weight, value, population, threshold)      
    print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0],:])
    return parameters, fitness_history


parameters, fitness_history = optimize(weight, value, initial_population, pop_size, num_generations, knapsack_threshold)
print('The optimized parameters for the given inputs are: \n{}'.format(parameters))
selected_items = item_number * parameters
print('\nSelected items that will maximize the knapsack without breaking it:')
for i in range(selected_items.shape[1]):
  if selected_items[0][i] != 0:
     print('{}\n'.format(selected_items[0][i]))

