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



seed = 5113
myPRNG = Random(seed)

n = 100


value = []
for i in range(0,n):
    value.append(round(myPRNG.triangular(5,1000,200),1))
    
weights = []
for i in range(0,n):
    weights.append(round(myPRNG.triangular(10,200,60),1))
    

maxWeight = 1500
solutionsChecked = 0


def evaluate(x):
          
    a=np.array(x)
    b=np.array(value)
    c=np.array(weights)
    
    totalValue = np.dot(a,b)     
    totalWeight = np.dot(a,c)   
    
    if totalWeight > maxWeight:
        raise ValueError
    return [totalValue, totalWeight]  
          
        
def neighborhood(x):
        
    nbrhood = []     
    
    for i in range(0,n):
        nbrhood.append(np.copy(x))
        if nbrhood[i][i] == 1:
            nbrhood[i][i] = 0
        else:
            nbrhood[i][i] = 1
    return nbrhood


def initial_solution():
    x = np.zeros(n, dtype=int) 
    num_ones = myPRNG.randint(5,int(n/4))  
    x[np.random.randint(n, size=num_ones)] = 1 
    total_w = np.dot(x,np.array(weights))
    
    
    while (total_w > maxWeight):
        x = np.zeros(n, dtype=int) 
        num_ones = myPRNG.randint(5,int(n/4))  
        x[np.random.randint(n, size=num_ones)] = 1 
        total_w = np.dot(x,np.array(weights))
    return x


solutionsChecked = 0

n_restarts = 200  
best_vals = []  
best_weights = [] 
best_x = []  



for nrs in range(n_restarts):
    x_curr = initial_solution()  
    x_best = np.copy(x_curr)     
    f_curr = evaluate(x_curr)    
    f_best = np.copy(f_curr)
    
    
    done = 0
        
    while done == 0:
            
        Neighborhood = neighborhood(x_curr)  
        for s in Neighborhood:              
            solutionsChecked = solutionsChecked + 1
            try:
                eval_s = evaluate(s)
            except:
                continue
            
            if eval_s[0] > f_best[0]: 
                x_best = np.copy(s)                
                f_best = np.copy(eval_s)      
            
        if list(f_best) == list(f_curr):              
            done = 1
        else:
            x_curr = np.copy(x_best)        
            f_curr = np.copy(f_best)        
            print ("\nTotal number of solutions checked: ", solutionsChecked)
            print ("Best value found so far: ", f_best)    
            
    best_vals.append(f_best[0])
    best_weights.append(f_best[1])
    best_x.append(x_best)
    
print ("\nFinal number of solutions checked: ", solutionsChecked)
print ("Best value found: ", np.max(best_vals))
print ("Weight is: ", best_weights[np.argmax(best_vals)])
print ("Total number of items selected: ", np.sum(best_x[np.argmax(best_vals)]))
print ("Best solution: ", best_x[np.argmax(best_vals)])