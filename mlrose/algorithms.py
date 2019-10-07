""" Functions to implement the randomized optimization and search algorithms.
"""

# Author: Genevieve Hayes
# License: BSD 3 clause

import numpy as np
from .decay import GeomDecay
import timeit


def hill_climb(problem, max_iters=np.inf, restarts=0, init_state=None,
               curve=False, timing=False, random_state=None):
    """Use standard hill climbing to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm for each restart.
    restarts: int, default: 0
        Number of random restarts.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
        If :code:`None`, then a random state is used.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    timing: bool, default: False
        Boolean to return fitness values per wall clock time elapsed. If this
        option is True, then curve will automatically be set to True. 
        If :code: `False`, then only fitness scores are returned as an np.array
        If :code:`True`, then a 2d array is returned with time in first col and 
        fitness in second column.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array containing the fitness at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if (not isinstance(restarts, int) and not restarts.is_integer()) \
       or (restarts < 0):
        raise Exception("""restarts must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    best_fitness = -1*np.inf
    best_state = None

    if timing:
        timings = []
        curve = True
    if curve:
        fitness_curve = []

    for _ in range(restarts + 1):
        # Initialize optimization problem
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)

        iters = 0
        start_time = timeit.default_timer()
        while iters < max_iters:
            iters += 1

            # Find neighbors and determine best neighbor
            problem.find_neighbors()
            next_state = problem.best_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # If best neighbor is an improvement, move to that state
            if next_fitness > problem.get_fitness():
                problem.set_state(next_state)

            else:
                break

        # Update best state and best fitness
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()

        if curve:
            fitness_curve.append(problem.get_fitness())
        if timing:
            timings.append(timeit.default_timer() - start_time)

    best_fitness = problem.get_maximize()*best_fitness

    if curve:
        fitness_curve = np.asarray(fitness_curve)
        if timing:
            return best_state, best_fitness, np.hstack((np.asarray(timings).reshape(-1,1),
                                                       fitness_curve.reshape(-1, 1)))
        else:
            return best_state, best_fitness, fitness_curve

    return best_state, best_fitness


def random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=0,
                      init_state=None, curve=False, timing=False,
                      random_state=None):
    """Use randomized hill climbing to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    restarts: int, default: 0
        Number of random restarts.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
        If :code:`None`, then a random state is used.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    timing: bool, default: False
        Boolean to return fitness values per wall clock time elapsed. If this
        option is True, then curve will automatically be set to True. 
        If :code: `False`, then only fitness scores are returned as an np.array
        If :code:`True`, then a 2d array is returned with time in first col and 
        fitness in second column.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array containing the fitness at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.

    References
    ----------
    Brownlee, J (2011). *Clever Algorithms: Nature-Inspired Programming
    Recipes*. `<http://www.cleveralgorithms.com>`_.
    """
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if (not isinstance(restarts, int) and not restarts.is_integer()) \
       or (restarts < 0):
        raise Exception("""restarts must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    best_fitness = -1*np.inf
    best_state = None

    if timing:
        timings = []
        curve = True
    if curve:
        fitness_curve = []

    for _ in range(restarts + 1):
        # Initialize optimization problem and attempts counter
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)

        attempts = 0
        iters = 0
        start_time = timeit.default_timer()

        while (attempts < max_attempts) and (iters < max_iters):
            iters += 1

            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # If best neighbor is an improvement,
            # move to that state and reset attempts counter
            if next_fitness > problem.get_fitness():
                problem.set_state(next_state)
                attempts = 0

            else:
                attempts += 1

            if curve:
                fitness_curve.append(problem.get_fitness())
            if timing:
                timings.append(timeit.default_timer() - start_time)

        # Update best state and best fitness
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()

    best_fitness = problem.get_maximize()*best_fitness

    if curve:
        fitness_curve = np.asarray(fitness_curve)
        if timing:
            return best_state, best_fitness, np.hstack((np.asarray(timings).reshape(-1,1),
                                                       fitness_curve.reshape(-1, 1)))
        else:
            return best_state, best_fitness, fitness_curve

    return best_state, best_fitness


def simulated_annealing(problem, schedule=GeomDecay(), max_attempts=10,
                        max_iters=np.inf, init_state=None, curve=False,
                        timing=False, random_state=None):
    """Use simulated annealing to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    schedule: schedule object, default: :code:`mlrose.GeomDecay()`
        Schedule used to determine the value of the temperature parameter.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
        If :code:`None`, then a random state is used.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    timing: bool, default: False
        Boolean to return fitness values per wall clock time elapsed. If this
        option is True, then curve will automatically be set to True. 
        If :code: `False`, then only fitness scores are returned as an np.array
        If :code:`True`, then a 2d array is returned with time in first col and 
        fitness in second column.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array containing the fitness at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    # Initialize problem, time and attempts counter
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)
    if timing:
        timings = []
        curve = True
    if curve:
        fitness_curve = []

    attempts = 0
    iters = 0
    start_time = timeit.default_timer()

    while (attempts < max_attempts) and (iters < max_iters):
        temp = schedule.evaluate(iters)
        iters += 1

        if temp == 0:
            break

        else:
            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # Calculate delta E and change prob
            delta_e = next_fitness - problem.get_fitness()
            prob = np.exp(delta_e/temp)

            # If best neighbor is an improvement or random value is less
            # than prob, move to that state and reset attempts counter
            if (delta_e > 0) or (np.random.uniform() < prob):
                problem.set_state(next_state)
                attempts = 0

            else:
                attempts += 1

        if curve:
            fitness_curve.append(problem.get_fitness())
        if timing:
            timings.append(timeit.default_timer() - start_time)

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    if curve:
        fitness_curve = np.asarray(fitness_curve)
        if timing:
            return best_state, best_fitness, np.hstack((np.asarray(timings).reshape(-1,1),
                                                       fitness_curve.reshape(-1, 1)))
        else:
            return best_state, best_fitness, fitness_curve

    return best_state, best_fitness


def genetic_alg(problem, pop_size=200, pop_breed_percent=0.75, elite_dreg_ratio=0.99,
                minimum_elites=0, minimum_dregs=0, mutation_prob=0.1,
                max_attempts=10, max_iters=np.inf, curve=False, timing=False,
                random_state=None, state_fitness_callback=None,
                hamming_factor=0.0, hamming_decay_factor=None):
    """Use a standard genetic algorithm to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
    pop_size: int, default: 200
        Size of population to be used in genetic algorithm.
    pop_breed_percent: float, default 0.75
        Percentage of population to breed in each iteration.
        The remainder of the population will be filled from the elite and
        dregs of the prior generation in a ratio specified by elite_dreg_ratio.
    elite_dreg_ratio: float, default:0.95
        The ratio of elites:dregs added directly to the next generation.
        For the default value, 95% of the added population will be elites,
        5% will be dregs.
    minimum_elites: int, default: 0
        Minimum number of elites to be added to next generation
    minimum_dregs: int, default: 0
        Minimum number of dregs to be added to next generation
    mutation_prob: float, default: 0.1
        Probability of a mutation at each element of the state vector
        during reproduction, expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    timing: bool, default: False
        Boolean to return fitness values per wall clock time elapsed. If this
        option is True, then curve will automatically be set to True. 
        If :code: `False`, then only fitness scores are returned as an np.array
        If :code:`True`, then a 2d array is returned with time in first col and 
        fitness in second column.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array of arrays containing the fitness of the entire population
        at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    breeding_pop_size = int(pop_size * pop_breed_percent) - (minimum_elites + minimum_dregs)
    if breeding_pop_size < 1:
        raise Exception("""pop_breed_percent must be large enough to ensure at least one mating.""")

    if pop_breed_percent > 1:
        raise Exception("""pop_breed_percent must be less than 1.""")

    if (elite_dreg_ratio < 0) or (elite_dreg_ratio > 1):
        raise Exception("""elite_dreg_ratio must be between 0 and 1.""")

    if (mutation_prob < 0) or (mutation_prob > 1):
        raise Exception("""mutation_prob must be between 0 and 1.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    if timing:
        timings = []
        curve = True
    if curve:
        fitness_curve = []

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    if state_fitness_callback is not None:
        # initial call with base data
        state_fitness_callback(iteration=0,
                               state=problem.get_state(),
                               fitness=problem.get_adjusted_fitness(),
                               user_data=callback_user_info)
    # check for hamming
    # get_hamming_distance_default_

    get_hamming_distance_func = None
    if hamming_factor > 0:
        g1 = problem.get_population()[0][0]
        if isinstance(g1, float) or g1.dtype == 'float64':
            get_hamming_distance_func = _get_hamming_distance_float
        else:
            get_hamming_distance_func = _get_hamming_distance_default

    attempts = 0
    iters = 0
    start_time = timeit.default_timer()

    # initialize survivor count, elite count and dreg count
    survivors_size = pop_size - breeding_pop_size
    dregs_size = max(int(survivors_size * (1.0 - elite_dreg_ratio)) if survivors_size > 1 else 0, minimum_dregs)
    elites_size = max(survivors_size - dregs_size, minimum_elites)
    if dregs_size + elites_size > survivors_size:
        over_population = dregs_size + elites_size - survivors_size
        breeding_pop_size -= over_population

    continue_iterating = True
    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Calculate breeding probabilities
        problem.eval_mate_probs()

                # Create next generation of population
        next_gen = []
        for _ in range(breeding_pop_size):
            # Select parents
            parent_1, parent_2 = _genetic_alg_select_parents(pop_size=pop_size,
                                                             problem=problem,
                                                             hamming_factor=hamming_factor,
                                                             get_hamming_distance_func=get_hamming_distance_func)

            # Create offspring
            child = problem.reproduce(parent_1, parent_2, mutation_prob)
            next_gen.append(child)

        # fill remaining population with elites/dregs
        if survivors_size > 0:
            last_gen = list(zip(problem.get_population(), problem.get_pop_fitness()))
            sorted_parents = sorted(last_gen, key=lambda f: -f[1])
            best_parents = sorted_parents[:elites_size]
            next_gen.extend([p[0] for p in best_parents])
            if dregs_size > 0:
                worst_parents = sorted_parents[-dregs_size:]
                next_gen.extend([p[0] for p in worst_parents])

        next_gen = np.array(next_gen[:pop_size])
        problem.set_population(next_gen)

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0

        else:
            attempts += 1

        if curve:
            fitness_curve.append(problem.get_fitness())
        if timing:
            timings.append(timeit.default_timer() - start_time)

    # decay hamming factor
        if hamming_decay_factor is not None and hamming_factor > 0.0:
            hamming_factor *= hamming_decay_factor
            hamming_factor = max(min(hamming_factor, 1.0), 0.0)
        # print(hamming_factor)

        # break out if requested
        if not continue_iterating:
            break
    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    if curve:
        fitness_curve = np.asarray(fitness_curve)
        if timing:
            return best_state, best_fitness, np.hstack((np.asarray(timings).reshape(-1,1),
                                                       fitness_curve.reshape(-1, 1)))
        else:
            return best_state, best_fitness, fitness_curve

    return best_state, best_fitness


def _get_hamming_distance_default(population, p1):
    hamming_distances = np.array([np.count_nonzero(p1 != p2) / len(p1) for p2 in population])
    return hamming_distances


def _get_hamming_distance_float(population, p1):
    # use squares instead?
    hamming_distances = np.array([np.abs(np.diff(p1, p2)) / len(p1) for p2 in population])
    return hamming_distances


def _genetic_alg_select_parents(pop_size, problem,
                                get_hamming_distance_func,
                                hamming_factor=0.0):
    mating_probabilities = problem.get_mate_probs()
    if get_hamming_distance_func is not None and hamming_factor > 0.01:
        selected = np.random.choice(pop_size, p=mating_probabilities)
        population = problem.get_population()
        p1 = population[selected]
        hamming_distances = get_hamming_distance_func(population, p1)
        hfa = hamming_factor / (1.0 - hamming_factor)
        hamming_distances = (hamming_distances * hfa) * mating_probabilities
        hamming_distances /= hamming_distances.sum()
        selected = np.random.choice(pop_size, p=hamming_distances)
        p2 = population[selected]

        return p1, p2

    selected = np.random.choice(pop_size,
                                size=2,
                                p=mating_probabilities)
    p1 = problem.get_population()[selected[0]]
    p2 = problem.get_population()[selected[1]]
    return p1, p2


def mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10,
          max_iters=np.inf, curve=False, timing=False,
          random_state=None, fast_mimic=False):
    """Use MIMIC to find the optimum for a given optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()` or :code:`TSPOpt()`.
    pop_size: int, default: 200
        Size of population to be used in algorithm.
    keep_pct: float, default: 0.2
        Proportion of samples to keep at each iteration of the algorithm,
        expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    timing: bool, default: False
        Boolean to return fitness values per wall clock time elapsed. If this
        option is True, then curve will automatically be set to True. 
        If :code: `False`, then only fitness scores are returned as an np.array
        If :code:`True`, then a 2d array is returned with time in first col and 
        fitness in second column.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.
    fast_mimic: bool, default: False
        Activate fast mimic mode to compute the mutual information in vectorized form
        Faster speed but requires more memory.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array containing the fitness at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.

    References
    ----------
    De Bonet, J., C. Isbell, and P. Viola (1997). MIMIC: Finding Optima by
    Estimating Probability Densities. In *Advances in Neural Information
    Processing Systems* (NIPS) 9, pp. 424–430.

    Note
    ----
    MIMIC cannot be used for solving continuous-state optimization problems.
    """
    if problem.get_prob_type() == 'continuous':
        raise Exception("""problem type must be discrete or tsp.""")

    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    if (keep_pct < 0) or (keep_pct > 1):
        raise Exception("""keep_pct must be between 0 and 1.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    if timing:
        timings = []
        curve = True
    if curve:
        fitness_curve = []

    if not((fast_mimic == True) or (fast_mimic == False)):
        raise Exception("""fast_mimic mode must be a boolean.""")
    else:
        problem.mimic_speed=fast_mimic

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0
    start_time = timeit.default_timer()

    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Get top n percent of population
        problem.find_top_pct(keep_pct)

        # Update probability estimates
        problem.eval_node_probs()

        # Generate new sample
        new_sample = problem.sample_pop(pop_size)
        problem.set_population(new_sample)

        next_state = problem.best_child()

        next_fitness = problem.eval_fitness(next_state)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0

        else:
            attempts += 1

        if curve:
            fitness_curve.append(problem.get_fitness())
        if timing:
            timings.append(timeit.default_timer() - start_time)

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state().astype(int)

    if curve:
        fitness_curve = np.asarray(fitness_curve)
        if timing:
            return best_state, best_fitness, np.hstack((np.asarray(timings).reshape(-1,1),
                                                       fitness_curve.reshape(-1, 1)))
        else:
            return best_state, best_fitness, fitness_curve

    return best_state, best_fitness
