'''
ways of roulette wheel selection
'''

import random
import numpy as np
import time
import matplotlib.pyplot as plt

def roulette_basic(fitness, times):
    '''
    input: a list of fitness, resampling times
    output: selected index
    O(N^2)
    '''
    sumFit = sum(fitness)
    resam_index = []
    for i in range(times):
        rand = random.uniform(0, sumFit)
        accum = 0.0
        for ind, val in enumerate(fitness):
            accum += val
            if accum >= rand:
                resam_index.append(ind)
                break
    return resam_index

def roulette_basic_2(fitness, times):
    '''
    input: a list of fitness, resampling times
    output: selected index
    O(N^2)
    '''
    accm_fitness = []
    accm_for_sum = []
    for val in fitness:
        accm_for_sum.append(val)
        accm_fitness.append(sum(accm_for_sum))

    sumFit = sum(fitness)
    resam_index = []
    for i in range(times):
        rand = random.uniform(0, sumFit)
        for ind, val in enumerate(accm_fitness):
            if val >= rand:
                resam_index.append(ind)
                break
    return resam_index

# ==========================
# time test

N = [1000,1000*2,1000*3,1000*4,1000*5,1000*6,1000*7,1000*8]
algos = [roulette_basic, roulette_basic_2]
times = [[], []]
for n in N:
    fitness = np.random.random((n,))
    for ind,algo in enumerate(algos):
        start = time.time()
        algo(fitness,n)
        end = time.time()
        times[ind].append(end-start)
lineStyle = ['b-o', 'g--p', 'r:*']
algoName = ['roulette_basic', 'roulette_basic_2']
for i in range(len(times)):
    plt.plot(N, times[i], lineStyle[i], label=algoName[i])
plt.legend(loc=2)
plt.title('plot of average running time')
plt.xlabel('N')
plt.ylabel('Average Running Time')
plt.grid('on')
plt.show()









'''

# from __future__ import division
import random
from bisect import bisect_left
import numpy as np
import timeit
import matplotlib.pyplot as plt

"""
Basic roulette wheel selection: O(N)
"""
def basic(fitness):
    # Input: a list of N fitness values (list or tuple)
    # Output: selected index

    sumFits = sum(fitness)
    # generate a random number
    rndPoint = random.uniform(0, sumFits)
    # calculate the index: O(N)
    accumulator = 0.0
    for ind, val in enumerate(fitness):
        accumulator += val
        if accumulator >= rndPoint:
            return ind


"""
Bisecting search roulette wheel selection: O(N + logN)
"""
def bisectSearch(fitness):
    # Input: a list of N fitness values (list or tuple)
    # Output: selected index
    sumFits = sum(fitness)
    # generate a random number
    rndPoint = random.uniform(0, sumFits)
    # calculate the accumulator: O(N)
    accumulator = []
    accsum = 0.0
    for fit in fitness:
        accsum += fit
        accumulator.append(accsum)
    return bisect_left(accumulator, rndPoint)   # O(logN)


"""
Stochastic Acceptance: O(1) if given the N and maxFit before
"""
def stochasticAccept(fitness):
    # Input: a list of N fitness values (list or tuple)
    # Output: selected index

    # calculate N and max fitness value
    N = len(fitness)
    maxFit = max(fitness)
    # select: O(1)
    while True:
        # randomly select an individual with uniform probability
        ind = int(N * random.random())
        # with probability wi/wmax to accept the selection
        if random.random() <= fitness[ind] / maxFit:
            return ind


"""
main function
"""
def main():
    # init number of fitness values
    N = [10, 10**2, 10**3, 10**4, 10**5]
    # calculate average total run time for each algorithm
    times = [[], [], []]
    algos = [basic, bisectSearch, stochasticAccept]
    for n in N:
        fitness = np.random.random((n,))
        for ind, algo in enumerate(algos):
            sample_times = []
            for _ in range(100):
                start = timeit.default_timer()
                algo(fitness)
                sample_times.append(timeit.default_timer() - start)
            times[ind].append(np.array(sample_times).mean())
    # plot the result
    lineStyle = ['b-o', 'g--p', 'r:*']
    algoName = ['basic', 'bisectSearch', 'stochasticAccept']
    for i in range(len(times)):
        plt.loglog(N, times[i], lineStyle[i], label=algoName[i])
    plt.legend(loc=2)
    plt.title('log-log plot of average running time')
    plt.xlabel('N')
    plt.ylabel('Average Running Time')
    plt.grid('on')
    plt.show()


if __name__ == "__main__":
    main()
'''
