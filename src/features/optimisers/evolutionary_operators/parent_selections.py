import random as rnd
import numpy as np
from abc import ABCMeta, abstractmethod


class ParentSelection(metaclass=ABCMeta):
    """
    The Strategy interface for parent selection operator implementations. The interface declares operations common to all supported parent selection versions.

    The bio-inspired optimizer uses this interface to call the algorithm defined by the concrete parent selection implementations.
    """

    @abstractmethod
    def select(self, population):
        pass


class RouletteWheel(ParentSelection):

    def __init__(self):
        self.name = "RWS"

    def select(self, population_fitness):
        fitness_sum = sum(item for item in population_fitness)
        random_pick = rnd.uniform(0, fitness_sum)

        current = 0
        index = 0

        for item in population_fitness:
            current += item

            if current >= random_pick:
                return index

            index += 1

        return


class StochasticUniversalSampling(ParentSelection):

    def __init__(self):
        self.name = "SUS"
        self.parents = []
        self.last_idx = 0

    def select(self, population_fitness):
        if not self.parents:
            self.parents = self.generate_parent_pool(population_fitness)

        if self.last_idx < len(population_fitness):
            self.last_idx += 1
            return self.parents[self.last_idx - 1]

        else:
            self.last_idx = 1
            self.parents = self.generate_parent_pool(population_fitness)
            return self.parents[self.last_idx - 1]

    def generate_parent_pool(self, population_fitness):
        fitness_sum = sum(item for item in population_fitness)
        population_size = len(population_fitness)
        point_distance = fitness_sum / population_size
        start_point = rnd.uniform(0, point_distance)
        points = [start_point + i * point_distance for i in range(population_size)]

        parents = []
        for point in points:
            fitness = 0
            for i in range(population_size):
                fitness += population_fitness[i]
                if fitness > point:
                    parents.append(i)
                    break

        rnd.shuffle(parents)
        return parents


class Tournament(ParentSelection):

    def __init__(self, k):
        self.name = "TS"
        self.k = k

    def select(self, population_fitness):
        individuals = rnd.choices(population_fitness, k=self.k)

        return np.argmax(individuals)


class Rank(ParentSelection):

    def __init__(self):
        self.name = "RS"

    def select(self, population_fitness):
        sorted_population_fitness = np.sort(population_fitness)[::-1]
        population_size = len(population_fitness)
        ranks = [population_size - list(sorted_population_fitness).index(candidate) for candidate in population_fitness]
        rank_sum = sum(item for item in ranks)
        random_pick = rnd.uniform(0, rank_sum)

        current = 0
        index = 0

        for item in ranks:
            current += item

            if current >= random_pick:
                return index

            index += 1

        return


class Random(ParentSelection):

    def __init__(self):
        self.name = "RandomS"

    def select(self, population_fitness):
        population_size = len(population_fitness)

        return rnd.randint(0, population_size-1)
