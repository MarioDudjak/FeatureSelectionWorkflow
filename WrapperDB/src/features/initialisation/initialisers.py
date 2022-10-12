import numpy as np
import random


class RandomComplexInitialiser:

    def __init__(self, bound_handler):
        self.bound_handler = bound_handler

    def create_population(self, population_size, dimensionality):

        interval = (self.bound_handler.upper_bound - self.bound_handler.lower_bound)
        population = np.random.rand(population_size, dimensionality) * interval + self.bound_handler.lower_bound

        middle_point = (self.bound_handler.upper_bound + self.bound_handler.lower_bound) / 2
        lower_interval = (middle_point - self.bound_handler.lower_bound)
        upper_interval = (self.bound_handler.upper_bound - middle_point)
        population = np.random.rand(population_size, dimensionality) * lower_interval + self.bound_handler.lower_bound

        for idx, candidate in enumerate(population):
            size = np.random.randint(1, dimensionality + 1)
            t = 0
            while t < size:
                random_features = np.random.randint(0, dimensionality, size)
                for feature in random_features:
                    candidate[feature] = np.random.rand() * upper_interval + middle_point
                    t += 1

        return population


class UniformFeatureCountInitialiser:

    def __init__(self, bound_handler):
        self.bound_handler = bound_handler

    def create_population(self, population_size, dimensionality):

        half_range = (self.bound_handler.upper_bound - self.bound_handler.lower_bound) / 2
        population = np.random.rand(population_size, dimensionality) * half_range + self.bound_handler.lower_bound

        for idx, candidate in enumerate(population):
            # Pick feature count at random:
            size = np.random.randint(1, dimensionality + 1)
            # Pick 'size' of unique features:
            random_features = random.sample(range(0, dimensionality), size)
            for feature in random_features:
                candidate[feature] = candidate[feature] + half_range

        return population


class BinaryUniformFeatureCountInitialiser:

    def __init__(self):
        self.name = "UniformFeatureCount"

    def create_population(self, population_size, dimensionality):
        population = np.random.choice([False],
                                      size=(population_size, dimensionality))

        for idx, candidate in enumerate(population):
            # Pick feature count at random:
            size = np.random.randint(1, dimensionality + 1)
            # Pick 'size' of unique features:
            random_features = random.sample(range(0, dimensionality), size)
            for feature in random_features:
                candidate[feature] = True

        return population


class BinaryRandomFeatureCountInitialiser:

    def __init__(self):
        self.name = "RandomFeatureCount"

    def create_population(self, population_size, dimensionality):
        population = np.random.choice([False, True],
                                      size=(population_size, dimensionality))
        return population