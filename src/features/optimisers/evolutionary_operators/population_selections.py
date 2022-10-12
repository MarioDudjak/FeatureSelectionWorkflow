import numpy as np

from abc import ABCMeta, abstractmethod


class PopulationSelection(metaclass=ABCMeta):
    """
    The Strategy interface for population selection operator implementations. The interface declares operations common to all supported population selection versions.

    The bio-inspired optimizer uses this interface to call the algorithm defined by the concrete population selection implementations.
    """

    @abstractmethod
    def generate_population(self, population, new_population, population_fitness, new_population_fitness):
        pass


class MuPlusLambda(PopulationSelection):

    def __init__(self):
        self.name = "MuPlusLambda"

    def generate_population(self, population, new_population, population_fitness, new_population_fitness):
        joined_population, joined_fitness = np.concatenate([population, new_population]), \
                                            np.concatenate([population_fitness, new_population_fitness])
        joined_tuples = np.array(list(zip(joined_population, joined_fitness)), dtype=object)

        temp_sorted = joined_tuples[joined_tuples[:, 1].argsort()][::-1]

        generated_population, generated_population_fitness = np.vstack(temp_sorted[:, 0]), np.stack(temp_sorted[:, 1])

        generated_population, generated_population_fitness = generated_population[
                                                             0:int(np.size(generated_population, 0) / 2),
                                                             :], generated_population_fitness[0:int(
            np.size(generated_population_fitness) / 2)]

        return generated_population, generated_population_fitness


class Generational(PopulationSelection):

    def __init__(self):
        self.name = "Generational"

    def generate_population(self, population, new_population, population_fitness, new_population_fitness):
        return new_population, new_population_fitness
