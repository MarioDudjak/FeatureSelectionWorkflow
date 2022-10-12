import random as rnd

from abc import ABCMeta, abstractmethod


class Mutation(metaclass=ABCMeta):
    """
    The Strategy interface for mutation operator implementations. The interface declares operations common to all supported mutation versions.

    The bio-inspired optimizer uses this interface to call the algorithm defined by the concrete mutation implementations.
    """

    @abstractmethod
    def mutate(self, population, mutation_rate):
        pass


class BitFlipMutation(Mutation):

    def __init__(self):
        self.name = "BitFlipMutation"

    def mutate(self, population, mutation_rate):
        for offspring in population:
            for index in range(len(offspring)):
                random_number = rnd.uniform(0, 1)

                if random_number < mutation_rate:
                    offspring[index] = type(offspring[index])(not offspring[index])

        return population


class AdaptiveMutation(Mutation):

    def __init__(self):
        self.name = "AdaptiveMutation"

    def mutate(self, population, mutation_rates):
        for offspring in population:
            for index in range(len(offspring)):
                random_number = rnd.uniform(0, 1)

                if random_number < mutation_rates[index]:
                    offspring[index] = type(offspring[index])(not offspring[index])

        return population

