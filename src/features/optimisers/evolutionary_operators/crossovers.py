import random as rnd
import numpy as np
import copy
from abc import ABCMeta, abstractmethod


class Crossover(metaclass=ABCMeta):
    """
    The Strategy interface for crossover operator implementations. The interface declares operations common to all supported crossover versions.

    The bio-inspired optimizer uses this interface to call the algorithm defined by the concrete crossover implementations.
    """

    @abstractmethod
    def mate(self, parents):
        pass


class SinglePointCrossover(Crossover):

    def __init__(self):
        self.name = "SinglePointCrossover"

    def mate(self, parents):
        parent_a, parent_b = parents

        crossover_point = rnd.randrange(1, len(parent_a) - 1)

        child_a = np.append(parent_a[:crossover_point], parent_b[crossover_point:])
        child_b = np.append(parent_b[:crossover_point], parent_a[crossover_point:])

        return [child_a, child_b]

class VariableSlicedCrossover(Crossover):

    def __init__(self):
        self.name = "VariableSlicedCrossover"

    def mate(self, parents):
        parent_a, parent_b = parents
        dimensionality = len(parent_a)
        child_a = np.zeros(dimensionality, dtype=bool)
        child_b = np.zeros(dimensionality, dtype=bool)
        if dimensionality < 30:
            random_point = 2
        else:
            random_point = np.random.randint(low=2, high=dimensionality//15 +1)

        feature_length = dimensionality // random_point
        feature_subset_ctr = 1
        beggining = 0
        end = feature_length + dimensionality % random_point
        while feature_subset_ctr <= random_point:
            p1_subset_score = np.sum(parent_a[beggining:end])
            p2_subset_score = np.sum(parent_b[beggining:end])

            if p1_subset_score >= p2_subset_score:
                if np.random.rand() < 0.5:
                    child_a[beggining:end] = copy.deepcopy(parent_a[beggining:end])
                    child_b[beggining:end] = copy.deepcopy(parent_b[beggining:end])
                else:
                    child_a[beggining:end] = copy.deepcopy(parent_b[beggining:end])
                    child_b[beggining:end] = copy.deepcopy(parent_a[beggining:end])
            else:
                child_a[beggining:end] = copy.deepcopy(parent_b[beggining:end])
                child_b[beggining:end] = copy.deepcopy(parent_a[beggining:end])

            beggining = end
            end = end + feature_length
            feature_subset_ctr += 1

        return [child_a, child_b]
