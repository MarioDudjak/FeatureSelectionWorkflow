import numpy as np

from abc import ABCMeta, abstractmethod


class Wrapper(metaclass=ABCMeta):
    """
    The Strategy interface for wrapper implementations. The interface declares operations common to all supported wrapper versions.

    The bio-inspired optimizer implements this interface.
    """
    def __init__(self, population_size):
        self.population_size = population_size

    @abstractmethod
    def search(self, experiment_name, fitness_function, initial_population):
        pass

    def _get_comprehensive_logging_data(self, population, fitness_function):
        population_fitness_test = self._evaluate_on_test(population, fitness_function)
        validation_metrics, population_scores = self._get_full_classification_scores(population,
                                                                                     fitness_function.evaluate_on_validation)
        test_metrics, population_scores_test = self._get_full_classification_scores(population,
                                                                                    fitness_function.evaluate_on_test)

        return population_fitness_test, validation_metrics, population_scores, test_metrics, population_scores_test

    def _evaluate_on_test(self, population, fitness_function):
        population_fitness_test = np.empty(self.population_size, float)
        for idx, candidate in enumerate(population):
            if not any(candidate):
                population_fitness_test[idx] = 0.0
            else:
                population_fitness_test[idx] = fitness_function.evaluate_on_test(candidate, comprehensive=False)

        return population_fitness_test

    def _get_full_classification_scores(self, population, fitness_function):
        population_scores = np.empty(self.population_size, dtype=dict)
        for idx, candidate in enumerate(population):
            if not any(candidate):
                population_scores[idx] = [0]
            else:
                population_scores[idx] = fitness_function(candidate, comprehensive=True)

        metrics = population_scores[0].keys()
        population_scores = [list(candidate_scores.values()) for candidate_scores in population_scores if
                             any(candidate_scores)]
        return metrics, population_scores

