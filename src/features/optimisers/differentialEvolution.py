import random
import numpy as np
import copy
import time

from src.features.logging.searchLogger import SearchLogger, log_points
from src.features.optimisers.wrapper import Wrapper


class DifferentialEvolution(Wrapper):

    def __init__(self, population_size, max_nfes, crossover_rate, scale_factor, bound_handler, binarizer):
        super().__init__(population_size)
        self.max_nfes = max_nfes
        self.crossover_rate = crossover_rate
        self.scale_factor = scale_factor
        self.bound_handler = bound_handler
        self.binariser = binarizer
        self.name = "DE"

    def search(self, experiment_name, fitness_function, initial_population=None):
        """
        :param initial_population:
        :param experiment_name: used to determine the filename for logging results and scores during search
        :param fitness_function: fitness_function used to evaluate population solutions
        :return:
        """
        logger = SearchLogger(optimiser_name='_'.join([experiment_name, self.name]), binariser_name='',
                              problem_name=fitness_function.name)
        spent_nfes = 0
        wasted_nfes = 0
        start_time = time.time()
        archive_duration = 0
        interval = (self.bound_handler.upper_bound - self.bound_handler.lower_bound)
        population = np.random.rand(self.population_size,
                                    fitness_function.dimensionality) * interval + self.bound_handler.lower_bound
        binary_population = np.empty((self.population_size, fitness_function.dimensionality), bool)
        population_fitness = np.empty(self.population_size, float)
        population_fitness_test = np.empty(self.population_size, float)

        archive = np.empty([self.population_size, fitness_function.dimensionality], bool)
        archive_fitness = np.empty(self.population_size, float)

        features_frequencies = np.zeros(fitness_function.dimensionality, dtype=int)

        validation_metrics = test_metrics = population_scores = population_scores_test = None

        for idx, candidate in enumerate(population):
            binary_population[idx] = self.binariser.binarise(candidate)
            if not any(binary_population[idx]):
                population_fitness[idx] = 0.0
                population_fitness_test[idx] = 0.0
            else:
                population_fitness[idx] = fitness_function.evaluate_on_validation(binary_population[idx])
                population_fitness_test[idx] = fitness_function.evaluate_on_test(binary_population[idx])

                features_frequencies = features_frequencies + binary_population[idx]

                archive[idx] = copy.deepcopy(binary_population[idx])
                archive_fitness[idx] = population_fitness[idx]

            spent_nfes += 1

            logger.log(spent_nfes=spent_nfes, max_nfes=self.max_nfes, wasted_nfes=wasted_nfes,
                       population=copy.deepcopy(population),
                       fitness_metric=fitness_function.fitness,
                       population_fitness=copy.deepcopy(population_fitness),
                       population_fitness_test=copy.deepcopy(population_fitness_test),
                       metrics=validation_metrics,
                       population_scores=copy.deepcopy(population_scores),
                       population_scores_test=copy.deepcopy(population_scores_test),
                       feature_frequencies=copy.deepcopy(features_frequencies))

        trial_population = np.empty((self.population_size, fitness_function.dimensionality), float)
        binary_trial_population = np.empty((self.population_size, fitness_function.dimensionality), bool)
        trial_population_fitness = np.empty(self.population_size, float)

        while spent_nfes < self.max_nfes:
            for idx, candidate in enumerate(population):
                mutant = self._create_mutant_vectors(idx=idx, population=population)
                trial = self._create_trial_vector(mutant, candidate)

                trial_population[idx] = trial
                binary_trial_population[idx] = self.binariser.binarise(trial_population[idx], binary_population[idx])

            for idx, binaryTrialCandidate in enumerate(binary_trial_population):
                if not any(binaryTrialCandidate):
                    trial_population_fitness[idx] = 0.0
                else:
                    trial_population_fitness[idx] = fitness_function.evaluate_on_validation(binaryTrialCandidate)
                    features_frequencies = features_frequencies + binaryTrialCandidate

                if (archive_duration == 0 and not any(np.array_equal(archive_element, binary_trial_population[idx]) for archive_element in archive)) and (
                        trial_population_fitness[idx] > np.min(archive_fitness) or (
                        trial_population_fitness[idx] == np.min(archive_fitness) and np.sum(binary_trial_population[idx]) < np.sum(
                        archive[np.argmin(archive_fitness)]))):
                    worst_idx = np.argmin(archive_fitness)
                    archive[worst_idx] = copy.deepcopy(binary_trial_population[idx])
                    archive_fitness[worst_idx] = trial_population_fitness[idx]


                spent_nfes += 1

                if spent_nfes == self.max_nfes - fitness_function.dimensionality:
                    archive_duration = time.time() - start_time

                if spent_nfes / self.max_nfes in log_points:
                    population_fitness_test, validation_metrics, population_scores, test_metrics, population_scores_test = self._get_comprehensive_logging_data(
                        binary_population, fitness_function)

                if spent_nfes >= self.max_nfes:
                    break
                else:

                    logger.log(spent_nfes=spent_nfes, max_nfes=self.max_nfes, wasted_nfes=wasted_nfes,
                               population=copy.deepcopy(population),
                               fitness_metric=fitness_function.fitness,
                               population_fitness=copy.deepcopy(population_fitness),
                               population_fitness_test=copy.deepcopy(population_fitness_test),
                               metrics=validation_metrics,
                               population_scores=copy.deepcopy(population_scores),
                               population_scores_test=copy.deepcopy(population_scores_test),
                               feature_frequencies=copy.deepcopy(features_frequencies))


            # weak inequality

            should_swap = trial_population_fitness > population_fitness

            for idx, swap in enumerate(should_swap):
                if swap:
                    if np.array_equal(binary_population[idx], binary_trial_population[idx]):
                        wasted_nfes += 1

                    population[idx] = trial_population[idx]
                    population_fitness[idx] = trial_population_fitness[idx]
                    binary_population[idx] = binary_trial_population[idx]

            if spent_nfes >= self.max_nfes:
                population_fitness_test, validation_metrics, population_scores, test_metrics, population_scores_test = self._get_comprehensive_logging_data(
                    binary_population, fitness_function)

                logger.log(spent_nfes=spent_nfes, max_nfes=self.max_nfes, wasted_nfes=wasted_nfes,
                           population=copy.deepcopy(population),
                           fitness_metric=fitness_function.fitness,
                           population_fitness=copy.deepcopy(population_fitness),
                           population_fitness_test=copy.deepcopy(population_fitness_test),
                           metrics=validation_metrics,
                           population_scores=copy.deepcopy(population_scores),
                           population_scores_test=copy.deepcopy(population_scores_test),
                           feature_frequencies=copy.deepcopy(features_frequencies))

                best_idx = np.argmax(population_fitness)
                output_quality = fitness_function.evaluate_final_solution(binary_population[best_idx])
                wrapper_duration = time.time() - start_time
                logger.log_output(output_quality, archive=copy.deepcopy(archive),
                                  archive_fitness=copy.deepcopy(archive_fitness),
                                  population=copy.deepcopy(binary_population),
                                  population_fitness=copy.deepcopy(population_fitness),wrapper_duration=wrapper_duration, archive_duration=archive_duration)

                return binary_population[best_idx], population_fitness[best_idx]

    def _create_trial_vector(self, mutant, candidate):
        """
        Creates a trial vector by combining the original candidate and the mutant with respect to
        the crossover rate of the algorithm. Ensures that at least one element of the solution will
        be crossed over.
        :param mutant: Mutant vector
        :param candidate: Original solution for which the trial is created
        :return: trial vector
        """

        should_cross = np.random.rand(len(candidate)) <= self.crossover_rate

        # Ensure at least one element is crossed over:
        random_index = int((len(candidate)) * random.random())
        should_cross[random_index] = True

        trial = np.where(should_cross, mutant, candidate)
        self.bound_handler.handle_bounds(trial)
        return trial

    def _create_mutant_vectors(self, population, idx):
        """
        Creates a mutant solution combining three randomly selected solutions selected from the population.
        :param population: The original solutions from which to chose from
        :param idx: Candidate solution index in the population, should not be used for creating the mutant
        :return: Mutant solution
        """

        r1, r2, r3 = self._select_mutation_vectors(idx)
        return population[r1] + self.scale_factor * (population[r2] - population[r3])

    def _select_mutation_vectors(self, idx):
        """
        Selects three distinct indices from the range [0, populationSize) ensuring that they are mutually exclusive
        and different from the original idx.
        :param idx: index to avoid, needs to be in [0, populationSize)
        :return: indices r1,r2,r3 of solutions in the population from which the mutant will be created
        """

        r1 = int(self.population_size * random.random())
        while r1 == idx:
            r1 = int(self.population_size * random.random())

        r2 = int(self.population_size * random.random())
        while r2 == idx or r2 == r1:
            r2 = int(self.population_size * random.random())

        r3 = int(self.population_size * random.random())
        while r3 == idx or r3 == r2 or r3 == r1:
            r3 = int(self.population_size * random.random())

        return r1, r2, r3
