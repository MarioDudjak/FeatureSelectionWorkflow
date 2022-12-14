import numpy as np
import random
import copy
import time
from src.features.logging.searchLogger import SearchLogger, log_points
from src.features.optimisers.wrapper import Wrapper


class GeneticAlgorithm(Wrapper):
    def __init__(self, population_size, max_nfes, crossover_rate, mutation_rate, crossover_operator, mutation_operator,
                 parent_selection_operator, population_selection_operator):
        super().__init__(population_size)
        self.max_nfes = max_nfes
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.crossover_strategy = crossover_operator
        self.mutation_strategy = mutation_operator
        self.parent_selection_strategy = parent_selection_operator
        self.population_selection_strategy = population_selection_operator
        self.name = "GAMu"
        self.mutation_rate = mutation_rate

    def search(self, experiment_name, fitness_function, initial_population=None):
        """
        Main function inside GA wrapper. It generates population, applies selection and variation operators and
        evaluates candidate solutions.

        :param experiment_name:
        :param fitness_function:
        :param initial_population:
        :return:
        """
        logger = SearchLogger('_'.join([experiment_name, self.name]), '', fitness_function.name)
        spent_nfes = 0
        wasted_nfes = 0
        #
        # Init random population, init other population variables
        #
        start_time = time.time()
        archive_duration = 0
        population = np.random.choice([False, True],
                                      size=(self.population_size, fitness_function.dimensionality))
        population_fitness = np.empty(self.population_size, float)
        population_fitness_test = np.empty(self.population_size, float)

        archive = np.empty([self.population_size, fitness_function.dimensionality], bool)
        archive_fitness = np.empty(self.population_size, float)

        features_frequencies = np.zeros(fitness_function.dimensionality, dtype=int)

        validation_metrics = population_scores = population_scores_test = None

        #
        # Evaluate candidate solutions from initial generation
        #
        for index, candidate in enumerate(population):
            if not any(candidate):
                population_fitness[index] = 0.0
                population_fitness_test[index] = 0.0
            else:
                population_fitness[index] = fitness_function.evaluate_on_validation(population[index],
                                                                                    comprehensive=False)
                population_fitness_test[index] = fitness_function.evaluate_on_test(population[index],
                                                                                   comprehensive=False)
                features_frequencies = features_frequencies + candidate

                archive[index] = copy.deepcopy(population[index])
                archive_fitness[index] = population_fitness[index]

            spent_nfes += 1

            if spent_nfes / self.max_nfes in log_points:
                population_fitness_test, validation_metrics, population_scores, test_metrics, population_scores_test = self._get_comprehensive_logging_data(
                    population, fitness_function)

            logger.log(spent_nfes=spent_nfes, max_nfes=self.max_nfes, wasted_nfes=wasted_nfes,
                       population=copy.deepcopy(population),
                       fitness_metric=fitness_function.fitness,
                       population_fitness=copy.deepcopy(population_fitness),
                       population_fitness_test=copy.deepcopy(population_fitness_test),
                       metrics=validation_metrics,
                       population_scores=copy.deepcopy(population_scores),
                       population_scores_test=copy.deepcopy(population_scores_test),
                       feature_frequencies=copy.deepcopy(features_frequencies))

        new_population = np.empty((self.population_size, fitness_function.dimensionality), bool)
        new_population_fitness = np.empty(self.population_size, float)

        while spent_nfes < self.max_nfes:
            #
            # Generate new population, apply selection, crossover and mutation
            #
            for x in range(0, len(population), 2):
                parent_a = copy.deepcopy(population[self._parent_selection(population_fitness)])
                parent_b = copy.deepcopy(population[self._parent_selection(population_fitness)])

                # while np.array_equal(parent_a, parent_b):
                #     parent_b = population[self._parent_selection(population_fitness)]

                offsprings = copy.deepcopy(self._crossover([parent_a, parent_b]))
                offsprings = copy.deepcopy(self._mutation(offsprings))

                new_population[x] = offsprings[0]
                new_population[x + 1] = offsprings[1]

            #
            # Generate fitness from new population
            #
            for index, candidate in enumerate(new_population):
                if not any(candidate):
                    new_population_fitness[index] = 0.0
                else:
                    new_population_fitness[index] = fitness_function.evaluate_on_validation(new_population[index],
                                                                                            comprehensive=False)
                    features_frequencies = features_frequencies + candidate

                if (archive_duration == 0 and not any(np.array_equal(archive_element, candidate) for archive_element in
                                                      archive)) and (
                        new_population_fitness[index] > np.min(archive_fitness) or (
                        new_population_fitness[index] == np.min(archive_fitness) and np.sum(
                    candidate) < np.sum(
                    archive[np.argmin(archive_fitness)]))):
                    worst_idx = np.argmin(archive_fitness)
                    archive[worst_idx] = copy.deepcopy(candidate)
                    archive_fitness[worst_idx] = new_population_fitness[index]

                spent_nfes += 1

                if spent_nfes == self.max_nfes - fitness_function.dimensionality:
                    archive_duration = time.time() - start_time

                if spent_nfes / self.max_nfes in log_points:
                    population_fitness_test, validation_metrics, population_scores, test_metrics, population_scores_test = self._get_comprehensive_logging_data(
                        population, fitness_function)

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

            wasted_nfes += sum(
                np.array([np.array_equal(population[i], solution) for i, solution in enumerate(new_population)]))
            #
            # Survivor selection - replace current population
            #
            generated_population, generated_population_fitness = self._population_selection(population,
                                                                                            population_fitness,
                                                                                            new_population,
                                                                                            new_population_fitness)

            population = copy.deepcopy(generated_population)
            population_fitness = copy.deepcopy(generated_population_fitness)

            #
            # When max nfes is reached, terminate and create final evaluation logs
            #
            if spent_nfes >= self.max_nfes:
                population_fitness_test, validation_metrics, population_scores, test_metrics, population_scores_test = self._get_comprehensive_logging_data(
                    population, fitness_function)

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
                output_quality = fitness_function.evaluate_final_solution(population[best_idx])
                wrapper_duration = time.time() - start_time
                logger.log_output(output_quality, archive=copy.deepcopy(archive),
                                  archive_fitness=copy.deepcopy(archive_fitness),
                                  population=copy.deepcopy(population),
                                  population_fitness=copy.deepcopy(population_fitness),
                                  wrapper_duration=wrapper_duration, archive_duration=archive_duration)

                return population[best_idx], population_fitness[best_idx]

    def _parent_selection(self, population_fitness):
        """
        Wrapper method for applying parent selection method based on given options. Returns solution
        index from given population

        :param population_fitness:
        :return:
        """
        candidate_id = self.parent_selection_strategy.select(population_fitness=copy.deepcopy(population_fitness))
        return candidate_id

    def _population_selection(self, population, population_fitness, new_population, new_population_fitness):
        """
         Wrapper method for applying population selection method based on given options

        :param population:
        :param population_fitness:
        :param new_population:
        :param new_population_fitness:
        :return:
        """
        generated_population, generated_population_fitness = self.population_selection_strategy.generate_population(
            copy.deepcopy(population),
            copy.deepcopy(new_population),
            copy.deepcopy(population_fitness),
            copy.deepcopy(new_population_fitness))
        return generated_population, generated_population_fitness

    def _crossover(self, parents):
        """
         Wrapper method for applying crossover method based on given options

        :param parents:
        :return:
        """

        if self.crossover_rate < random.random():
            offsprings = parents

        else:
            offsprings = self.crossover_strategy.mate(copy.deepcopy(parents))

        return offsprings

    def _mutation(self, offsprings):
        """
         Wrapper method for applying mutation method based on given options

        :param offsprings:
        :return:
        """
        mutated_offsprings = self.mutation_strategy.mutate(copy.deepcopy(offsprings), self.mutation_rate)
        return mutated_offsprings


class EGAFS(Wrapper):
    """
    2018 - An improved genetic algorithm for feature selection in the classification of Disaster-related Twitter messages
    https://ieeexplore.ieee.org/abstract/document/8405477
    """

    def __init__(self, population_size, max_nfes, crossover_rate, mutation_rate, balancing_factor, crossover_operator,
                 mutation_operator,
                 parent_selection_operator, population_selection_operator):
        super().__init__(population_size)
        self.max_nfes = max_nfes
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.crossover_strategy = crossover_operator
        self.mutation_strategy = mutation_operator
        self.parent_selection_strategy = parent_selection_operator
        self.population_selection_strategy = population_selection_operator
        self.name = "EGAFS"
        self.mutation_rate = mutation_rate
        self.balancing_factor = balancing_factor

    def search(self, experiment_name, fitness_function, initial_population=None):
        """
        Main function inside GA wrapper. It generates population, applies selection and variation operators and
        evaluates candidate solutions.

        :param experiment_name:
        :param fitness_function:
        :param initial_population:
        :return:
        """
        fitness_function.fitness = 'accuracy'
        logger = SearchLogger('_'.join([experiment_name, self.name]), '', fitness_function.name)
        spent_nfes = 0
        wasted_nfes = 0
        #
        # Init random population, init other population variables
        #
        population = np.random.choice([False, True],
                                      size=(self.population_size, fitness_function.dimensionality))
        population_fitness = np.empty(self.population_size, float)
        population_fitness_test = np.empty(self.population_size, float)

        archive = np.empty([self.max_nfes, fitness_function.dimensionality], bool)
        archive_fitness = np.empty(self.max_nfes, float)

        features_frequencies = np.zeros(fitness_function.dimensionality, dtype=int)

        validation_metrics = population_scores = population_scores_test = None

        #
        # Evaluate candidate solutions from initial generation
        #
        for index, candidate in enumerate(population):
            if not any(candidate):
                population_fitness[index] = 0.0
                population_fitness_test[index] = 0.0
            else:
                population_fitness[index] = self.balancing_factor * fitness_function.evaluate_on_validation(
                    population[index],
                    comprehensive=False) + (1 - self.balancing_factor) * (1 / np.sum(population[index]))
                population_fitness_test[index] = fitness_function.evaluate_on_test(population[index],
                                                                                   comprehensive=False)
                features_frequencies = features_frequencies + candidate

                archive[spent_nfes] = copy.deepcopy(population[index])
                archive_fitness[spent_nfes] = population_fitness[index]

            spent_nfes += 1

            if spent_nfes / self.max_nfes in log_points:
                population_fitness_test, validation_metrics, population_scores, test_metrics, population_scores_test = self._get_comprehensive_logging_data(
                    population, fitness_function)

            logger.log(spent_nfes=spent_nfes, max_nfes=self.max_nfes, wasted_nfes=wasted_nfes,
                       population=copy.deepcopy(population),
                       fitness_metric=fitness_function.fitness,
                       population_fitness=copy.deepcopy(population_fitness),
                       population_fitness_test=copy.deepcopy(population_fitness_test),
                       metrics=validation_metrics,
                       population_scores=copy.deepcopy(population_scores),
                       population_scores_test=copy.deepcopy(population_scores_test),
                       feature_frequencies=copy.deepcopy(features_frequencies))

        new_population = np.empty((self.population_size, fitness_function.dimensionality), bool)
        new_population_fitness = np.empty(self.population_size, float)

        while spent_nfes < self.max_nfes:
            #
            # Generate new population, apply selection, crossover and mutation
            #
            for x in range(0, len(population), 2):
                parent_a = copy.deepcopy(population[self._parent_selection(population_fitness)])
                parent_b = copy.deepcopy(population[self._parent_selection(population_fitness)])

                # while np.array_equal(parent_a, parent_b):
                #     parent_b = population[self._parent_selection(population_fitness)]

                offsprings = copy.deepcopy(self._crossover([parent_a, parent_b]))
                offsprings = copy.deepcopy(self._mutation(offsprings))

                new_population[x] = offsprings[0]
                new_population[x + 1] = offsprings[1]

            #
            # Generate fitness from new population
            #
            for index, candidate in enumerate(new_population):
                if not any(candidate):
                    new_population_fitness[index] = 0.0
                else:
                    new_population_fitness[index] = self.balancing_factor * fitness_function.evaluate_on_validation(
                        new_population[index],
                        comprehensive=False) + (1 - self.balancing_factor) * (1 / np.sum(new_population[index]))
                    features_frequencies = features_frequencies + candidate

                archive[spent_nfes] = copy.deepcopy(new_population[index])
                archive_fitness[spent_nfes] = new_population_fitness[index]
                spent_nfes += 1

                if spent_nfes / self.max_nfes in log_points:
                    population_fitness_test, validation_metrics, population_scores, test_metrics, population_scores_test = self._get_comprehensive_logging_data(
                        population, fitness_function)

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

            wasted_nfes += sum(
                np.array([np.array_equal(population[i], solution) for i, solution in enumerate(new_population)]))
            #
            # Survivor selection - replace current population
            #
            generated_population, generated_population_fitness = self._population_selection(population,
                                                                                            population_fitness,
                                                                                            new_population,
                                                                                            new_population_fitness)

            # for idx, candidate in enumerate(generated_population):
            #     if not any(np.array_equal(archive_element, candidate) for archive_element in archive) and \
            #             generated_population_fitness[idx] > np.min(archive_fitness):
            #         worst_idx = np.argmin(archive_fitness)
            #         archive[worst_idx] = copy.deepcopy(candidate)
            #         archive_fitness[worst_idx] = generated_population_fitness[idx]

            population = copy.deepcopy(generated_population)
            population_fitness = copy.deepcopy(generated_population_fitness)

            #
            # When max nfes is reached, terminate and create final evaluation logs
            #
            if spent_nfes >= self.max_nfes:
                population_fitness_test, validation_metrics, population_scores, test_metrics, population_scores_test = self._get_comprehensive_logging_data(
                    population, fitness_function)

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
                output_quality = fitness_function.evaluate_final_solution(population[best_idx])
                logger.log_output(output_quality, archive=copy.deepcopy(archive),
                                  archive_fitness=copy.deepcopy(archive_fitness),
                                  population=copy.deepcopy(population),
                                  population_fitness=copy.deepcopy(population_fitness))

                return population[best_idx], population_fitness[best_idx]

    def _parent_selection(self, population_fitness):
        """
        Wrapper method for applying parent selection method based on given options. Returns solution
        index from given population

        :param population_fitness:
        :return:
        """
        candidate_id = self.parent_selection_strategy.select(population_fitness=copy.deepcopy(population_fitness))
        return candidate_id

    def _population_selection(self, population, population_fitness, new_population, new_population_fitness):
        """
         Wrapper method for applying population selection method based on given options

        :param population:
        :param population_fitness:
        :param new_population:
        :param new_population_fitness:
        :return:
        """
        generated_population, generated_population_fitness = self.population_selection_strategy.generate_population(
            copy.deepcopy(population),
            copy.deepcopy(new_population),
            copy.deepcopy(population_fitness),
            copy.deepcopy(new_population_fitness))
        return generated_population, generated_population_fitness

    def _crossover(self, parents):
        """
         Wrapper method for applying crossover method based on given options

        :param parents:
        :return:
        """

        if self.crossover_rate < random.random():
            offsprings = parents

        else:
            offsprings = self.crossover_strategy.mate(copy.deepcopy(parents))

        return offsprings

    def _mutation(self, offsprings):
        """
         Wrapper method for applying mutation method based on given options

        :param offsprings:
        :return:
        """
        mutated_offsprings = self.mutation_strategy.mutate(copy.deepcopy(offsprings), self.mutation_rate)
        return mutated_offsprings
