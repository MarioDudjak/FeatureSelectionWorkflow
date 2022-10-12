import numpy as np
import random
import copy

from src.features.logging.searchLogger import SearchLogger, log_points
from src.features.optimisers.wrapper import Wrapper


class GeneticAlgorithmAM(Wrapper):
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
        self.name = "GAAM" + "(pm=" + str(mutation_rate) + ")"

    def search(self, experiment_name, fitness_function, initial_population):
        """
        Main function inside GA wrapper. It generates population, applies selection and variation operators and
        evaluates candidate solutions.

        :param experiment_name:
        :param fitness_function:
        :return:
        """
        logger = SearchLogger('_'.join([experiment_name, self.name]), '', fitness_function.name)
        spent_nfes = 0
        wasted_nfes = 0

        #
        # Init random population, init other population variables
        #
        population = copy.deepcopy(initial_population)
        population_fitness = np.empty(self.population_size, float)
        population_fitness_test = np.empty(self.population_size, float)
        features_frequencies = np.zeros(fitness_function.dimensionality, dtype=int)
        validation_metrics = test_metrics = population_scores = population_scores_test = None

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
            spent_nfes += 1

            logger.log(spent_nfes=spent_nfes, max_nfes=self.max_nfes, population=population,
                       fitness_metric=fitness_function.fitness,
                       population_fitness=population_fitness,
                       population_fitness_test=population_fitness_test, metrics=validation_metrics,
                       population_scores=population_scores,
                       population_scores_test=population_scores_test, feature_frequencies=features_frequencies)

        new_population = np.empty((self.population_size, fitness_function.dimensionality), bool)
        new_population_fitness = np.empty(self.population_size, float)

        while spent_nfes < self.max_nfes:
            #
            # Generate new population, apply selection, crossover and mutation
            #
            for x in range(0, len(population), 2):
                parent_a = population[self._parent_selection(population_fitness)]
                parent_b = population[self._parent_selection(population_fitness)]

                while np.array_equal(parent_a, parent_b):
                    parent_b = population[self._parent_selection(population_fitness)]
                    print("equal parents")

                offsprings = self._crossover([parent_a, parent_b])
                offsprings = self._mutation(offsprings, features_frequencies, spent_nfes)

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

                spent_nfes += 1

                if spent_nfes / self.max_nfes in log_points:
                    population_fitness_test, validation_metrics, population_scores, test_metrics, population_scores_test = self._get_comprehensive_logging_data(
                        population, fitness_function)

                logger.log(spent_nfes=spent_nfes, max_nfes=self.max_nfes, population=population,
                           fitness_metric=fitness_function.fitness,
                           population_fitness=population_fitness,
                           population_fitness_test=population_fitness_test, metrics=validation_metrics,
                           population_scores=population_scores,
                           population_scores_test=population_scores_test, feature_frequencies=features_frequencies)

                if spent_nfes > self.max_nfes:
                    break

            #
            # Survival selection - replace current population
            #
            new_population, new_population_fitness = self._population_selection(population,
                                                                                population_fitness,
                                                                                new_population,
                                                                                new_population_fitness)
            wasted_nfes += sum(
                np.array([np.array_equal(population[i], solution) for i, solution in enumerate(new_population)]))
            population = copy.deepcopy(new_population)
            population_fitness = copy.deepcopy(new_population_fitness)

            #
            # When max nfes is reached, terminate and create final evaluation logs
            #
            if spent_nfes >= self.max_nfes:
                population_fitness_test, validation_metrics, population_scores, test_metrics, population_scores_test = self._get_comprehensive_logging_data(
                    population, fitness_function)

                logger.log(spent_nfes=spent_nfes, max_nfes=self.max_nfes, population=population,
                           fitness_metric=fitness_function.fitness,
                           population_fitness=population_fitness,
                           population_fitness_test=population_fitness_test, metrics=validation_metrics,
                           population_scores=population_scores,
                           population_scores_test=population_scores_test, feature_frequencies=features_frequencies)

                best_idx = np.argmax(population_fitness)
                output_quality, confusion_matrix = fitness_function.evaluate_final_solution(population[best_idx])
                logger.log_output(output_quality, confusion_matrix)

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
        population, population_fitness = self.population_selection_strategy.generate_population(copy.deepcopy(population),
                                                                                                copy.deepcopy(new_population),
                                                                                                copy.deepcopy(population_fitness),
                                                                                                copy.deepcopy(new_population_fitness))
        return population, population_fitness

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

    def _mutation(self, offsprings, feature_frequencies, spent_nfes):
        """
         Wrapper method for applying mutation method based on given options

        :param offsprings:
        :return:
        """
        feature_relative_frequencies = feature_frequencies / spent_nfes
        mutation_rates = self.mutation_rate - abs(feature_relative_frequencies - self.mutation_rate / 2) * 2
        mutated_offsprings = self.mutation_strategy.mutate(copy.deepcopy(offsprings), mutation_rates)
        return mutated_offsprings


