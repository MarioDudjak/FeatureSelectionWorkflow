import random
import numpy as np
import copy
import math

from src.features.logging.searchLogger import SearchLogger, log_points
from src.features.optimisers.wrapper import Wrapper


class ParticleSwarmOptimisation(Wrapper):

    def __init__(self, population_size, max_nfes, c1, c2, inertia, bound_handler, binarizer):
        super().__init__(population_size)
        self.max_nfes = max_nfes
        self.c1 = c1
        self.c2 = c2
        self.inertia = inertia
        self.bound_handler = bound_handler
        self.min_bound = bound_handler.lower_bound
        self.max_bound = bound_handler.upper_bound
        self.binariser = binarizer
        self.name = "PSO"

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

        particle_positions = np.empty((self.population_size, fitness_function.dimensionality), float)
        particle_fitness = np.zeros(self.population_size, float)
        particle_fitness_test = np.empty(self.population_size, float)
        particle_sizes = np.empty(self.population_size)
        particle_velocities = np.zeros((self.population_size, fitness_function.dimensionality), float)

        particle_personal_best = np.empty((self.population_size, fitness_function.dimensionality), float)
        particle_personal_best_fitness = np.zeros(self.population_size)
        particle_personal_best_fitness_test = np.zeros(self.population_size)
        particle_personal_best_sizes = np.empty(self.population_size)

        particle_global_best = np.empty(fitness_function.dimensionality)
        particle_global_best_fitness = -1
        particle_global_best_size = -1

        binary_population = np.empty((self.population_size, fitness_function.dimensionality), bool)
        binary_trial_population = np.empty((self.population_size, fitness_function.dimensionality), bool)

        archive = np.empty([self.max_nfes, fitness_function.dimensionality], bool)
        archive_fitness = np.empty(self.max_nfes, float)

        features_frequencies = np.zeros(fitness_function.dimensionality, dtype=int)

        validation_metrics = test_metrics = population_scores = population_scores_test = None

        for i in range(self.population_size):
            particle_positions[i] = self.create_random_particle(fitness_function.dimensionality)
            binary_trial_population[i] = self.binariser.binarise(particle_positions[i])
            binary_population[i] = copy.deepcopy(binary_trial_population[i])
            if not any(binary_trial_population[i]):
                particle_fitness[i] = 0.0
                particle_fitness_test[i] = 0.0
            else:
                particle_fitness[i] = fitness_function.evaluate_on_validation(binary_trial_population[i])
                particle_fitness_test[i] = fitness_function.evaluate_on_test(binary_trial_population[i])

            particle_sizes[i] = np.sum(binary_population[i])
            particle_personal_best[i] = copy.deepcopy(particle_positions[i])
            particle_personal_best_fitness[i] = particle_fitness[i]
            particle_personal_best_fitness_test[i] = particle_fitness_test[i]
            particle_personal_best_sizes[i] = particle_sizes[i]

            if particle_personal_best_fitness[i] > particle_global_best_fitness or particle_global_best_fitness == -1:
                particle_global_best = copy.deepcopy(particle_personal_best[i])
                particle_global_best_fitness = particle_personal_best_fitness[i]
                particle_global_best_size = particle_personal_best_sizes[i]

            features_frequencies = features_frequencies + binary_population[i]

            archive[spent_nfes] = copy.deepcopy(binary_population[i])
            archive_fitness[spent_nfes] = particle_fitness[i]

            spent_nfes += 1

            logger.log(spent_nfes=spent_nfes, max_nfes=self.max_nfes, wasted_nfes=wasted_nfes,
                       population=copy.deepcopy(particle_personal_best),
                       fitness_metric=fitness_function.fitness,
                       population_fitness=copy.deepcopy(particle_personal_best_fitness),
                       population_fitness_test=copy.deepcopy(particle_personal_best_fitness_test),
                       metrics=validation_metrics,
                       population_scores=copy.deepcopy(population_scores),
                       population_scores_test=copy.deepcopy(population_scores_test),
                       feature_frequencies=copy.deepcopy(features_frequencies))

        while spent_nfes < self.max_nfes:
            for i in range(self.population_size):
                particle_velocities[i] = self.calculate_velocity(i, particle_velocities, particle_positions,
                                                                 particle_personal_best, particle_global_best)
                particle_positions[i] = self.update_particle(i, particle_positions, particle_velocities)
                binary_trial_population[i] = self.binariser.binarise(particle_positions[i])

                if not any(binary_trial_population[i]):
                    particle_fitness[i] = 0.0
                    particle_fitness_test[i] = 0.0
                else:
                    particle_fitness[i] = fitness_function.evaluate_on_validation(binary_trial_population[i])
                    # particle_fitness_test[i] = fitness_function.evaluate_on_test(binary_population[i])

                particle_sizes[i] = np.sum(binary_trial_population[i])
                features_frequencies = features_frequencies + binary_trial_population[i]

                archive[spent_nfes] = copy.deepcopy(binary_trial_population[i])
                archive_fitness[spent_nfes] = particle_fitness[i]

                if particle_fitness[i] > particle_personal_best_fitness[i]:
                    binary_population[i] = copy.deepcopy(binary_trial_population[i])
                    particle_personal_best[i] = copy.deepcopy(particle_positions[i])
                    particle_personal_best_fitness[i] = particle_fitness[i]
                    particle_personal_best_sizes[i] = particle_sizes[i]
                    particle_personal_best_fitness_test[i] = fitness_function.evaluate_on_test(binary_population[i])


                spent_nfes += 1

                if spent_nfes / self.max_nfes in log_points:
                    population_fitness_test, validation_metrics, population_scores, test_metrics, population_scores_test = self._get_comprehensive_logging_data(
                        binary_population, fitness_function)

                if spent_nfes >= self.max_nfes:
                    break
                else:
                    logger.log(spent_nfes=spent_nfes, max_nfes=self.max_nfes, wasted_nfes=wasted_nfes,
                               population=copy.deepcopy(particle_personal_best),
                               fitness_metric=fitness_function.fitness,
                               population_fitness=copy.deepcopy(particle_personal_best_fitness),
                               population_fitness_test=copy.deepcopy(particle_personal_best_fitness_test),
                               metrics=validation_metrics,
                               population_scores=copy.deepcopy(population_scores),
                               population_scores_test=copy.deepcopy(population_scores_test),
                               feature_frequencies=copy.deepcopy(features_frequencies))

            particle_global_best, particle_global_best_fitness, particle_global_best_size = self.update_global_best(
                particle_global_best, particle_global_best_fitness, particle_global_best_size,
                particle_personal_best_fitness, particle_personal_best, particle_personal_best_sizes)

            if spent_nfes >= self.max_nfes:
                population_fitness_test, validation_metrics, population_scores, test_metrics, population_scores_test = self._get_comprehensive_logging_data(
                    binary_population, fitness_function)

                logger.log(spent_nfes=spent_nfes, max_nfes=self.max_nfes, wasted_nfes=wasted_nfes,
                           population=copy.deepcopy(particle_personal_best),
                           fitness_metric=fitness_function.fitness,
                           population_fitness=copy.deepcopy(particle_personal_best_fitness),
                           population_fitness_test=copy.deepcopy(particle_personal_best_fitness_test),
                           metrics=validation_metrics,
                           population_scores=copy.deepcopy(population_scores),
                           population_scores_test=copy.deepcopy(population_scores_test),
                           feature_frequencies=copy.deepcopy(features_frequencies))

        output_quality = fitness_function.evaluate_final_solution(self.binariser.binarise(particle_global_best))
        logger.log_output(output_quality, archive=copy.deepcopy(archive),
                          archive_fitness=copy.deepcopy(archive_fitness),
                          population=copy.deepcopy(binary_population),
                          population_fitness=copy.deepcopy(particle_fitness))

        return self.binariser.binarise(particle_global_best), particle_global_best_fitness

    def calculate_velocity(self, idx, particle_velocities, particle_positions, particle_personal_best,
                           particle_global_best):
        velocity = np.zeros(len(particle_global_best))
        maxVelocity = 0.1
        for j in range(len(particle_global_best)):
            velocity[j] = self.inertia * particle_velocities[idx][j] + self.c1 * np.random.rand() * (
                    particle_personal_best[idx][j] - particle_positions[idx][j]) + self.c2 * np.random.rand() * (
                                  particle_global_best[j] - particle_positions[idx][j])

            if velocity[j] > maxVelocity:
                velocity[j] = maxVelocity
            if velocity[j] < maxVelocity * (-1):
                velocity[j] = maxVelocity * (-1)

        return velocity

    def update_particle(self, idx, particle_positions, particle_velocities):
        for j in range(len(particle_positions[0])):
            particle_positions[idx][j] += particle_velocities[idx][j]

            if particle_positions[idx][j] > self.max_bound:
                particle_positions[idx][j] = self.max_bound

            if particle_positions[idx][j] < self.min_bound:
                particle_positions[idx][j] = self.min_bound

        return particle_positions[idx]

    def update_global_best(self, global_best, global_best_fitness, global_best_size, particle_personal_best_fitness,
                           particle_personal_best, particle_personal_best_sizes):
        personal_best_max = np.max(particle_personal_best_fitness)
        if personal_best_max > global_best_fitness:
            personal_best_max_idx = np.argmax(particle_personal_best_fitness)
            global_best = copy.deepcopy(particle_personal_best[personal_best_max_idx])
            global_best_fitness = particle_personal_best_fitness[personal_best_max_idx]
            global_best_size = particle_personal_best_sizes[personal_best_max_idx]

        return global_best, global_best_fitness, global_best_size

    def create_random_particle(self, dimensionality):
        random_particle = np.empty(dimensionality)
        for j in range(dimensionality):
            random_particle[j] = self.min_bound + np.random.rand() * (self.max_bound - self.min_bound)

        return random_particle


class ParticleSwarmOptimisationIniPG(Wrapper):

    def __init__(self, population_size, max_nfes, c1, c2, inertia, bound_handler, binarizer):
        super().__init__(population_size)
        self.max_nfes = max_nfes
        self.c1 = c1
        self.c2 = c2
        self.inertia = inertia
        self.bound_handler = bound_handler
        self.min_bound = bound_handler.lower_bound
        self.max_bound = bound_handler.upper_bound
        self.binariser = binarizer
        self.binariser.alpha = 0.6
        self.name = "PSOIniPG"
        self.small_size = 0.1
        self.small_amount = 2 / 3

    def search(self, experiment_name, fitness_function, initial_population=None):
        """
        :param initial_population:
        :param experiment_name: used to determine the filename for logging results and scores during search
        :param fitness_function: fitness_function used to evaluate population solutions
        :return:
        """
        fitness_function.fitness = 'accuracy'
        logger = SearchLogger(optimiser_name='_'.join([experiment_name, self.name]), binariser_name='',
                              problem_name=fitness_function.name)
        spent_nfes = 0
        wasted_nfes = 0

        particle_positions = np.empty((self.population_size, fitness_function.dimensionality), float)
        particle_fitness = np.zeros(self.population_size, float)
        particle_fitness_test = np.empty(self.population_size, float)
        particle_sizes = np.empty(self.population_size)
        particle_velocities = np.zeros((self.population_size, fitness_function.dimensionality), float)

        particle_personal_best = np.empty((self.population_size, fitness_function.dimensionality), float)
        particle_personal_best_fitness = np.zeros(self.population_size)
        particle_personal_best_fitness_test = np.zeros(self.population_size)
        particle_personal_best_sizes = np.empty(self.population_size)

        particle_global_best = np.empty(fitness_function.dimensionality)
        particle_global_best_fitness = -1
        particle_global_best_size = -1

        binary_population = np.empty((self.population_size, fitness_function.dimensionality), bool)
        binary_trial_population = np.empty((self.population_size, fitness_function.dimensionality), bool)

        archive = np.empty([self.max_nfes, fitness_function.dimensionality], bool)
        archive_fitness = np.empty(self.max_nfes, float)

        features_frequencies = np.zeros(fitness_function.dimensionality, dtype=int)

        validation_metrics = test_metrics = population_scores = population_scores_test = None

        particle_positions = self.initialise(fitness_function, self.population_size, self.small_size, self.small_amount)
        for i in range(self.population_size):
            # particle_positions[i] = self.create_random_particle(fitness_function.dimensionality)
            binary_trial_population[i] = self.binariser.binarise(particle_positions[i])
            binary_population[i] = copy.deepcopy(binary_trial_population[i])
            if not any(binary_trial_population[i]):
                particle_fitness[i] = 0.0
                particle_fitness_test[i] = 0.0
            else:
                particle_fitness[i] = fitness_function.evaluate_on_validation(binary_trial_population[i])
                particle_fitness_test[i] = fitness_function.evaluate_on_test(binary_trial_population[i])

            particle_sizes[i] = np.sum(binary_population[i])
            particle_personal_best[i] = copy.deepcopy(particle_positions[i])
            particle_personal_best_fitness[i] = particle_fitness[i]
            particle_personal_best_fitness_test[i] = particle_fitness_test[i]
            particle_personal_best_sizes[i] = particle_sizes[i]

            if particle_personal_best_fitness[i] > particle_global_best_fitness or particle_global_best_fitness == -1:
                particle_global_best = copy.deepcopy(particle_personal_best[i])
                particle_global_best_fitness = particle_personal_best_fitness[i]
                particle_global_best_size = particle_personal_best_sizes[i]

            elif particle_personal_best_fitness[i] == particle_global_best_fitness and particle_personal_best_sizes[
                i] < particle_global_best_size:
                particle_global_best = copy.deepcopy(particle_personal_best[i])
                particle_global_best_fitness = particle_personal_best_fitness[i]
                particle_global_best_size = particle_personal_best_sizes[i]

            features_frequencies = features_frequencies + binary_population[i]

            archive[spent_nfes] = copy.deepcopy(binary_population[i])
            archive_fitness[spent_nfes] = particle_fitness[i]

            spent_nfes += 1

            logger.log(spent_nfes=spent_nfes, max_nfes=self.max_nfes, wasted_nfes=wasted_nfes,
                       population=copy.deepcopy(particle_personal_best),
                       fitness_metric=fitness_function.fitness,
                       population_fitness=copy.deepcopy(particle_personal_best_fitness),
                       population_fitness_test=copy.deepcopy(particle_personal_best_fitness_test),
                       metrics=validation_metrics,
                       population_scores=copy.deepcopy(population_scores),
                       population_scores_test=copy.deepcopy(population_scores_test),
                       feature_frequencies=copy.deepcopy(features_frequencies))

        while spent_nfes < self.max_nfes:
            for i in range(self.population_size):
                particle_velocities[i] = self.calculate_velocity(i, particle_velocities, particle_positions,
                                                                 particle_personal_best, particle_global_best)
                particle_positions[i] = self.update_particle(i, particle_positions, particle_velocities)
                binary_trial_population[i] = self.binariser.binarise(particle_positions[i])

                if not any(binary_trial_population[i]):
                    particle_fitness[i] = 0.0
                    particle_fitness_test[i] = 0.0
                else:
                    particle_fitness[i] = fitness_function.evaluate_on_validation(binary_trial_population[i])
                    # particle_fitness_test[i] = fitness_function.evaluate_on_test(binary_population[i])

                particle_sizes[i] = np.sum(binary_trial_population[i])
                features_frequencies = features_frequencies + binary_trial_population[i]

                archive[spent_nfes] = copy.deepcopy(binary_trial_population[i])
                archive_fitness[spent_nfes] = particle_fitness[i]

                if particle_fitness[i] > particle_personal_best_fitness[i]:
                    binary_population[i] = copy.deepcopy(binary_trial_population[i])
                    particle_personal_best[i] = copy.deepcopy(particle_positions[i])
                    particle_personal_best_fitness[i] = particle_fitness[i]
                    particle_personal_best_sizes[i] = particle_sizes[i]
                    particle_personal_best_fitness_test[i] = fitness_function.evaluate_on_test(binary_population[i])

                elif particle_fitness[i] == particle_personal_best_fitness[i] and particle_sizes[i] < \
                        particle_personal_best_sizes[i]:
                    binary_population[i] = copy.deepcopy(binary_trial_population[i])
                    particle_personal_best[i] = copy.deepcopy(particle_positions[i])
                    particle_personal_best_fitness[i] = particle_fitness[i]
                    particle_personal_best_sizes[i] = particle_sizes[i]
                    particle_personal_best_fitness_test[i] = fitness_function.evaluate_on_test(binary_population[i])

                spent_nfes += 1

                if spent_nfes / self.max_nfes in log_points:
                    population_fitness_test, validation_metrics, population_scores, test_metrics, population_scores_test = self._get_comprehensive_logging_data(
                        binary_population, fitness_function)

                if spent_nfes >= self.max_nfes:
                    break
                else:
                    logger.log(spent_nfes=spent_nfes, max_nfes=self.max_nfes, wasted_nfes=wasted_nfes,
                               population=copy.deepcopy(particle_personal_best),
                               fitness_metric=fitness_function.fitness,
                               population_fitness=copy.deepcopy(particle_personal_best_fitness),
                               population_fitness_test=copy.deepcopy(particle_personal_best_fitness_test),
                               metrics=validation_metrics,
                               population_scores=copy.deepcopy(population_scores),
                               population_scores_test=copy.deepcopy(population_scores_test),
                               feature_frequencies=copy.deepcopy(features_frequencies))

            particle_global_best, particle_global_best_fitness, particle_global_best_size = self.update_global_best(
                particle_global_best, particle_global_best_fitness, particle_global_best_size,
                particle_personal_best_fitness, particle_personal_best, particle_personal_best_sizes)

            if spent_nfes >= self.max_nfes:
                population_fitness_test, validation_metrics, population_scores, test_metrics, population_scores_test = self._get_comprehensive_logging_data(
                    binary_population, fitness_function)

                logger.log(spent_nfes=spent_nfes, max_nfes=self.max_nfes, wasted_nfes=wasted_nfes,
                           population=copy.deepcopy(particle_personal_best),
                           fitness_metric=fitness_function.fitness,
                           population_fitness=copy.deepcopy(particle_personal_best_fitness),
                           population_fitness_test=copy.deepcopy(particle_personal_best_fitness_test),
                           metrics=validation_metrics,
                           population_scores=copy.deepcopy(population_scores),
                           population_scores_test=copy.deepcopy(population_scores_test),
                           feature_frequencies=copy.deepcopy(features_frequencies))

        output_quality = fitness_function.evaluate_final_solution(self.binariser.binarise(particle_global_best))
        logger.log_output(output_quality, archive=copy.deepcopy(archive),
                          archive_fitness=copy.deepcopy(archive_fitness),
                          population=copy.deepcopy(binary_population),
                          population_fitness=copy.deepcopy(particle_fitness))

        return self.binariser.binarise(particle_global_best), particle_global_best_fitness

    def initialise(self, fitness_function, population_size, small_size, small_amount):
        half_range = (self.bound_handler.upper_bound - self.bound_handler.lower_bound) / 2
        population = np.random.rand(population_size,
                                    fitness_function.dimensionality) * half_range + self.bound_handler.lower_bound

        for i in range(0, int(small_amount * population_size)):
            random_features = random.sample(range(0, fitness_function.dimensionality),
                                            math.ceil(small_size * fitness_function.dimensionality))
            for feature in random_features:
                population[i][feature] = half_range + np.random.rand() * (self.max_bound - self.min_bound)

        for i in range(int(population_size * small_amount), population_size):
            # Pick feature count at random:
            size = np.random.randint(fitness_function.dimensionality // 2, fitness_function.dimensionality + 1)
            # Pick 'size' of unique features:
            random_features = random.sample(range(0, fitness_function.dimensionality), size)
            for feature in random_features:
                population[i][feature] = half_range + np.random.rand() * (self.max_bound - self.min_bound)

        return population

    def calculate_velocity(self, idx, particle_velocities, particle_positions, particle_personal_best,
                           particle_global_best):
        velocity = np.zeros(len(particle_global_best))
        maxVelocity = 6.0
        for j in range(len(particle_global_best)):
            velocity[j] = self.inertia * particle_velocities[idx][j] + self.c1 * np.random.rand() * (
                    particle_personal_best[idx][j] - particle_positions[idx][j]) + self.c2 * np.random.rand() * (
                                  particle_global_best[j] - particle_positions[idx][j])

            if velocity[j] > maxVelocity:
                velocity[j] = maxVelocity
            if velocity[j] < maxVelocity * (-1):
                velocity[j] = maxVelocity * (-1)

        return velocity

    def update_particle(self, idx, particle_positions, particle_velocities):
        for j in range(len(particle_positions[0])):
            particle_positions[idx][j] += particle_velocities[idx][j]

            if particle_positions[idx][j] > self.max_bound:
                particle_positions[idx][j] = self.max_bound

            if particle_positions[idx][j] < self.min_bound:
                particle_positions[idx][j] = self.min_bound

        return particle_positions[idx]

    def update_global_best(self, global_best, global_best_fitness, global_best_size, particle_personal_best_fitness,
                           particle_personal_best, particle_personal_best_sizes):
        personal_best_max = np.max(particle_personal_best_fitness)
        personal_best_max_idx = np.argmax(particle_personal_best_fitness)
        if personal_best_max > global_best_fitness:
            personal_best_max_idx = np.argmax(particle_personal_best_fitness)
            global_best = copy.deepcopy(particle_personal_best[personal_best_max_idx])
            global_best_fitness = particle_personal_best_fitness[personal_best_max_idx]
            global_best_size = particle_personal_best_sizes[personal_best_max_idx]

        elif personal_best_max == global_best_fitness and particle_personal_best_sizes[
            personal_best_max_idx] < global_best_size:
            personal_best_max_idx = np.argmax(particle_personal_best_fitness)
            global_best = copy.deepcopy(particle_personal_best[personal_best_max_idx])
            global_best_fitness = particle_personal_best_fitness[personal_best_max_idx]
            global_best_size = particle_personal_best_sizes[personal_best_max_idx]

        return global_best, global_best_fitness, global_best_size

    def create_random_particle(self, dimensionality):
        random_particle = np.empty(dimensionality)
        for j in range(dimensionality):
            random_particle[j] = self.min_bound + np.random.rand() * (self.max_bound - self.min_bound)

        return random_particle