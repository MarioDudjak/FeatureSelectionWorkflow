import numpy as np

from src.utils.file_handling.processors import CsvProcessor


class OutputQualityLogger:

    def __init__(self, optimiser_name, binariser_name, problem_name):
        self.filename = 'outputQuality/' + '_'.join([optimiser_name, problem_name])
        self.metrics = []
        self.results = []

    def log(self, final_solution_quality):
        if not self.metrics:
            self.metrics = final_solution_quality.keys()
        self.results.append(list(final_solution_quality.values()))

    def persist_logs(self):
        CsvProcessor().save_log_results(filename=self.filename, header=self.metrics, data=self.results)


class PopulationQualityLogger:

    def __init__(self, optimiser_name, binariser_name, problem_name):
        self.filename = 'populationQuality/' + '_'.join([optimiser_name, problem_name])
        self.population_stats = []
        self.metrics = []
        self.fitness_metric = ''

    def log(self, fitness_metric, population_fitness, metrics, population_scores):
        if not self.fitness_metric:
            self.fitness_metric = fitness_metric

        if not self.metrics and metrics is not None:
            for metric in metrics:
                self.metrics.append('_'.join([metric, "_best"]))

        scores_list = [np.max(population_fitness), np.mean(population_fitness), np.min(population_fitness)]

        best_idx = np.nanargmax(population_fitness)
        if type(best_idx) is np.ndarray:
            best_idx = best_idx[0]

        if population_scores is not None:
            for col in range(len(population_scores[0])):
                scores_per_metric = [scores[col] for scores in population_scores]
                scores_list.append(scores_per_metric[best_idx])
        scores_list.append(np.unique(population_fitness).shape[0])
        scores_list.append(np.std(population_fitness))
        self.population_stats.append(scores_list)

    def persist_logs(self):
        header = [self.fitness_metric + "_" + suffix for suffix in
                  ['best', 'mean', 'worst']] + self.metrics + ['#phenotypes', 'stddev']

        CsvProcessor().save_log_results(filename=self.filename, header=header, data=self.population_stats)


class PopulationQualityLoggerTest:

    def __init__(self, optimiser_name, binariser_name, problem_name):
        self.filename = 'populationQuality/' + '_'.join([optimiser_name, problem_name])
        self.population_stats = []
        self.metrics = []
        self.fitness_metric = ''

    def log(self, fitness_metric, population_fitness, population_fitness_test, test_metrics, population_scores_test):
        if not self.fitness_metric:
            self.fitness_metric = fitness_metric

        if not self.metrics and test_metrics is not None:
            for metric in test_metrics:
                self.metrics.append('_'.join([metric, "_best"]))

        best_idx = np.nanargmax(population_fitness)
        if type(best_idx) is np.ndarray:
            best_idx = best_idx[0]

        worst_idx = np.nanargmin(population_fitness)
        if type(worst_idx) is np.ndarray:
            worst_idx = worst_idx[0]

        scores_list = [population_fitness_test[best_idx], np.mean(population_fitness_test),
                       population_fitness_test[worst_idx]]

        if population_scores_test is not None:
            for col in range(len(population_scores_test[0])):
                scores_per_metric = [scores[col] for scores in population_scores_test]
                scores_list.append(scores_per_metric[best_idx])
        scores_list.append(np.unique(population_fitness_test).shape[0])
        scores_list.append(np.std(population_fitness_test))
        self.population_stats.append(scores_list)

    def persist_logs(self):
        header = [self.fitness_metric + "_" + suffix for suffix in
                  ['best', 'mean', 'worst']] + self.metrics + ['#phenotypes', 'stddev']

        CsvProcessor().save_log_results(filename=self.filename, header=header, data=self.population_stats)


class VectorSizeLogger:

    def __init__(self, optimiser_name, binariser_name, problem_name):
        self.filename = 'vectorSize/' + '_'.join([optimiser_name, problem_name])
        self.vector_size_stats = []

    def log(self, binary_population):
        sizes = [sum(candidate) for candidate in binary_population]
        self.vector_size_stats.append([np.min(sizes), np.mean(sizes), np.max(sizes)])

    def persist_logs(self):
        pass
        CsvProcessor().save_log_results(filename=self.filename, header=['min', 'mean', 'max'],
                                        data=self.vector_size_stats)


class FeatureFrequencyLogger:

    def __init__(self, optimiser_name, binariser_name, problem_name):
        self.filename = 'featureFrequency/' + '_'.join([optimiser_name, problem_name])
        self.feature_frequency_stats = []

    def log(self, feature_frequencies, spent_nfes):
        relative_feature_frequencies = feature_frequencies / spent_nfes
        self.feature_frequency_stats.append(relative_feature_frequencies)

    def persist_logs(self):
        CsvProcessor().save_log_results(filename=self.filename,
                                        header=['f' + str(i + 1) for i in range(len(self.feature_frequency_stats[0]))],
                                        data=self.feature_frequency_stats)


class PopulationDiversityLogger:

    def __init__(self, optimiser_name, binariser_name, problem_name):
        self.filename = 'populationDiversity/' + '_'.join([optimiser_name, problem_name])
        self.diversity_stats = []

    def log(self, population):
        avg_distances = []
        median_distances = []
        dimensionality = len(population[0])
        for idx, candidate in enumerate(population):
            candidate_distance_from_all = [self._get_candidates_distance(candidate, candidate2) / dimensionality for
                                           i, candidate2 in
                                           enumerate(population) if idx != i]
            avg_distances.append(sum(candidate_distance_from_all) / len(candidate_distance_from_all))
            median_distances.append(np.median(candidate_distance_from_all))

        unique_candidates = np.unique(population, axis=0)
        self.diversity_stats.append(
            [np.min(avg_distances), np.mean(avg_distances), np.max(avg_distances), np.std(avg_distances),
             np.min(median_distances), np.mean(median_distances), np.max(median_distances), np.std(median_distances),
             unique_candidates.shape[0]])

    def persist_logs(self):
        CsvProcessor().save_log_results(filename=self.filename,
                                        header=['Avg_Hamming_min', 'Avg_Hamming_mean', 'Avg_Hamming_max',
                                                'Avg_Hamming_std',
                                                'Median_Hamming_min', 'Median_Hamming_mean', 'Median_Hamming_max',
                                                'Median_Hamming_std',
                                                '#genotypes'], data=self.diversity_stats)

    @staticmethod
    def _get_candidates_distance(candidate1, candidate2):
        diff = [0 if candidate1[idx] == candidate2[idx] else 1 for idx in range(len(candidate1))]
        return sum(diff)


class WastedEvaluationsLogger:

    def __init__(self, optimiser_name, binariser_name, problem_name):
        self.filename = 'wastedEvaluations/' + '_'.join([optimiser_name, binariser_name, problem_name])
        self.wasted_evaluations = []

    def log(self, wasted_evaluations, spent_NFEs, max_NFEs):
        self.wasted_evaluations.append([wasted_evaluations, spent_NFEs, max_NFEs])

    def persist_logs(self):
        CsvProcessor().save_log_results(filename=self.filename, header=['wasted', 'spend', 'max'],
                                        data=self.wasted_evaluations)


class ArchiveLogger:

    def __init__(self, optimiser_name, binariser_name, problem_name):
        self.filename = 'archive/' + '_'.join([optimiser_name, binariser_name, problem_name])
        self.archive = []
        self.archive_fitness = []

    def log(self, archive, archive_fitness):
        self.archive = archive * 1
        self.archive_fitness = archive_fitness

    def persist_logs(self):
        CsvProcessor().save_log_results(filename=self.filename + "_archive",
                                        header=['f' + str(i + 1) for i in range(len(self.archive[0]))],
                                        data=self.archive * 1)
        CsvProcessor().save_log_results(filename=self.filename + "_archive_fitness",
                                        header=['validation_fitness'],
                                        data=self.archive_fitness)


class PopulationLogger:

    def __init__(self, optimiser_name, binariser_name, problem_name):
        self.filename = 'population/' + '_'.join([optimiser_name, binariser_name, problem_name])
        self.population = []
        self.population_fitness = []

    def log(self, population, population_fitness):
        self.population = population * 1
        self.population_fitness = population_fitness

    def persist_logs(self):
        CsvProcessor().save_log_results(filename=self.filename + "_population",
                                        header=['f' + str(i + 1) for i in range(len(self.population[0]))],
                                        data=self.population * 1)
        CsvProcessor().save_log_results(filename=self.filename + "_population_fitness",
                                        header=['validation_fitness'],
                                        data=self.population_fitness)
