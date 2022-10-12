from src.features.logging.loggers import PopulationQualityLogger, VectorSizeLogger, OutputQualityLogger, \
    FeatureFrequencyLogger, PopulationDiversityLogger, WastedEvaluationsLogger, ArchiveLogger, \
    PopulationQualityLoggerTest, PopulationLogger

log_points = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class SearchLogger:

    def __init__(self, optimiser_name, binariser_name, problem_name):
        self.population_quality_logger = PopulationQualityLogger(optimiser_name, binariser_name, problem_name)
        self.population_quality_logger_test = PopulationQualityLoggerTest(optimiser_name, binariser_name,
                                                                          problem_name + '_test')
        self.vector_size_logger = VectorSizeLogger(optimiser_name, binariser_name, problem_name)
        self.output_quality_logger = OutputQualityLogger(optimiser_name, binariser_name, problem_name)
        self.feature_frequency_logger = FeatureFrequencyLogger(optimiser_name, binariser_name, problem_name)
        self.population_diversity_logger = PopulationDiversityLogger(optimiser_name, binariser_name, problem_name)
        self.wasted_evaluations_logger = WastedEvaluationsLogger(optimiser_name, binariser_name, problem_name)
        self.archive_logger = ArchiveLogger(optimiser_name, binariser_name, problem_name)
        self.population_logger = PopulationLogger(optimiser_name, binariser_name, problem_name)

    def log(self, spent_nfes, max_nfes, wasted_nfes, population, fitness_metric, population_fitness,
            population_fitness_test, metrics, population_scores, population_scores_test, feature_frequencies):
        ratio = spent_nfes / max_nfes
        if ratio in log_points:
            self.population_quality_logger.log(fitness_metric, population_fitness, metrics, population_scores)
            self.population_quality_logger_test.log(fitness_metric, population_fitness, population_fitness_test,
                                                    metrics, population_scores_test)
            self.vector_size_logger.log(population)
            self.feature_frequency_logger.log(feature_frequencies, spent_nfes)
            self.population_diversity_logger.log(population)
            self.wasted_evaluations_logger.log(wasted_nfes, spent_nfes, max_nfes)
            if ratio == 1.0:
                self.population_quality_logger.persist_logs()
                self.population_quality_logger_test.persist_logs()
                self.vector_size_logger.persist_logs()
                self.feature_frequency_logger.persist_logs()
                self.population_diversity_logger.persist_logs()
                self.wasted_evaluations_logger.persist_logs()

    def log_output(self, quality, archive, archive_fitness, population, population_fitness):
        self.output_quality_logger.log(quality)
        self.output_quality_logger.persist_logs()

        self.archive_logger.log(archive, archive_fitness)
        self.archive_logger.persist_logs()

        self.population_logger.log(population, population_fitness)
        self.population_logger.persist_logs()
