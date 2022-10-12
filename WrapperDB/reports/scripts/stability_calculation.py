from src.experiment.setup import fitnesses, classifiers, classification_metrics, experiment_setup
from src.features.wrappers import fs_wrappers
from src.models.classification.classificationProblem import ClassificationProblem
from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor

import pandas as pd
import numpy as np
import copy


def calculate_ASM(population):
    for i in range(len(population)):
        population[i] = np.array(population[i]).astype(bool)
    sa = 0
    m = len(population[0])
    for i in range(len(population) - 1):
        for j in range(i + 1, len(population)):
            s = sum(population[i] * population[j]) - (
                        sum(population[i]) * sum(population[j]) / m)
            s /= (min(sum(population[i]), sum(population[j])) - max(0, sum(population[i]) + sum(population[j]) - m))
            sa += s

    asm = (2 * sa) / (len(population) * (len(population) - 1))

    return asm


datasets = DatasetProvider().get_processed_dataset_list()

evaluation_procedures = ['_holdout', '3Fold', '5Fold']

header_idx = {
    'f1_val': 0,
    'accuracy_val': 1,
    'size_val': 2,
    'vector_val': 3,
    'f1_test': 4,
    'accuracy_test': 5,
    'size_test': 6,
    'vector_test': 7
}
population_size = 100
summary_results = {}

for fitness in fitnesses:
    for optimiser in fs_wrappers:
        for procedure in evaluation_procedures:
            for classifier in classifiers:
                summary_results[fitness + optimiser.name + procedure + classifier.name] = []
                for file in datasets:

                    dataset = pd.read_csv(file.path, header=0, index_col=0)
                    data = dataset.iloc[:, :-1].values
                    dimensionality = data.shape[1]

                    population_quality_filename = '_'.join(
                        [fitness, procedure, optimiser.name, classifier.name, file.name])
                    print("Processing file {0}".format(population_quality_filename))

                    val_header, val_data = CsvProcessor().read_file(
                        filename='logs/populationQuality/' + population_quality_filename)
                    test_header, test_data = CsvProcessor().read_file(
                        filename='logs/populationQuality/' + population_quality_filename + "_test")

                    search_results = []
                    if val_header is not None and val_data is not None and test_header is not None and test_data is not None:
                        val_data = [row for row in val_data if row]
                        test_data = [row for row in test_data if row]

                        for idx, data in enumerate(zip(val_data, test_data)):
                            if idx % 14 == 13:  # 13 jer se gleda best solution na kraju runa
                                search_results.append(
                                    [float(data[0][0]), float(data[1][0])])  # 0 jer se gleda samo best

                    filename = '_'.join(
                        [fitness, procedure, optimiser.name + "_", classifier.name, file.name,
                         'archive'])
                    print("Processing file {0}".format(filename))
                    test_header, test_data = CsvProcessor().read_file(filename='logs/archive/' + filename + "_test")

                    if test_header is not None and test_data is not None:
                        test_data = [row for row in test_data if row]
                        test_data = np.array(test_data)

                    if test_header is not None and test_data is not None:
                        run = 0
                        index = 0
                        population = np.empty([population_size, dimensionality])
                        population_fitness = np.zeros(population_size)
                        population_test_fitness = np.zeros(population_size)
                        best_solution = population[index]
                        best_test_fitness = 0
                        fitness_function = ClassificationProblem(file, classifier,
                                                                 random_state=run + 42,
                                                                 test_size=experiment_setup["test_size"],
                                                                 validation_size=experiment_setup[
                                                                     "validation_size"],
                                                                 wrapper_fitness_metric=fitness,
                                                                 metrics=classification_metrics)
                        for idx, candidate in enumerate(test_data):
                            if idx > 0 and idx % population_size == 0:
                                best_solution = np.array(best_solution).astype(bool)
                                best_solution = best_solution * 1

                                run += 1
                                index = 0
                                population = np.empty([population_size, dimensionality], dtype=bool)
                                population_fitness = np.zeros(population_size, dtype=float)
                                population_test_fitness = np.zeros(population_size, dtype=float)
                                fitness_function = ClassificationProblem(file, classifier,
                                                                         random_state=run + 42,
                                                                         test_size=experiment_setup["test_size"],
                                                                         validation_size=experiment_setup[
                                                                             "validation_size"],
                                                                         wrapper_fitness_metric=fitness,
                                                                         metrics=classification_metrics)
                                best_test_fitness = 0

                            population[index] = [True if bit == '1' else False for bit in
                                                 candidate[header_idx['vector_val']]]

                            population_fitness[index] = fitness_function.evaluate_on_validation(
                                np.array(population[index]).astype(bool))
                            population_test_fitness[index] = fitness_function.evaluate_on_test(
                                np.array(population[index]).astype(bool))
                            if population_test_fitness[index] == search_results[run][1] and population_fitness[index] == \
                                    search_results[run][0]:
                                best_solution = population[index]

                            if population_test_fitness[index] > best_test_fitness:
                                best_test_fitness = population_test_fitness[index]
                            index += 1
