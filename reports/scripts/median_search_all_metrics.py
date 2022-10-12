import numpy as np

from src.experiment.setup import classifiers, fitnesses
from src.features.wrappers import fs_wrappers
from src.features.initialisation import initialisers
from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor

header = ['dataset_metric'] + [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
overall_results = {}
evaluation_procedures = ['_holdout', '3Fold', '5Fold']

for fitness in fitnesses:
    for procedure in evaluation_procedures:
        overall_results[fitness + "_" + procedure] = {}
        for optimiser in fs_wrappers:
            overall_results[fitness + "_" + procedure][optimiser.name] = {}
            for classifier in classifiers:
                overall_results[fitness + "_" + procedure][optimiser.name][classifier.name] = {}
                for dataset in DatasetProvider().get_processed_dataset_list():

                    filename = '_'.join([fitness, procedure, optimiser.name, classifier.name, dataset.name])
                    print("Processing file {0}".format(filename))

                    val_header, val_data = CsvProcessor().read_file(filename='logs/populationQuality/' + filename)
                    test_header, test_data = CsvProcessor().read_file(
                        filename='logs/populationQuality/' + filename + "_test")

                    if val_header is not None and val_data is not None:
                        overall_results[fitness + "_" + procedure][optimiser.name][classifier.name][dataset.name] = {}
                        data = [row for row in val_data if row]
                        data = np.array(data)
                        for idx, metric in enumerate(val_header[0:4]):
                            results = []
                            for search_point in range(0, 14):
                                results.append(np.median(
                                    np.array([float(row[idx]) for i, row in enumerate(data) if i % 14 == search_point])))
                            overall_results[fitness + "_" + procedure][optimiser.name][classifier.name][dataset.name][
                                metric] = results

                            CsvProcessor().save_summary_results(
                                filename='_'.join([fitness + "_" + procedure, optimiser.name, classifier.name]) + "_search",
                                header=header,
                                data=['_'.join([dataset.name.replace('_', ''), metric])] + results)

                    if test_header is not None and test_data is not None:
                        overall_results[fitness + "_" + procedure][optimiser.name][classifier.name][dataset.name] = {}
                        data = [row for row in test_data if row]
                        data = np.array(data)
                        for idx, metric in enumerate(test_header[0:4]):
                            results = []
                            for search_point in range(0, 14):
                                results.append(np.median(
                                    np.array([float(row[idx]) for i, row in enumerate(data) if i % 14 == search_point])))

                            overall_results[fitness + "_" + procedure][optimiser.name][classifier.name][dataset.name][
                                metric] = results

                            CsvProcessor().save_summary_results(
                                filename='_'.join(
                                    [fitness, procedure, optimiser.name, classifier.name]) + "_search_test",
                                header=header,
                                data=['_'.join([dataset.name.replace('_', ''), metric])] + results)

print(overall_results)
