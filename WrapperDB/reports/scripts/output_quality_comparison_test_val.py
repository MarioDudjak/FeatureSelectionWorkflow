import numpy as np

from src.experiment.setup import classifiers, fitnesses
from src.features.wrappers import fs_wrappers
from src.features.initialisation import initialisers
from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor

header = ['dataset_metric'] + [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
overall_results = {}
evaluation_procedures = ['_holdout', '3Fold', '5Fold']

for dataset in DatasetProvider().get_processed_dataset_list():
    overall_results[dataset.name] = {}
    for fitness in fitnesses:
        for procedure in evaluation_procedures:
            overall_results[dataset.name][fitness + "_" + procedure] = {}
            for optimiser in fs_wrappers:
                overall_results[dataset.name][fitness + "_" + procedure][optimiser.name] = {}
                for classifier in classifiers:
                    overall_results[dataset.name][fitness + "_" + procedure][optimiser.name][classifier.name] = {}

                    filename = '_'.join([fitness, procedure, optimiser.name, classifier.name, dataset.name])
                    print("Processing file {0}".format(filename))

                    val_header, val_data = CsvProcessor().read_file(filename='logs/populationQuality/' + filename)
                    test_header, test_data = CsvProcessor().read_file(
                        filename='logs/populationQuality/' + filename + "_test")

                    if val_header is not None and val_data is not None and test_header is not None and test_data is not None:
                        val_data = [row for row in val_data if row]
                        test_data = [row for row in test_data if row]

                        results = []
                        for idx, data in enumerate(zip(val_data, test_data)):
                            if idx % 14 == 13:  # 13 jer se gleda best solution na kraju runa
                                results.append([float(data[0][0]), float(data[1][0])])       # 0 jer se gleda samo best

                        overall_results[dataset.name][fitness + "_" + procedure][optimiser.name][
                            classifier.name] = results

                        header = [fitness + "_" + procedure + "_" + optimiser.name + "_val", fitness + "_" + procedure + "_" + optimiser.name + "_test"]
                        CsvProcessor().save_summary_results(
                            filename=dataset.name + "_output_val_test_comparison",
                            header=header,
                            data=[header] + results)

print(overall_results)
