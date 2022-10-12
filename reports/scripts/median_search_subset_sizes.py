import numpy as np

from src.experiments.fs_study.setup import classifiers
from src.features.selection.wrappers import fs_wrappers
from src.experiments.fs_study.setup import experiment_setups
from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor


for experiment_name in experiment_setups.keys():
    overall_results = {}

    for classifier in classifiers:
        overall_results[classifier.name] = {}
        for optimiser_name in fs_wrappers.keys():
            for dataset in DatasetProvider().get_processed_dataset_list():
                for classifier_name in classifiers.keys():

                    filename = '_'.join([experiment_name, optimiser_name, classifier_name, dataset.name])
                    print("Processing file {0}".format(filename))

                    val_header, val_data = CsvProcessor().read_file(filename='logs/populationQuality/' + filename)
                    test_header, test_data = CsvProcessor().read_file(filename='logs/populationQuality/' + filename + "_test")

                    if val_header is not None and val_data is not None:
                        data = [row for i, row in enumerate(val_data) if row]
                        data = np.array(data)
                        results = []
                        for search_point in range(0, 14):
                            results.append(np.median(np.array([float(row[1]) for i, row in enumerate(data) if i % 14 == search_point])))

                        # CsvProcessor().save_summary_results(filename=filename, header=header,
                        #                                     data=results)

                        overall_results[classifier.name][dataset.name] = results

            print(overall_results)

    for alg_name, alg_results in overall_results.items():
        header = ['dataset'] + [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        data = [[str(dataset)] + [float(value) for value in values] for (dataset, values) in alg_results.items()]
        CsvProcessor().save_summary_results(filename=experiment_name + "/" + alg_name + "_fss_search", header=header,
                                            data=data)


