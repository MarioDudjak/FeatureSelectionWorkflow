import numpy as np
from pathlib import Path
from src.experiment.setup import classifiers
from src.features.wrappers import fs_wrappers
from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor

header = ['dataset_metric'] + [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

overall_results = {}
files= []
count=0
for experiment_name in experiment_setups.keys():
    overall_results[experiment_name] = {}
    for optimiser in fs_wrappers:
        overall_results[experiment_name][optimiser.name] = {}
        for classifier in classifiers:
            overall_results[experiment_name][optimiser.name][classifier.name] = {}
            for dataset in DatasetProvider().get_processed_dataset_list():

                filename = '_'.join([experiment_name, optimiser.name + "_", classifier.name, dataset.name])
                _current_directory = Path(__file__).resolve()
                _logs_directory = _current_directory.parent.parent.joinpath('logs').joinpath('outputQuality')
                filepath = _logs_directory.joinpath(filename + '.csv')
                print("Processing file {0}".format(filepath))

                if not filepath.exists():
                    files.append(filename)
                    count += 1

print(files)
print(count)
