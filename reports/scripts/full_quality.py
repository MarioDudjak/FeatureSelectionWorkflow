from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor
from src.experiment.setup import classifiers

datasets = DatasetProvider().get_processed_dataset_list()
runs = 30
for classifier in classifiers:
    results = []
    for dataset in datasets:
        filename = '_'.join(
            ['Full', classifier.name, dataset.name])
        print("Processing file {0}".format(filename))

        full_header, full_data = CsvProcessor().read_file(filename='logs/Full/' + filename)

        if full_header is not None and full_data is not None:
            results.append([dataset.name, sum([float(row[0]) for row in full_data if row]) / runs, sum([float(row[1]) for row in full_data if row]) / runs, sum([float(row[6]) for row in full_data if row]) / runs])

    CsvProcessor().save_log_results(
        filename='Full/' + classifier.name + "_avg",
        header=['dataset', 'f1_macro_avg', 'accuracy_avg', 'recall_avg'],
        data=results)
