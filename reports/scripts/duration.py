from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor
from src.experiment.setup import classifiers
from src.features.wrappers import fs_wrappers

# datasets = DatasetProvider().get_processed_dataset_list()
# runs = 30
# wrapper = 'PSO'
# for classifier in classifiers:
#     results = []
#     for dataset in datasets:
#         filename = '_'.join(
#             ['', wrapper, classifier.name, dataset.name])
#         print("Processing file {0}".format(filename))
#
#         full_header, full_data = CsvProcessor().read_file(filename='logs/outputQuality/' + filename)
#         archive_header, archive_data = CsvProcessor().read_file(filename='summary/' + wrapper + "_" + classifier.name + "__archive_combinations")
#         archive_aggregation_duration = archive_header.index("duration")
#
#         if full_header is not None and full_data is not None:
#             results.append([dataset.name, sum([float(row[-2]) for row in full_data if row]) / runs, sum([float(row[-1]) for row in full_data if row]) / runs , sum([float(row[archive_aggregation_duration]) for row in archive_data if row[0]==dataset.name]) / runs])
#
#     CsvProcessor().save_log_results(
#         filename='outputQuality/duration/' + wrapper + "_" + classifier.name + "_duration",
#         header=['dataset', 'wrapper_duration', 'archive_duration', 'archive_aggregation_duration'],
#         data=results)



datasets = DatasetProvider().get_processed_dataset_list()
runs = 30
wrapper = 'PSO(4-2))'
for classifier in classifiers:
    results = []
    for dataset in datasets:
        filename = '_'.join(
            ['',wrapper, classifier.name, dataset.name])
        print("Processing file {0}".format(filename))

        full_header, full_data = CsvProcessor().read_file(filename='logs/outputQuality/' + filename)
        archive_header, archive_data = CsvProcessor().read_file(filename='summary/' + wrapper + "_" + classifier.name + "__archive_combinations")
        archive_aggregation_duration = archive_header.index("duration")

        if full_header is not None and full_data is not None:
            results.append([dataset.name, sum([float(row[-2]) for row in full_data if row]) / runs, sum([float(row[-1]) for row in full_data if row]) / runs , sum([float(row[archive_aggregation_duration]) for row in archive_data if row[0]==dataset.name]) / runs])

    CsvProcessor().save_log_results(
        filename='outputQuality/duration/' + wrapper + "_" + classifier.name + "_duration",
        header=['dataset', 'wrapper_duration', 'archive_duration', 'archive_aggregation_duration'],
        data=results)
