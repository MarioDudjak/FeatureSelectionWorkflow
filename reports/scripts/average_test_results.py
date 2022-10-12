import numpy as np

from src.experiment.setup import classifiers
from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor

overall_results = {}
for classifier in classifiers:
    overall_results[classifier.name] = {}

optimiser_name = 'PSOIniPG'
binariser_name = 'FixedBinarizer'
runs = 30
log_points = 14


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


for file in DatasetProvider().get_processed_dataset_list():
    for classifier in classifiers:
        filename = '_'.join(['FixedSplit_Holdout_F1', optimiser_name, classifier.name, file.name])
        print("Processing file {0}".format(filename))

        header, data = CsvProcessor().read_file(filename='logs/outputQuality/' + filename)
        search_header, search_data = CsvProcessor().read_file(filename='logs/populationQuality/' + filename)
        if header is not None and data is not None and search_header is not None and search_data is not None:
            data = [row for i, row in enumerate(data) if i % 2 == 0]
            data = np.array(data)
            search_data = [row for i, row in enumerate(search_data) if row]
            search_data = np.array(search_data)
            results = {}
            results[header[0] + "_val"] = [float(best_sol[0]) for j, best_sol in enumerate(search_data) if
                                           j % (log_points - 1) == 0 and j != 0]

            col = 0
            for metric in header[:-1]:
                # results[metric] = sum([float(value) for value in data[:, col]]) / len(data[:, col])
                results[metric] = [float(value) for value in data[:, col]]
                # if metric == 'size':
                #     results[metric] = np.round(results[metric], 2)
                col = col + 1

            string_solutions = data[:, -1]
            solutions = np.zeros((runs, len(string_solutions[0])), dtype=bool)
            for i in range(len(string_solutions)):
                solutions[i] = [feature == '1' for feature in string_solutions[i]]

            asm = calculate_ASM(solutions)

            CsvProcessor().save_summary_results(filename=filename, header=header,
                                                data=list(results.values()))
            results["asm"] = [asm]
            overall_results[classifier.name][file.name] = results.values()

print(overall_results)

for alg_name, alg_results in overall_results.items():
    header = ['dataset'] + ['f_score_val', 'f_score_test', 'accuracy', 'mcr', 'size', 'asm']
    # data = [[str(dataset)] + [float(value) for value in values] for (dataset, values) in alg_results.items()]
    data = [[str(dataset)] + [np.round(np.average(value), 4) for value in values] for (dataset, values) in alg_results.items()]
    CsvProcessor().save_summary_results(filename=optimiser_name + "_" + alg_name, header=header,
                                        data=data)
