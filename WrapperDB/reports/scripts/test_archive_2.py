import numpy as np
from src.experiment.setup import experiment_setup, classifiers, classification_metrics
from src.features.wrappers import fs_wrappers
from src.models.classification.classificationProblem import ClassificationProblem
from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor

evaluation_procedures = ['FixedSplit_Holdout_F1']
evaluations = 10000

full_header_indices = {
    "f1_macro": 0,
    "accuracy": 1,
    "mcr": 2,
    "size": 3,
    "vector": 4
}
fitness = 'f1_macro'

def main():
    datasets = DatasetProvider().get_processed_dataset_list()
    for optimiser in fs_wrappers:
        for procedure in evaluation_procedures:
            for classifier in classifiers:
                for dataset in datasets:

                    filename = '_'.join(
                        [procedure, optimiser.name + "_", classifier.name, dataset.name,
                         'archive'])
                    print("Processing file {0}".format(filename))

                    archive_header, archive_data = CsvProcessor().read_file(filename='logs/archive/' + filename)
                    archive_fitness_header, archive_fitness_data = CsvProcessor().read_file(
                        filename='logs/archive/' + filename + "_fitness")

                    if archive_header is not None and archive_data is not None:
                        candidates = [row for row in archive_data if row]
                        candidates_fitness = [row for row in archive_fitness_data if row]

                        run = 0
                        index = 0
                        fitness_function = ClassificationProblem(dataset, classifier,
                                                                 random_state=42,  # dodati run + za variable
                                                                 test_size=experiment_setup["test_size"],
                                                                 validation_size=experiment_setup[
                                                                     "validation_size"],
                                                                 wrapper_fitness_metric=fitness,
                                                                 metrics=classification_metrics)

                        full_header, full_data = CsvProcessor().read_file(filename='logs/outputQuality/Full_val__' + classifier.name + "_" + dataset.name)
                        full_fitness = float(full_data[run][int(full_header_indices[fitness])])
                        archive = []
                        archive_fitness = []

                        for idx, candidate in enumerate(candidates):
                            if idx > 0 and idx % evaluations == 0:
                                run += 1
                                index = 0

                                fitness_function = ClassificationProblem(dataset, classifier,
                                                                         random_state=42,
                                                                         # dodati run + za variable
                                                                         test_size=experiment_setup["test_size"],
                                                                         validation_size=experiment_setup[
                                                                             "validation_size"],
                                                                         wrapper_fitness_metric=fitness,
                                                                         metrics=classification_metrics)

                                for j, archive_el in enumerate(archive):
                                    feature_vector = np.array(archive_el).astype(bool)  # bool array from integer
                                    output_quality = fitness_function.evaluate_final_solution(
                                        feature_vector)
                                    validation_output_quality = fitness_function.evaluate_final_solution_on_validation(
                                        feature_vector)

                                    # output_quality['idx'] = j * run + j

                                    test_metrics = output_quality.keys()
                                    test_metrics = [metric + "_test" for metric in test_metrics]
                                    validation_metrics = validation_output_quality.keys()
                                    validation_metrics = [metric + "_val" for metric in validation_metrics]

                                    metrics = list(validation_metrics) + list(test_metrics)
                                    results = list(validation_output_quality.values()) + list(output_quality.values())
                                    CsvProcessor().save_log_results(filename='archive/' + filename + "_full_test",
                                                                    header=metrics,
                                                                    data=results)

                                CsvProcessor().save_log_results(filename='archive/' + filename + "_full_test",
                                                                header=['stop', 'run'],
                                                                data=['stop', str(run)])

                                archive = []
                                archive_fitness = []

                            feature_vector = np.array(candidate).astype(bool)  # bool array from integer
                            # validation_fitness = fitness_function.evaluate_on_validation(feature_vector, comprehensive=False)
                            validation_fitness = float(archive_fitness_data[run][index])
                            index += 1

                            if not any(
                                    np.array_equal(archive_element, feature_vector) for archive_element in archive) and \
                                    validation_fitness > full_fitness:
                                archive.append(feature_vector)
                                archive_fitness.append(validation_fitness)

                        run += 1
                        index = 0

                        fitness_function = ClassificationProblem(dataset, classifier,
                                                                 random_state=42,
                                                                 # dodati run + za variable
                                                                 test_size=experiment_setup["test_size"],
                                                                 validation_size=experiment_setup[
                                                                     "validation_size"],
                                                                 wrapper_fitness_metric=fitness,
                                                                 metrics=classification_metrics)

                        for j, archive_el in enumerate(archive):
                            feature_vector = np.array(archive_el).astype(bool)  # bool array from integer
                            output_quality = fitness_function.evaluate_final_solution(
                                feature_vector)
                            validation_output_quality = fitness_function.evaluate_final_solution_on_validation(
                                feature_vector)

                            # output_quality['idx'] = j * run + j

                            test_metrics = output_quality.keys()
                            test_metrics = [metric + "_test" for metric in test_metrics]
                            validation_metrics = validation_output_quality.keys()
                            validation_metrics = [metric + "_val" for metric in validation_metrics]

                            metrics = list(validation_metrics) + list(test_metrics)
                            results = list(validation_output_quality.values()) + list(output_quality.values())
                            CsvProcessor().save_log_results(filename='archive/' + filename + "_full_test",
                                                            header=metrics,
                                                            data=results)

                        CsvProcessor().save_log_results(filename='archive/' + filename + "_full_test",
                                                        header=['stop', 'run'],
                                                        data=['stop', str(run)])



if __name__ == "__main__":
    main()
