import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment.setup import experiment_setup, classifiers, classification_metrics, fitness
from src.features.initialisation import initialisers
from src.features.wrappers import fs_wrappers, global_params
from src.models.classification.classificationProblem import ClassificationProblem
from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor


def draw_initial_population(population, dimensionality):
    vectorCounts = [np.sum(candidate) for candidate in population]
    featuresActivated = np.sum(population, axis=0)
    counts, bins = np.histogram(vectorCounts)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.show()
    plt.close()


def main():
    datasets = DatasetProvider().get_processed_dataset_list()

    for file in datasets:
        dataset = pd.read_csv(file.path, header=0, index_col=0)
        data = dataset.iloc[:, :-1].values
        dimensionality = data.shape[1]
        for run in range(experiment_setup["runs"]):
            initialiser = initialisers.BinaryUniformFeatureCountInitialiser()
            initial_population = initialiser.create_population(global_params['populationSize'], dimensionality)
            # draw_initial_population(initial_population, dimensionality)
            for classifier in classifiers:
                fitness_function = ClassificationProblem(file, classifier,
                                                         random_state=42,  # fixed split
                                                         test_size=experiment_setup["test_size"],
                                                         validation_size=experiment_setup["validation_size"],
                                                         wrapper_fitness_metric=fitness,
                                                         metrics=classification_metrics)

                print(
                    "Performing evaluation on full fs subset for classifier {0} on file {1}".format(
                        classifier.name, file.name))
                feature_vector = np.ones(dimensionality, dtype=bool)
                output_quality = fitness_function.evaluate_final_solution_on_validation(feature_vector)
                metrics = output_quality.keys()
                results = list(output_quality.values())
                # CsvProcessor().save_log_results(
                #     filename='outputQuality/' + '_'.join(['Full_val_', classifier.name, file.name]), header=metrics,
                #     data=results)

                for wrapper in fs_wrappers:
                    print(
                        "Performing search in wrapper {0} for classifier {1} on file {2}".format(
                            wrapper.name, classifier.name, file.name))
                    wrapper.search('FixedSplit_Holdout_silhouette', fitness_function, initial_population)

                    print("Completed run {0}/{1} for classifier {2} on file {3} in".format(run + 1,
                                                                                            run,
                                                                                            classifier.name,
                                                                                            file.name,
                                                                                            ))


if __name__ == "__main__":
    main()
