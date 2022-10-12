import numpy as np
import pandas as pd
import copy
from ReliefF import ReliefF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from src.experiment.setup import classifiers, experiment_setup, classification_metrics
from src.features.wrappers import fs_wrappers
from src.models.classification.classificationProblem import ClassificationProblem, ClassificationProblemCV
from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor

datasets = DatasetProvider().get_processed_dataset_list()

evaluation_procedures = ['FixedSplit_Holdout_F1']

header_idx = {
    'f1_macro_val': 0,
    'accuracy_val': 1,
    'mcr_val': 2,
    'size_val': 3,
    'vector_val': 4,
    'f1_macro_test': 4,
    'accuracy_test': 5,
    'mcr_test': 6,
    'size_test': 7,
    'vector_test': 8
}
aggregation_principles_to_evaluate = {
    'union': 0,
    'intersection': 0,
    'multiintersection': 0,
    'majority_voting': 0,
    'minority_voting': 0,
    'large_majority_voting': 0,
    'fitness_weighted_majority_voting': 0,
    'fitness_rank_weighted_majority_voting': 0,
    'size_weighted_majority_voting': 0,
    'fitness_size_rank_weighted_majority_voting_solutions': 0,
    'fitness_size_weighted_majority_voting_solutions': 0,
    'db_weighted_majority_voting_solutions': 0,
    'db_rank_weighted_majority_voting_solutions': 0,
    'pairwise_feature_frequency_majority_voting_solutions': 0,
    'filter_scores_voting_solutions': 1
}

population_size = 50
fitness = 'f1_macro'
summary_results = {}
asm_stability_results = {}
anhd_stability_results = {}


def get_Hamming_distance(candidate1, candidate2):
    diff = [0 if candidate1[idx] == candidate2[idx] else 1 for idx in range(len(candidate1))]
    return sum(diff)


def get_Davies_Bouldin_score(X, y, binary_mask):
    if binary_mask is not None:
        if any(binary_mask):
            X = X[:, binary_mask]
        else:
            return np.float("nan")

    return davies_bouldin_score(X, y)


def get_silhouette_score(X, y, binary_mask):
    if binary_mask is not None:
        if any(binary_mask):
            X = X[:, binary_mask]
        else:
            return -1

    return silhouette_score(X, y)


def get_feature_pairwise_frequency_matrix(population):
    dimensionality = len(population[0])
    matrix = np.zeros((dimensionality, dimensionality))
    for i in range(dimensionality):
        for j in range(dimensionality):
            if i != j:
                solutions_with_i_feature = [sol for sol in population if sol[i]]
                if solutions_with_i_feature:
                    matrix[i][j] += np.sum(solutions_with_i_feature, axis=0)[j]
                else:
                    matrix[i][j] += 0
            else:
                matrix[i][j] = 0

    return matrix


def calculate_ASM(population):
    to_ignore = 0
    for i in range(len(population)):
        population[i] = np.array(population[i]).astype(bool)
        if sum(population[i]) == 0:
            to_ignore += 1
    sa = 0
    m = len(population[0])
    for i in range(len(population) - 1):
        for j in range(i + 1, len(population)):
            if (sum(population[i]) == 0) or (sum(population[j]) == 0):
                continue
            if (sum(population[i]) == 0 and sum(population[j]) == 0) or (
                    sum(population[i]) == m and sum(population[j]) == m):
                sa += 1
            else:
                s = sum(population[i] * population[j]) - (
                        sum(population[i]) * sum(population[j]) / m)
                s /= (min(sum(population[i]), sum(population[j])) - max(0, sum(population[i]) + sum(population[j]) - m))
                sa += s

    if (to_ignore < len(population) - 1):
        asm = (2 * sa) / ((len(population) - to_ignore) * (len(population) - 1 - to_ignore))
    else:
        asm = 1

    return asm


def calculate_ANHD(population):
    nhi = 0
    for i in range(len(population) - 1):
        for j in range(i + 1, len(population)):
            hd = get_Hamming_distance(population[i], population[j])
            nhi += 1 - hd / len(population[i])

    anhd = (2 * nhi) / (len(population) * (len(population) - 1))
    return anhd

def calculate_chi2(X,y):
    fs = SelectKBest(chi2, k=len(X[0]))
    fs.fit(X,y)
    normalised_feature_scores = [0.1 + 0.8*(score - np.min(fs.scores_)) / (np.max(fs.scores_) - np.min(fs.scores_)) for score in fs.scores_]

    return normalised_feature_scores

def calculate_ReliefF(X, y):
    y = y.to_numpy()
    fs = ReliefF(n_neighbors=1, n_features_to_keep=len(X[0]))
    fs.fit(X, y)
    feature_scores = fs.feature_scores
    normalised_feature_scores = [0.1 + 0.8*(score - np.min(feature_scores)) / (np.max(feature_scores) - np.min(feature_scores)) for score in feature_scores]
    return normalised_feature_scores


def feature_subset_aggregation(dataset, X, y, run, population, population_fitness, population_kfold_fitness,
                               population_clustering_index,
                               search_results,
                               best_test_fitness, feature_scores):
    fitness_function_CAC = ClassificationProblem(file, classifier,
                                                 random_state=42,  # dodati run + za variable
                                                 test_size=experiment_setup["test_size"],
                                                 validation_size=experiment_setup[
                                                     "validation_size"],
                                                 wrapper_fitness_metric='accuracy',
                                                 metrics=classification_metrics)

    fitness_function_F1 = ClassificationProblem(file, classifier,
                                                random_state=42,  # dodati run + za variable
                                                test_size=experiment_setup["test_size"],
                                                validation_size=experiment_setup[
                                                    "validation_size"],
                                                wrapper_fitness_metric='f1_macro',
                                                metrics=classification_metrics)
    fitness_function_sensitivity = ClassificationProblem(file, classifier,
                                                         random_state=42,  # dodati run + za variable
                                                         test_size=experiment_setup["test_size"],
                                                         validation_size=experiment_setup[
                                                             "validation_size"],
                                                         wrapper_fitness_metric='sensitivity',
                                                         metrics=classification_metrics)
    fitness_function_CV = ClassificationProblemCV(file, classifier,
                                                  random_state=42,  # dodati run + za variable
                                                  test_size=experiment_setup["test_size"],
                                                  k_folds=5,
                                                  wrapper_fitness_metric='f1_macro',
                                                  metrics=classification_metrics)

    best_solution = np.asarray([f == '1' for f in search_results[run][-1]])
    val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(best_solution)
    kfold_fitness = fitness_function_CV.evaluate_on_validation(best_solution)
    best_solution = ','.join([str(feature) for feature in search_results[run][-1]])
    best_solution_size = np.sum([str(feature) == '1' for feature in search_results[run][-1]])
    results = {"dataset": dataset, "run": run, "best_solution": best_solution, "best_solution_size": best_solution_size,
               "validation_fitness": float(search_results[run][0]), "F1_test": float(search_results[run][1]),
               "CAC_test": float(search_results[run][2]), "best_test_fitness": float(best_test_fitness),
               "best_solution_val_sensitivity": val_sensitivity, "best_solution_kfold_fitness": kfold_fitness}

    intersection = np.array([True] * len(population[0]), dtype=bool)
    union = np.array([False] * len(population[0]), dtype=bool)
    multiintersection = np.array([False] * len(population[0]), dtype=bool)

    if aggregation_principles_to_evaluate['majority_voting']:
        activated_features = np.sum(population, axis=0)
        activated_features = activated_features >= (len(population) // 2)
        majority_voting = activated_features * 1

        fitness = fitness_function.evaluate_on_validation(majority_voting)
        test_CAC = fitness_function_CAC.evaluate_on_test(majority_voting)
        test_F1 = fitness_function_F1.evaluate_on_test(majority_voting)
        majority_voting_db_index = get_Davies_Bouldin_score(X, y, majority_voting)
        majority_voting_silhouette_index = get_silhouette_score(X, y, majority_voting)

        majority_voting = majority_voting * 1
        results["majority_voting"] = ','.join([str(feature) for feature in majority_voting])
        results["majority_voting_validation_fitness"] = fitness
        results["majority_voting_test_CAC"] = test_CAC
        results["majority_voting_test_F1"] = test_F1
        results["majority_voting_size"] = np.sum(majority_voting)
        results["majority_voting_DB"] = majority_voting_db_index
        results["majority_voting_silhouette"] = majority_voting_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(majority_voting)
        results["majority_voting_val_sensitivity"] = val_sensitivity
        results["majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['minority_voting']:
        activated_features = np.sum(population, axis=0)
        activated_features = activated_features >= (len(population) // 3)
        minority_voting = activated_features * 1

        minority_voting_fitness = fitness_function.evaluate_on_validation(minority_voting)
        minority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(minority_voting)
        minority_voting_test_F1 = fitness_function_F1.evaluate_on_test(minority_voting)
        minority_voting = minority_voting * 1
        minority_voting_db_index = get_Davies_Bouldin_score(X, y, minority_voting)
        minority_voting_silhouette_index = get_silhouette_score(X, y, minority_voting)

        results["minority_voting"] = ','.join([str(feature) for feature in minority_voting])
        results["minority_voting_validation_fitness"] = minority_voting_fitness
        results["minority_voting_test_CAC"] = minority_voting_test_CAC
        results["minority_voting_test_F1"] = minority_voting_test_F1
        results["minority_voting_size"] = np.sum(minority_voting)
        results["minority_voting_DB"] = minority_voting_db_index
        results["minority_voting_silhouette"] = minority_voting_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(minority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(minority_voting)
        results["minority_voting_val_sensitivity"] = val_sensitivity
        results["minority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['large_majority_voting']:
        activated_features = np.sum(population, axis=0)
        activated_features = activated_features >= (2 * len(population) // 3)
        large_majority_voting = activated_features * 1

        large_majority_voting_fitness = fitness_function.evaluate_on_validation(large_majority_voting)
        large_majority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(large_majority_voting)
        large_majority_voting_test_F1 = fitness_function_F1.evaluate_on_test(large_majority_voting)
        large_majority_voting = large_majority_voting * 1
        large_majority_voting_db_index = get_Davies_Bouldin_score(X, y, large_majority_voting)
        large_majority_voting_silhouette_index = get_silhouette_score(X, y, large_majority_voting)

        results["large_majority_voting"] = ','.join([str(feature) for feature in large_majority_voting])
        results["large_majority_voting_validation_fitness"] = large_majority_voting_fitness
        results["large_majority_voting_test_CAC"] = large_majority_voting_test_CAC
        results["large_majority_voting_test_F1"] = large_majority_voting_test_F1
        results["large_majority_voting_size"] = np.sum(large_majority_voting)
        results["large_majority_voting_DB"] = large_majority_voting_db_index
        results["large_majority_voting_silhouette"] = large_majority_voting_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(large_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(large_majority_voting)
        results["large_majority_voting_val_sensitivity"] = val_sensitivity
        results["large_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['fitness_weighted_majority_voting']:
        activated_features = np.zeros(len(population[0]), dtype=float)
        total_fitness = 0
        for idx, candidate in enumerate(population):
            activated_features += population_kfold_fitness[idx] * np.array(candidate).astype(int)
            total_fitness += population_kfold_fitness[idx]
        fitness_weighted_majority_voting = (activated_features >= total_fitness // 2) * 1

        fitness_weighted_majority_voting_fitness = fitness_function.evaluate_on_validation(
            fitness_weighted_majority_voting)
        fitness_weighted_majority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(
            fitness_weighted_majority_voting)
        fitness_weighted_majority_voting_test_F1 = fitness_function_F1.evaluate_on_test(
            fitness_weighted_majority_voting)
        fitness_weighted_majority_voting_db_index = get_Davies_Bouldin_score(X, y, fitness_weighted_majority_voting)
        fitness_weighted_majority_voting_silhouette_index = get_silhouette_score(X, y, fitness_weighted_majority_voting)

        results["fitness_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in fitness_weighted_majority_voting])
        results["fitness_weighted_majority_voting_validation_fitness"] = fitness_weighted_majority_voting_fitness
        results["fitness_weighted_majority_voting_test_CAC"] = fitness_weighted_majority_voting_test_CAC
        results["fitness_weighted_majority_voting_test_F1"] = fitness_weighted_majority_voting_test_F1
        results["fitness_weighted_majority_voting_size"] = np.sum(fitness_weighted_majority_voting)
        results["fitness_weighted_majority_voting_DB"] = fitness_weighted_majority_voting_db_index
        results["fitness_weighted_majority_voting_silhouette"] = fitness_weighted_majority_voting_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(fitness_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(fitness_weighted_majority_voting)
        results["fitness_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["fitness_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['fitness_rank_weighted_majority_voting']:
        activated_features = np.zeros(len(population[0]), dtype=int)
        total_votes = 0
        for idx, candidate in enumerate(population):
            rank = population_kfold_fitness[population_kfold_fitness <= population_kfold_fitness[idx]].shape[0]
            activated_features += rank * np.array(candidate).astype(int)
            total_votes += rank

        fitness_rank_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        fitness_rank_weighted_majority_voting_fitness = fitness_function.evaluate_on_validation(
            fitness_rank_weighted_majority_voting)
        fitness_rank_weighted_majority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(
            fitness_rank_weighted_majority_voting)
        fitness_rank_weighted_majority_voting_test_F1 = fitness_function_F1.evaluate_on_test(
            fitness_rank_weighted_majority_voting)
        fitness_rank_weighted_majority_voting_db_index = get_Davies_Bouldin_score(X, y,
                                                                                  fitness_rank_weighted_majority_voting)
        fitness_rank_weighted_majority_voting_silhouette_index = get_silhouette_score(X, y,
                                                                                      fitness_rank_weighted_majority_voting)

        results["fitness_rank_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in fitness_rank_weighted_majority_voting])
        results[
            "fitness_rank_weighted_majority_voting_validation_fitness"] = fitness_rank_weighted_majority_voting_fitness
        results["fitness_rank_weighted_majority_voting_test_CAC"] = fitness_rank_weighted_majority_voting_test_CAC
        results["fitness_rank_weighted_majority_voting_test_F1"] = fitness_rank_weighted_majority_voting_test_F1
        results["fitness_rank_weighted_majority_voting_size"] = np.sum(fitness_rank_weighted_majority_voting)
        results["fitness_rank_weighted_majority_voting_DB"] = fitness_rank_weighted_majority_voting_db_index
        results[
            "fitness_rank_weighted_majority_voting_silhouette"] = fitness_rank_weighted_majority_voting_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(fitness_rank_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(fitness_rank_weighted_majority_voting)
        results["fitness_rank_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["fitness_rank_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['size_weighted_majority_voting']:
        activated_features = np.zeros(len(population[0]), dtype=int)
        total_votes = 0
        for idx, candidate in enumerate(population):
            rank = len(population[np.sum(population, axis=1) >= np.sum(population[idx])])
            activated_features += rank * np.array(candidate).astype(int)
            total_votes += rank

        size_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        size_weighted_majority_voting_fitness = fitness_function.evaluate_on_validation(size_weighted_majority_voting)
        size_weighted_majority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(size_weighted_majority_voting)
        size_weighted_majority_voting_test_F1 = fitness_function_F1.evaluate_on_test(size_weighted_majority_voting)
        size_weighted_majority_voting_db_index = get_Davies_Bouldin_score(X, y,
                                                                          size_weighted_majority_voting)
        size_weighted_majority_voting_silhouette_index = get_silhouette_score(X, y,
                                                                              size_weighted_majority_voting)

        results["size_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in size_weighted_majority_voting])
        results["size_weighted_majority_voting_validation_fitness"] = size_weighted_majority_voting_fitness
        results["size_weighted_majority_voting_test_CAC"] = size_weighted_majority_voting_test_CAC
        results["size_weighted_majority_voting_test_F1"] = size_weighted_majority_voting_test_F1
        results["size_weighted_majority_voting_size"] = np.sum(size_weighted_majority_voting)
        results["size_weighted_majority_voting_DB"] = size_weighted_majority_voting_db_index
        results["size_weighted_majority_voting_silhouette"] = size_weighted_majority_voting_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(size_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(size_weighted_majority_voting)
        results["size_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["size_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['fitness_size_rank_weighted_majority_voting_solutions']:
        activated_features = np.zeros(len(population[0]), dtype=int)
        total_votes = 0
        for idx, candidate in enumerate(population):
            rank = population_kfold_fitness[population_kfold_fitness <= population_kfold_fitness[idx]].shape[0]
            rank += len(population[np.sum(population, axis=1) >= np.sum(population[idx])])
            activated_features += rank * np.array(candidate).astype(int)
            total_votes += rank

        fitness_size_rank_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        fitness_size_rank_weighted_majority_voting_fitness = fitness_function.evaluate_on_validation(
            fitness_size_rank_weighted_majority_voting)
        fitness_size_rank_weighted_majority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(
            fitness_size_rank_weighted_majority_voting)
        fitness_size_rank_weighted_majority_voting_test_F1 = fitness_function_F1.evaluate_on_test(
            fitness_size_rank_weighted_majority_voting)
        fitness_size_rank_weighted_majority_voting_db_index = get_Davies_Bouldin_score(X, y,
                                                                                       fitness_size_rank_weighted_majority_voting)
        fitness_size_rank_weighted_majority_voting_silhouette_index = get_silhouette_score(X, y,
                                                                                           fitness_size_rank_weighted_majority_voting)

        results["fitness_size_rank_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in fitness_size_rank_weighted_majority_voting])
        results[
            "fitness_size_rank_weighted_majority_voting_validation_fitness"] = fitness_size_rank_weighted_majority_voting_fitness
        results[
            "fitness_size_rank_weighted_majority_voting_test_CAC"] = fitness_size_rank_weighted_majority_voting_test_CAC
        results[
            "fitness_size_rank_weighted_majority_voting_test_F1"] = fitness_size_rank_weighted_majority_voting_test_F1
        results["fitness_size_rank_weighted_majority_voting_size"] = np.sum(fitness_size_rank_weighted_majority_voting)
        results["fitness_size_rank_weighted_majority_voting_DB"] = fitness_size_rank_weighted_majority_voting_db_index
        results[
            "fitness_size_rank_weighted_majority_voting_silhouette"] = fitness_size_rank_weighted_majority_voting_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            fitness_size_rank_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(fitness_size_rank_weighted_majority_voting)
        results["fitness_size_rank_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["fitness_size_rank_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['fitness_size_weighted_majority_voting_solutions']:
        activated_features = np.zeros(len(population[0]), dtype=float)
        total_votes = 0
        for idx, candidate in enumerate(population):
            weight = population_kfold_fitness[idx] + ((len(candidate) - np.sum(candidate)) / len(candidate))
            activated_features += weight * np.array(candidate).astype(int)
            total_votes += weight

        fitness_size_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        fitness_size_weighted_majority_voting_fitness = fitness_function.evaluate_on_validation(
            fitness_size_weighted_majority_voting)
        fitness_size_weighted_majority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(
            fitness_size_weighted_majority_voting)
        fitness_size_weighted_majority_voting_test_F1 = fitness_function_F1.evaluate_on_test(
            fitness_size_weighted_majority_voting)
        fitness_size_weighted_majority_voting_db_index = get_Davies_Bouldin_score(X, y,
                                                                                  fitness_size_weighted_majority_voting)
        fitness_size_weighted_majority_voting_silhouette_index = get_silhouette_score(X, y,
                                                                                      fitness_size_weighted_majority_voting)

        results["fitness_size_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in fitness_size_weighted_majority_voting])
        results[
            "fitness_size_weighted_majority_voting_validation_fitness"] = fitness_size_weighted_majority_voting_fitness
        results["fitness_size_weighted_majority_voting_test_CAC"] = fitness_size_weighted_majority_voting_test_CAC
        results["fitness_size_weighted_majority_voting_test_F1"] = fitness_size_weighted_majority_voting_test_F1
        results["fitness_size_weighted_majority_voting_size"] = np.sum(fitness_size_weighted_majority_voting)
        results["fitness_size_weighted_majority_voting_DB"] = fitness_size_weighted_majority_voting_db_index
        results[
            "fitness_size_weighted_majority_voting_silhouette"] = fitness_size_weighted_majority_voting_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(fitness_size_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(fitness_size_weighted_majority_voting)
        results["fitness_size_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["fitness_size_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['db_rank_weighted_majority_voting_solutions']:
        activated_features = np.zeros(len(population[0]), dtype=int)
        total_votes = 0
        for idx, candidate in enumerate(population):
            rank = population_clustering_index[population_clustering_index >= population_clustering_index[idx]].shape[0]
            activated_features += rank * np.array(candidate).astype(int)
            total_votes += rank

        db_rank_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        db_rank_weighted_majority_voting_fitness = fitness_function.evaluate_on_validation(
            db_rank_weighted_majority_voting)
        db_rank_weighted_majority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(
            db_rank_weighted_majority_voting)
        db_rank_weighted_majority_voting_test_F1 = fitness_function_F1.evaluate_on_test(
            db_rank_weighted_majority_voting)
        db_rank_weighted_majority_voting_db_rank_index = get_Davies_Bouldin_score(X, y,
                                                                                  db_rank_weighted_majority_voting)
        db_rank_weighted_majority_voting_silhouette_index = get_silhouette_score(X, y,
                                                                                 db_rank_weighted_majority_voting)

        results["db_rank_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in db_rank_weighted_majority_voting])
        results[
            "db_rank_weighted_majority_voting_fitness"] = db_rank_weighted_majority_voting_fitness
        results["db_rank_weighted_majority_voting_test_CAC"] = db_rank_weighted_majority_voting_test_CAC
        results["db_rank_weighted_majority_voting_test_F1"] = db_rank_weighted_majority_voting_test_F1
        results["db_rank_weighted_majority_voting_size"] = np.sum(db_rank_weighted_majority_voting)
        results["db_rank_weighted_majority_voting_db_rank"] = db_rank_weighted_majority_voting_db_rank_index
        results["db_rank_weighted_majority_voting_silhouette"] = db_rank_weighted_majority_voting_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(db_rank_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(db_rank_weighted_majority_voting)
        results["db_rank_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["db_rank_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['db_weighted_majority_voting_solutions']:
        activated_features = np.zeros(len(population[0]), dtype=float)
        total_votes = 0
        for idx, candidate in enumerate(population):
            activated_features += (1 / population_clustering_index[idx]) * np.array(candidate).astype(int)
            total_votes += (1 / population_clustering_index[idx])

        db_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        db_weighted_majority_voting_fitness = fitness_function.evaluate_on_validation(
            db_weighted_majority_voting)
        db_weighted_majority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(
            db_weighted_majority_voting)
        db_weighted_majority_voting_test_F1 = fitness_function_F1.evaluate_on_test(
            db_weighted_majority_voting)
        db_weighted_majority_voting_db_index = get_Davies_Bouldin_score(X, y,
                                                                        db_weighted_majority_voting)
        db_weighted_majority_voting_silhouette_index = get_silhouette_score(X, y,
                                                                            db_weighted_majority_voting)

        results["db_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in db_weighted_majority_voting])
        results[
            "db_weighted_majority_voting_fitness"] = db_weighted_majority_voting_fitness
        results["db_weighted_majority_voting_test_CAC"] = db_weighted_majority_voting_test_CAC
        results["db_weighted_majority_voting_test_F1"] = db_weighted_majority_voting_test_F1
        results["db_weighted_majority_voting_size"] = np.sum(db_weighted_majority_voting)
        results["db_weighted_majority_voting_DB"] = db_weighted_majority_voting_db_index
        results["db_weighted_majority_voting_silhouette"] = db_weighted_majority_voting_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(db_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(db_weighted_majority_voting)
        results["db_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["db_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['pairwise_feature_frequency_majority_voting_solutions']:
        pairwise_feature_frequency_matrix = get_feature_pairwise_frequency_matrix(population)
        activated_features = np.sum(pairwise_feature_frequency_matrix, axis=0)
        total_votes = np.sum(activated_features)

        pairwise_feature_frequency_majority_voting = (activated_features >= total_votes / dimensionality) * 1

        pairwise_feature_frequency_majority_voting_fitness = fitness_function.evaluate_on_validation(
            pairwise_feature_frequency_majority_voting)
        pairwise_feature_frequency_majority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(
            pairwise_feature_frequency_majority_voting)
        pairwise_feature_frequency_majority_voting_test_F1 = fitness_function_F1.evaluate_on_test(
            pairwise_feature_frequency_majority_voting)
        pairwise_feature_frequency_majority_voting_db_rank_index = get_Davies_Bouldin_score(X, y,
                                                                                            pairwise_feature_frequency_majority_voting)
        pairwise_feature_frequency_majority_voting_silhouette_index = get_silhouette_score(X, y,
                                                                                           pairwise_feature_frequency_majority_voting)

        results["pairwise_feature_frequency_majority_voting"] = ','.join(
            [str(feature) for feature in pairwise_feature_frequency_majority_voting])
        results[
            "pairwise_feature_frequency_majority_voting_fitness"] = pairwise_feature_frequency_majority_voting_fitness
        results[
            "pairwise_feature_frequency_majority_voting_test_CAC"] = pairwise_feature_frequency_majority_voting_test_CAC
        results[
            "pairwise_feature_frequency_majority_voting_test_F1"] = pairwise_feature_frequency_majority_voting_test_F1
        results["pairwise_feature_frequency_majority_voting_size"] = np.sum(pairwise_feature_frequency_majority_voting)
        results[
            "pairwise_feature_frequency_majority_voting_rank"] = pairwise_feature_frequency_majority_voting_db_rank_index
        results[
            "pairwise_feature_frequency_majority_voting_silhouette"] = pairwise_feature_frequency_majority_voting_silhouette_index

        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            pairwise_feature_frequency_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(pairwise_feature_frequency_majority_voting)
        results["pairwise_feature_frequency_majority_voting_val_sensitivity"] = val_sensitivity
        results["pairwise_feature_frequency_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['filter_scores_voting_solutions']:
        feature_thresholds = [1 - score for score in feature_scores]
        activated_features = np.sum(population, axis=0)
        feature_thresholds = [population.shape[0] * thr for thr in feature_thresholds]
        activated_features = activated_features >= feature_thresholds
        filter_scores_majority_voting = activated_features * 1

        fitness = fitness_function.evaluate_on_validation(filter_scores_majority_voting)
        test_CAC = fitness_function_CAC.evaluate_on_test(filter_scores_majority_voting)
        test_F1 = fitness_function_F1.evaluate_on_test(filter_scores_majority_voting)
        filter_scores_majority_voting_db_index = get_Davies_Bouldin_score(X, y, filter_scores_majority_voting)
        filter_scores_majority_voting_silhouette_index = get_silhouette_score(X, y, filter_scores_majority_voting)

        filter_scores_majority_voting = filter_scores_majority_voting * 1
        results["filter_scores_majority_voting"] = ','.join([str(feature) for feature in filter_scores_majority_voting])
        results["filter_scores_majority_voting_validation_fitness"] = fitness
        results["filter_scores_majority_votingtest_CAC"] = test_CAC
        results["filter_scores_majority_voting_test_F1"] = test_F1
        results["filter_scores_majority_voting_size"] = np.sum(filter_scores_majority_voting)
        results["filter_scores_majority_voting_DB"] = filter_scores_majority_voting_db_index
        results["filter_scores_majority_voting_silhouette"] = filter_scores_majority_voting_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(filter_scores_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(filter_scores_majority_voting)
        results["filter_scores_majority_voting_val_sensitivity"] = val_sensitivity
        results["filter_scores_majority_voting_kfold_fitness"] = kfold_fitness

    for solution in population:
        intersection = intersection * np.array(solution).astype(bool)
        union = union + np.array(solution).astype(bool)

    if aggregation_principles_to_evaluate['intersection']:
        if any(intersection):
            intersection_validation_fitness = fitness_function.evaluate_on_validation(intersection)
            intersection_test_CAC = fitness_function_CAC.evaluate_on_test(intersection)
            intersection_test_F1 = fitness_function_F1.evaluate_on_test(intersection)
        else:
            intersection_validation_fitness = 0
            intersection_test_CAC = 0
            intersection_test_F1 = 0

        intersection_db_index = get_Davies_Bouldin_score(X, y,
                                                         intersection)
        intersection_silhouette_index = get_silhouette_score(X, y,
                                                             intersection)

        intersection = intersection * 1
        results["intersection_solution"] = ','.join([str(feature) for feature in intersection])
        results["intersection_validation_fitness"] = intersection_validation_fitness
        results["intersection_test_CAC"] = intersection_test_CAC
        results["intersection_test_F1"] = intersection_test_F1
        results["intersection_size"] = np.sum(intersection)
        results["intersection_DB"] = intersection_db_index
        results["intersection_silhouette"] = intersection_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(intersection)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(intersection)
        results["intersection_val_sensitivity"] = val_sensitivity
        results["intersection_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['union']:
        union_validation_fitness = fitness_function.evaluate_on_validation(union)
        union_test_CAC = fitness_function_CAC.evaluate_on_test(union)
        union_test_F1 = fitness_function_F1.evaluate_on_test(union)
        union_db_index = get_Davies_Bouldin_score(X, y, union)
        union_silhouette_index = get_silhouette_score(X, y, union)

        union = union * 1
        results["union_solution"] = ','.join([str(feature) for feature in union])
        results["union_validation_fitness"] = union_validation_fitness
        results["unions_test_CAC"] = union_test_CAC
        results["unions_test_F1"] = union_test_F1
        results["union_size"] = np.sum(union)
        results["union_DB"] = union_db_index
        results["union_silhouette"] = union_silhouette_index

        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(union)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(union)
        results["union_val_sensitivity"] = val_sensitivity
        results["union_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['multiintersection']:
        intersections = []
        for candidate1 in population:
            for candidate2 in population:
                if not np.array_equal(candidate1, candidate2):
                    intersections.append(np.array(candidate1).astype(bool) * np.array(candidate2).astype(bool))

        activated_features = np.sum(intersections, axis=0)
        activated_features = activated_features >= (len(intersections) // 2)
        multiintersection = activated_features * 1
        # for intersect in intersections:
        #     multiintersection = multiintersection + np.array(intersect).astype(bool)

        if any(multiintersection):
            multiintersection_validation_fitness = fitness_function.evaluate_on_validation(multiintersection)
            multiintersection_test_CAC = fitness_function_CAC.evaluate_on_test(multiintersection)
            multiintersection_test_F1 = fitness_function_F1.evaluate_on_test(multiintersection)
        else:
            multiintersection_validation_fitness = 0
            multiintersection_test_CAC = 0
            multiintersection_test_F1 = 0

        multiintersection_db_index = get_Davies_Bouldin_score(X, y,
                                                              multiintersection)
        multiintersection_silhouette_index = get_silhouette_score(X, y,
                                                                  multiintersection)

        multiintersection = multiintersection * 1
        results["multiintersection_solution"] = ','.join([str(feature) for feature in multiintersection])
        results["multiintersection_validation_fitness"] = multiintersection_validation_fitness
        results["multiintersection_test_CAC"] = multiintersection_test_CAC
        results["multiintersection_test_F1"] = multiintersection_test_F1
        results["multiintersection_size"] = np.sum(multiintersection)
        results["multiintersection_DB"] = multiintersection_db_index
        results["multiintersection_silhouette"] = multiintersection_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(multiintersection)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(multiintersection)
        results["multiintersection_val_sensitivity"] = val_sensitivity
        results["multiintersection_kfold_fitness"] = kfold_fitness

    coalition = copy.deepcopy(population)
    coalition_fitness = copy.deepcopy(population_fitness)
    coalition_kfold_fitness = copy.deepcopy(population_kfold_fitness)
    coalition_clustering_index = copy.deepcopy(population_clustering_index)
    for idx1 in range(len(population)):
        best_idx = idx1
        best_partner = copy.deepcopy(population[idx1])
        best_partner_fitness = population_kfold_fitness[idx1]
        best_partner_clustering_index = population_clustering_index[idx1]
        for idx2 in range(len(population)):
            if get_Hamming_distance(population[idx1], population[idx2]) == 1:
                # if population_fitness[idx2] > best_partner_fitness:
                #     best_partner = copy.deepcopy(population[idx2])
                #     best_partner_fitness = population_fitness[idx2]
                #     best_partner_clustering_index = population_clustering_index[idx2]
                #
                # elif (population_fitness[idx2] == best_partner_fitness) and np.sum(population[idx2]) < np.sum(
                #         best_partner):
                #     best_partner = copy.deepcopy(population[idx2])
                #     best_partner_fitness = population_fitness[idx2]
                #     best_partner_clustering_index = population_clustering_index[idx2]

                if population_kfold_fitness[idx2] > best_partner_fitness:
                    best_idx = idx2
                    best_partner = copy.deepcopy(population[idx2])
                    best_partner_fitness = population_kfold_fitness[idx2]
                    best_partner_clustering_index = population_clustering_index[idx2]

                elif (population_kfold_fitness[idx2] == best_partner_fitness) and np.sum(population[idx2]) < np.sum(
                        best_partner):
                    best_idx = idx2
                    best_partner = copy.deepcopy(population[idx2])
                    best_partner_fitness = population_kfold_fitness[idx2]
                    best_partner_clustering_index = population_clustering_index[idx2]

        coalition[idx1] = copy.deepcopy(best_partner)
        coalition_fitness[idx1] = population_fitness[best_idx]
        coalition_kfold_fitness[idx1] = population_kfold_fitness[best_idx]
        coalition_clustering_index[idx1] = best_partner_clustering_index

    if aggregation_principles_to_evaluate['majority_voting']:
        activated_features = np.sum(coalition, axis=0)
        activated_features = activated_features >= (len(coalition) // 2)
        coalition_solution_majority_voting = activated_features * 1

        feature_vector = np.array(coalition_solution_majority_voting).astype(bool)  # bool array from integer
        if any(feature_vector):
            coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            coalition_validation_fitness = 0
            coalition_test_CAC = 0
            coalition_test_F1 = 0

        coalition_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        coalition_silhouette_index = get_silhouette_score(X, y, feature_vector)

        coalition_solution_majority_voting = coalition_solution_majority_voting * 1
        results["coalition_solution_majority_voting"] = ','.join(
            [str(feature) for feature in coalition_solution_majority_voting])
        results["coalition_solution_majority_voting_validation_fitness"] = coalition_validation_fitness
        results["coalition_solution_majority_voting_test_CAC"] = coalition_test_CAC
        results["coalition_solution_majority_voting_test_F1"] = coalition_test_F1
        results["coalition_solution_majority_voting_size"] = np.sum(coalition_solution_majority_voting)
        results["coalition_solution_majority_voting_DB_index"] = coalition_db_index
        results["coalition_solution_majority_voting_silhouette_index"] = coalition_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(coalition_solution_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(coalition_solution_majority_voting)
        results["coalition_solution_majority_voting_val_sensitivity"] = val_sensitivity
        results["coalition_solution_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['minority_voting']:
        activated_features = np.sum(coalition, axis=0)
        activated_features = activated_features >= (len(coalition) // 3)
        coalition_solution_minority_voting = activated_features * 1

        feature_vector = np.array(coalition_solution_minority_voting).astype(bool)  # bool array from integer
        if any(feature_vector):
            coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            coalition_validation_fitness = 0
            coalition_test_CAC = 0
            coalition_test_F1 = 0

        coalition_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        coalition_silhouette_index = get_silhouette_score(X, y, feature_vector)

        coalition_solution_minority_voting = coalition_solution_minority_voting * 1
        results["coalition_solution_minority_voting"] = ','.join(
            [str(feature) for feature in coalition_solution_minority_voting])
        results["coalition_solution_minority_voting_validation_fitness"] = coalition_validation_fitness
        results["coalition_solution_minority_voting_test_CAC"] = coalition_test_CAC
        results["coalition_solution_minority_voting_test_F1"] = coalition_test_F1
        results["coalition_solution_minority_voting_size"] = np.sum(coalition_solution_minority_voting)
        results["coalition_solution_minority_voting_DB_index"] = coalition_db_index
        results["coalition_solution_minority_voting_silhouette_index"] = coalition_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(coalition_solution_minority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(coalition_solution_minority_voting)
        results["coalition_solution_minority_voting_val_sensitivity"] = val_sensitivity
        results["coalition_solution_minority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['large_majority_voting']:
        activated_features = np.sum(coalition, axis=0)
        activated_features = activated_features >= (2 * len(coalition) // 3)
        coalition_solution_large_majority_voting = activated_features * 1

        feature_vector = np.array(coalition_solution_large_majority_voting).astype(bool)  # bool array from integer
        if any(feature_vector):
            coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            coalition_validation_fitness = 0
            coalition_test_CAC = 0
            coalition_test_F1 = 0

        coalition_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        coalition_silhouette_index = get_silhouette_score(X, y, feature_vector)

        coalition_solution_large_majority_voting = coalition_solution_large_majority_voting * 1
        results["coalition_solution_large_majority_voting"] = ','.join(
            [str(feature) for feature in coalition_solution_large_majority_voting])
        results["coalition_solution_large_majority_voting_validation_fitness"] = coalition_validation_fitness
        results["coalition_solution_large_majority_voting_test_CAC"] = coalition_test_CAC
        results["coalition_solution_large_majority_voting_test_F1"] = coalition_test_F1
        results["coalition_solution_large_majority_voting_size"] = np.sum(coalition_solution_large_majority_voting)
        results["coalition_solution_large_majority_voting_DB_index"] = coalition_db_index
        results["coalition_solution_large_majority_voting_silhouette_index"] = coalition_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(coalition_solution_large_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(coalition_solution_large_majority_voting)
        results["coalition_solution_large_majority_voting_val_sensitivity"] = val_sensitivity
        results["coalition_solution_large_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['fitness_weighted_majority_voting']:
        activated_features = np.zeros(len(coalition[0]), dtype=float)
        total_fitness = 0
        for idx, candidate in enumerate(coalition):
            activated_features += coalition_kfold_fitness[idx] * np.array(candidate).astype(int)
            total_fitness += coalition_kfold_fitness[idx]

        coalition_fitness_weighted_majority_voting = (activated_features >= total_fitness // 2) * 1

        feature_vector = np.array(coalition_fitness_weighted_majority_voting).astype(bool)  # bool array from integer
        if any(feature_vector):
            coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            coalition_validation_fitness = 0
            coalition_test_CAC = 0
            coalition_test_F1 = 0

        coalition_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        coalition_silhouette_index = get_silhouette_score(X, y, feature_vector)

        coalition_fitness_weighted_majority_voting = coalition_fitness_weighted_majority_voting * 1
        results["coalition_fitness_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in coalition_fitness_weighted_majority_voting])
        results["coalition_fitness_weighted_majority_voting_validation_fitness"] = coalition_validation_fitness
        results["coalition_fitness_weighted_majority_voting_test_CAC"] = coalition_test_CAC
        results["coalition_fitness_weighted_majority_voting_test_F1"] = coalition_test_F1
        results["coalition_fitness_weighted_majority_voting_size"] = np.sum(coalition_fitness_weighted_majority_voting)
        results["coalition_fitness_weighted_majority_voting_DB_index"] = coalition_db_index
        results["coalition_fitness_weighted_majority_voting_silhouette_index"] = coalition_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            coalition_fitness_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(coalition_fitness_weighted_majority_voting)
        results["coalition_fitness_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["coalition_fitness_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['fitness_rank_weighted_majority_voting']:
        activated_features = np.zeros(len(coalition[0]), dtype=int)
        total_votes = 0
        for idx, candidate in enumerate(coalition):
            rank = coalition_kfold_fitness[coalition_kfold_fitness <= coalition_kfold_fitness[idx]].shape[0]
            activated_features += rank * np.array(candidate).astype(int)
            total_votes += rank

        coalition_fitness_rank_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        feature_vector = np.array(coalition_fitness_rank_weighted_majority_voting).astype(
            bool)  # bool array from integer
        if any(feature_vector):
            coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            coalition_validation_fitness = 0
            coalition_test_CAC = 0
            coalition_test_F1 = 0

        coalition_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        coalition_silhouette_index = get_silhouette_score(X, y, feature_vector)

        coalition_fitness_rank_weighted_majority_voting = coalition_fitness_rank_weighted_majority_voting * 1
        results["coalition_fitness_rank_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in coalition_fitness_rank_weighted_majority_voting])
        results["coalition_fitness_rank_weighted_majority_voting_validation_fitness"] = coalition_validation_fitness
        results["coalition_fitness_rank_weighted_majority_voting_test_CAC"] = coalition_test_CAC
        results["coalition_fitness_rank_weighted_majority_voting_test_F1"] = coalition_test_F1
        results["coalition_fitness_rank_weighted_majority_voting_size"] = np.sum(
            coalition_fitness_rank_weighted_majority_voting)
        results["coalition_fitness_rank_weighted_majority_voting_DB_index"] = coalition_db_index
        results["coalition_fitness_rank_weighted_majority_voting_silhouette_index"] = coalition_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            coalition_fitness_rank_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(coalition_fitness_rank_weighted_majority_voting)
        results["coalition_fitness_rank_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["coalition_fitness_rank_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['size_weighted_majority_voting']:
        activated_features = np.zeros(len(coalition[0]), dtype=int)
        total_votes = 0
        for idx, candidate in enumerate(coalition):
            rank = len(coalition[np.sum(coalition, axis=1) >= np.sum(coalition[idx])])
            activated_features += rank * np.array(candidate).astype(int)
            total_votes += rank

        coalition_size_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        feature_vector = np.array(coalition_size_weighted_majority_voting).astype(bool)  # bool array from integer
        if any(feature_vector):
            coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            coalition_validation_fitness = 0
            coalition_test_CAC = 0
            coalition_test_F1 = 0

        coalition_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        coalition_silhouette_index = get_silhouette_score(X, y, feature_vector)

        coalition_size_weighted_majority_voting = coalition_size_weighted_majority_voting * 1
        results["coalition_size_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in coalition_size_weighted_majority_voting])
        results["coalition_size_weighted_majority_voting_validation_fitness"] = coalition_validation_fitness
        results["coalition_size_weighted_majority_voting_test_CAC"] = coalition_test_CAC
        results["coalition_size_weighted_majority_voting_test_F1"] = coalition_test_F1
        results["coalition_size_weighted_majority_voting_size"] = np.sum(coalition_size_weighted_majority_voting)
        results["coalition_size_weighted_majority_voting_DB_index"] = coalition_db_index
        results["coalition_size_weighted_majority_voting_silhouette_index"] = coalition_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            coalition_size_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(coalition_size_weighted_majority_voting)
        results["coalition_size_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["coalition_size_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['fitness_size_rank_weighted_majority_voting_solutions']:
        activated_features = np.zeros(len(coalition[0]), dtype=int)
        total_votes = 0
        for idx, candidate in enumerate(coalition):
            rank = coalition_kfold_fitness[coalition_kfold_fitness <= coalition_kfold_fitness[idx]].shape[0]
            rank += len(coalition[np.sum(coalition, axis=1) >= np.sum(coalition[idx])])
            activated_features += rank * np.array(candidate).astype(int)
            total_votes += rank

        coalition_fitness_size_rank_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        feature_vector = np.array(coalition_fitness_size_rank_weighted_majority_voting).astype(
            bool)  # bool array from integer
        if any(feature_vector):
            coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            coalition_validation_fitness = 0
            coalition_test_CAC = 0
            coalition_test_F1 = 0

        coalition_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        coalition_silhouette_index = get_silhouette_score(X, y, feature_vector)

        coalition_fitness_size_rank_weighted_majority_voting = coalition_fitness_size_rank_weighted_majority_voting * 1
        results["coalition_fitness_size_rank_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in coalition_fitness_size_rank_weighted_majority_voting])
        results[
            "coalition_fitness_size_rank_weighted_majority_voting_validation_fitness"] = coalition_validation_fitness
        results["coalition_fitness_size_rank_weighted_majority_voting_test_CAC"] = coalition_test_CAC
        results["coalition_fitness_size_rank_weighted_majority_voting_test_F1"] = coalition_test_F1
        results["coalition_fitness_size_rank_weighted_majority_voting_size"] = np.sum(
            coalition_fitness_size_rank_weighted_majority_voting)
        results["coalition_fitness_size_rank_weighted_majority_voting_DB_index"] = coalition_db_index
        results["coalition_fitness_size_rank_weighted_majority_voting_silhouette_index"] = coalition_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            coalition_fitness_size_rank_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(coalition_fitness_size_rank_weighted_majority_voting)
        results["coalition_fitness_size_rank_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["coalition_fitness_size_rank_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['fitness_size_weighted_majority_voting_solutions']:
        activated_features = np.zeros(len(coalition[0]), dtype=float)
        total_votes = 0
        for idx, candidate in enumerate(coalition):
            weight = coalition_kfold_fitness[idx] + ((len(candidate) - np.sum(candidate)) / len(candidate))
            activated_features += weight * np.array(candidate).astype(int)
            total_votes += weight

        coalition_fitness_size_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        feature_vector = np.array(coalition_fitness_size_weighted_majority_voting).astype(
            bool)  # bool array from integer
        if any(feature_vector):
            coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            coalition_validation_fitness = 0
            coalition_test_CAC = 0
            coalition_test_F1 = 0

        coalition_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        coalition_silhouette_index = get_silhouette_score(X, y, feature_vector)

        coalition_fitness_size_weighted_majority_voting = coalition_fitness_size_weighted_majority_voting * 1
        results["coalition_fitness_size_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in coalition_fitness_size_weighted_majority_voting])
        results["coalition_fitness_size_weighted_majority_voting_validation_fitness"] = coalition_validation_fitness
        results["coalition_fitness_size_weighted_majority_voting_test_CAC"] = coalition_test_CAC
        results["coalition_fitness_size_weighted_majority_voting_test_F1"] = coalition_test_F1
        results["coalition_fitness_size_weighted_majority_voting_size"] = np.sum(
            coalition_fitness_size_weighted_majority_voting)
        results["coalition_fitness_size_weighted_majority_voting_DB_index"] = coalition_db_index
        results["coalition_fitness_size_weighted_majority_voting_silhouette_index"] = coalition_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            coalition_fitness_size_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(coalition_fitness_size_weighted_majority_voting)
        results["coalition_fitness_size_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["coalition_fitness_size_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['db_rank_weighted_majority_voting_solutions']:
        activated_features = np.zeros(len(coalition[0]), dtype=int)
        total_votes = 0
        for idx, candidate in enumerate(coalition):
            rank = coalition_clustering_index[coalition_clustering_index >= coalition_clustering_index[idx]].shape[0]
            activated_features += rank * np.array(candidate).astype(int)
            total_votes += rank

        coalition_db_rank_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        feature_vector = np.array(coalition_db_rank_weighted_majority_voting).astype(
            bool)  # bool array from integer
        if any(feature_vector):
            coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            coalition_validation_fitness = 0
            coalition_test_CAC = 0
            coalition_test_F1 = 0

        coalition_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        coalition_silhouette_index = get_silhouette_score(X, y, feature_vector)

        coalition_db_rank_weighted_majority_voting = coalition_db_rank_weighted_majority_voting * 1
        results["coalition_db_rank_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in coalition_db_rank_weighted_majority_voting])
        results["coalition_db_rank_weighted_majority_voting_validation_fitness"] = coalition_validation_fitness
        results["coalition_db_rank_weighted_majority_voting_test_CAC"] = coalition_test_CAC
        results["coalition_db_rank_weighted_majority_voting_test_F1"] = coalition_test_F1
        results["coalition_db_rank_weighted_majority_voting_size"] = np.sum(
            coalition_db_rank_weighted_majority_voting)
        results["coalition_db_rank_weighted_majority_voting_DB_index"] = coalition_db_index
        results["coalition_db_rank_weighted_majority_voting_silhouette_index"] = coalition_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            coalition_db_rank_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(coalition_db_rank_weighted_majority_voting)
        results["coalition_db_rank_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["coalition_db_rank_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['db_weighted_majority_voting_solutions']:
        activated_features = np.zeros(len(coalition[0]), dtype=float)
        total_votes = 0
        for idx, candidate in enumerate(coalition):
            activated_features += (1 / coalition_clustering_index[idx]) * np.array(candidate).astype(int)
            total_votes += (1 / coalition_clustering_index[idx])

        coalition_db_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        feature_vector = np.array(coalition_db_weighted_majority_voting).astype(
            bool)  # bool array from integer
        if any(feature_vector):
            coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            coalition_validation_fitness = 0
            coalition_test_CAC = 0
            coalition_test_F1 = 0

        coalition_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        coalition_silhouette_index = get_silhouette_score(X, y, feature_vector)

        coalition_db_weighted_majority_voting = coalition_db_weighted_majority_voting * 1
        results["coalition_db_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in coalition_db_weighted_majority_voting])
        results["coalition_db_weighted_majority_voting_validation_fitness"] = coalition_validation_fitness
        results["coalition_db_weighted_majority_voting_test_CAC"] = coalition_test_CAC
        results["coalition_db_weighted_majority_voting_test_F1"] = coalition_test_F1
        results["coalition_db_weighted_majority_voting_size"] = np.sum(
            coalition_db_weighted_majority_voting)
        results["coalition_db_weighted_majority_voting_DB_index"] = coalition_db_index
        results["coalition_db_weighted_majority_voting_silhouette_index"] = coalition_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            coalition_db_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(coalition_db_weighted_majority_voting)
        results["coalition_db_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["coalition_db_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['pairwise_feature_frequency_majority_voting_solutions']:
        pairwise_feature_frequency_matrix = get_feature_pairwise_frequency_matrix(coalition)
        activated_features = np.sum(pairwise_feature_frequency_matrix, axis=0)
        total_votes = np.sum(activated_features)

        coalition_pairwise_feature_frequency_majority_voting = (activated_features >= total_votes / dimensionality) * 1

        feature_vector = np.array(coalition_pairwise_feature_frequency_majority_voting).astype(
            bool)  # bool array from integer
        if any(feature_vector):
            coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            coalition_validation_fitness = 0
            coalition_test_CAC = 0
            coalition_test_F1 = 0

        coalition_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        coalition_silhouette_index = get_silhouette_score(X, y, feature_vector)

        coalition_pairwise_feature_frequency_majority_voting = coalition_pairwise_feature_frequency_majority_voting * 1
        results["coalition_pairwise_feature_frequency_majority_voting"] = ','.join(
            [str(feature) for feature in coalition_pairwise_feature_frequency_majority_voting])
        results[
            "coalition_pairwise_feature_frequency_majority_voting_validation_fitness"] = coalition_validation_fitness
        results["coalition_pairwise_feature_frequency_majority_voting_test_CAC"] = coalition_test_CAC
        results["coalition_pairwise_feature_frequency_majority_voting_test_F1"] = coalition_test_F1
        results["coalition_pairwise_feature_frequency_majority_voting_size"] = np.sum(
            coalition_pairwise_feature_frequency_majority_voting)
        results["coalition_pairwise_feature_frequency_majority_voting_DB_index"] = coalition_db_index
        results["coalition_pairwise_feature_frequency_majority_voting_silhouette_index"] = coalition_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            coalition_pairwise_feature_frequency_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(coalition_pairwise_feature_frequency_majority_voting)
        results["coalition_pairwise_feature_frequency_majority_voting_val_sensitivity"] = val_sensitivity
        results["coalition_pairwise_feature_frequency_majority_voting_kfold_fitness"] = kfold_fitness

    coalition_solution_union = np.array([False] * len(coalition[0]), dtype=bool)
    coalition_solution_intersection = np.array([True] * len(coalition[0]), dtype=bool)
    for solution in coalition:
        coalition_solution_intersection = coalition_solution_intersection * np.array(solution).astype(bool)
        coalition_solution_union = coalition_solution_union + np.array(solution).astype(bool)

    if aggregation_principles_to_evaluate['union']:
        feature_vector = np.array(coalition_solution_union).astype(bool)  # bool array from integer
        coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)

        coalition_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        coalition_silhouette_index = get_silhouette_score(X, y, feature_vector)

        coalition_solution_union = coalition_solution_union * 1
        results["coalition_solution_union"] = ','.join([str(feature) for feature in coalition_solution_union])
        results["coalition_solution_union_validation_fitness"] = coalition_validation_fitness
        results["coalition_solution_union_test_CAC"] = coalition_test_CAC
        results["coalition_solution_union_test_F1"] = coalition_test_F1
        results["coalition_solution_union_size"] = np.sum(coalition_solution_union)
        results["coalition_solution_union_DB_index"] = coalition_db_index
        results["coalition_solution_union_index"] = coalition_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            coalition_solution_union)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(coalition_solution_union)
        results["coalition_solution_union_val_sensitivity"] = val_sensitivity
        results["coalition_solution_union_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['intersection']:
        feature_vector = np.array(coalition_solution_intersection).astype(bool)  # bool array from integer
        if any(feature_vector):
            coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            coalition_validation_fitness = 0
            coalition_test_CAC = 0
            coalition_test_F1 = 0

        coalition_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        coalition_silhouette_index = get_silhouette_score(X, y, feature_vector)

        coalition_solution_intersection = coalition_solution_intersection * 1
        results["coalition_solution_intersection"] = ','.join(
            [str(feature) for feature in coalition_solution_intersection])
        results["coalition_solution_intersection_validation_fitness"] = coalition_validation_fitness
        results["coalition_solution_intersection_test_CAC"] = coalition_test_CAC
        results["coalition_solution_intersection_test_F1"] = coalition_test_F1
        results["coalition_solution_intersection_size"] = np.sum(coalition_solution_intersection)
        results["coalition_solution_intersection_DB_index"] = coalition_db_index
        results["coalition_solution_intersection_index"] = coalition_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            coalition_solution_intersection)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(coalition_solution_intersection)
        results["coalition_solution_intersection_val_sensitivity"] = val_sensitivity
        results["coalition_solution_intersection_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['multiintersection']:
        intersections = []
        for candidate1 in coalition:
            for candidate2 in coalition:
                if not np.array_equal(candidate1, candidate2):
                    intersections.append(np.array(candidate1).astype(bool) * np.array(candidate2).astype(bool))

        coalition_solution_multiintersection = np.array([False] * len(coalition[0]), dtype=bool)

        activated_features = np.sum(intersections, axis=0)
        activated_features = activated_features >= (len(intersections) // 2)
        coalition_solution_multiintersection = activated_features * 1

        # for intersect in intersections:
        #     coalition_solution_multiintersection = coalition_solution_multiintersection + np.array(intersect).astype(bool)

        feature_vector = np.array(coalition_solution_multiintersection).astype(bool)  # bool array from integer
        if any(feature_vector):
            coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            coalition_validation_fitness = 0
            coalition_test_CAC = 0
            coalition_test_F1 = 0

        coalition_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        coalition_silhouette_index = get_silhouette_score(X, y, feature_vector)

        coalition_solution_multiintersection = coalition_solution_multiintersection * 1
        results["coalition_solution_multiintersection"] = ','.join(
            [str(feature) for feature in coalition_solution_multiintersection])
        results["coalition_solution_multiintersection_validation_fitness"] = coalition_validation_fitness
        results["coalition_solution_multiintersection_test_CAC"] = coalition_test_CAC
        results["coalition_solution_multiintersection_test_F1"] = coalition_test_F1
        results["coalition_solution_multiintersection_size"] = np.sum(coalition_solution_multiintersection)
        results["coalition_solution_multiintersection_DB_index"] = coalition_db_index
        results["coalition_solution_multiintersection_index"] = coalition_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            coalition_solution_multiintersection)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(coalition_solution_multiintersection)
        results["coalition_solution_multiintersection_val_sensitivity"] = val_sensitivity
        results["coalition_solution_multiintersection_kfold_fitness"] = kfold_fitness

    # similar_counts = []
    # indices_to_throw = []
    # for idx1 in range(len(population) - 1):
    #     count = 0
    #     for idx2 in range(idx1 + 1, len(population)):
    #         if get_Hamming_distance(population[idx1], population[idx2]) == 1:
    #             count += 1
    #             if population_fitness[idx1] > population_fitness[idx2]:
    #                 if idx2 not in indices_to_throw:
    #                     indices_to_throw.append(idx2)
    #             elif population_fitness[idx1] == population_fitness[idx2]:
    #                 if sum(population[idx1]) < sum(population[idx2]):
    #                     if idx2 not in indices_to_throw:
    #                         indices_to_throw.append(idx2)
    #                 else:
    #                     if idx1 not in indices_to_throw:
    #                         indices_to_throw.append(idx1)
    #             else:
    #                 if idx2 not in indices_to_throw:
    #                     indices_to_throw.append(idx2)
    #     similar_counts.append(count)

    similar_counts = []
    indices_to_throw = []
    for idx1 in range(len(population) - 1):
        count = 0
        for idx2 in range(idx1 + 1, len(population)):
            if get_Hamming_distance(population[idx1], population[idx2]) == 1:
                count += 1
                if population_kfold_fitness[idx1] <= population_kfold_fitness[idx2] and sum(population[idx1]) > sum(
                        population[idx2]):
                    if idx1 not in indices_to_throw:
                        indices_to_throw.append(idx1)
                elif population_kfold_fitness[idx2] <= population_kfold_fitness[idx1] and sum(population[idx2]) > sum(
                        population[idx1]):
                    if idx2 not in indices_to_throw:
                        indices_to_throw.append(idx2)
        similar_counts.append(count)

    print(similar_counts)

    reduced_population = copy.deepcopy(population)
    reduced_population = np.delete(reduced_population, indices_to_throw, axis=0)
    reduced_population_fitness = np.delete(population_fitness, indices_to_throw, axis=0)
    reduced_population_kfold_fitness = np.delete(population_kfold_fitness, indices_to_throw, axis=0)
    reduced_population_clustering_index = np.delete(population_clustering_index, indices_to_throw, axis=0)
    population = copy.deepcopy(reduced_population)
    population_fitness = copy.deepcopy(reduced_population_fitness)
    population_kfold_fitness = copy.deepcopy(reduced_population_kfold_fitness)
    population_clustering_index = copy.deepcopy(reduced_population_clustering_index)
    results["reduced_population_size"] = len(population)

    if aggregation_principles_to_evaluate['majority_voting']:
        activated_features = np.sum(population, axis=0)
        activated_features = activated_features >= (population.shape[0] // 2)
        reduced_solution_majority_voting = activated_features * 1

        feature_vector = np.array(reduced_solution_majority_voting).astype(bool)  # bool array from integer
        if any(feature_vector):
            reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            reduced_validation_fitness = 0
            reduced_test_CAC = 0
            reduced_test_F1 = 0

        reduced_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        reduced_silhouette_index = get_silhouette_score(X, y, feature_vector)

        reduced_solution_majority_voting = reduced_solution_majority_voting * 1
        results["reduced_solution_majority_voting"] = ','.join(
            [str(feature) for feature in reduced_solution_majority_voting])
        results["reduced_solution_majority_voting_validation_fitness"] = reduced_validation_fitness
        results["reduced_solution_majority_voting_test_CAC"] = reduced_test_CAC
        results["reduced_solution_majority_voting_test_F1"] = reduced_test_F1
        results["reduced_solution_majority_voting_size"] = np.sum(reduced_solution_majority_voting)
        results["reduced_solution_majority_voting_DB_index"] = reduced_db_index
        results["reduced_solution_majority_voting_silhouette_index"] = reduced_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(reduced_solution_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(reduced_solution_majority_voting)
        results["reduced_solution_majority_voting_val_sensitivity"] = val_sensitivity
        results["reduced_solution_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['minority_voting']:
        activated_features = np.sum(population, axis=0)
        activated_features = activated_features >= (population.shape[0] // 3)
        reduced_solution_minority_voting = activated_features * 1

        feature_vector = np.array(reduced_solution_minority_voting).astype(bool)  # bool array from integer
        if any(feature_vector):
            reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            reduced_validation_fitness = 0
            reduced_test_CAC = 0
            reduced_test_F1 = 0

        reduced_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        reduced_silhouette_index = get_silhouette_score(X, y, feature_vector)

        reduced_solution_minority_voting = reduced_solution_minority_voting * 1
        results["reduced_solution_minority_voting"] = ','.join(
            [str(feature) for feature in reduced_solution_minority_voting])
        results["reduced_solution_minority_voting_validation_fitness"] = reduced_validation_fitness
        results["reduced_solution_minority_voting_test_CAC"] = reduced_test_CAC
        results["reduced_solution_minority_voting_test_F1"] = reduced_test_F1
        results["reduced_solution_minority_voting_size"] = np.sum(reduced_solution_minority_voting)
        results["reduced_solution_minority_voting_DB_index"] = reduced_db_index
        results["reduced_solution_minority_voting_silhouette_index"] = reduced_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(reduced_solution_minority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(reduced_solution_minority_voting)
        results["reduced_solution_minority_voting_val_sensitivity"] = val_sensitivity
        results["reduced_solution_minority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['large_majority_voting']:
        activated_features = np.sum(population, axis=0)
        activated_features = activated_features >= (2 * population.shape[0] // 3)
        reduced_solution_large_majority_voting = activated_features * 1

        feature_vector = np.array(reduced_solution_large_majority_voting).astype(bool)  # bool array from integer
        if any(feature_vector):
            reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            reduced_validation_fitness = 0
            reduced_test_CAC = 0
            reduced_test_F1 = 0

        reduced_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        reduced_silhouette_index = get_silhouette_score(X, y, feature_vector)

        reduced_solution_large_majority_voting = reduced_solution_large_majority_voting * 1
        results["reduced_solution_large_majority_voting"] = ','.join(
            [str(feature) for feature in reduced_solution_large_majority_voting])
        results["reduced_solution_large_majority_voting_validation_fitness"] = reduced_validation_fitness
        results["reduced_solution_large_majority_voting_test_CAC"] = reduced_test_CAC
        results["reduced_solution_large_majority_voting_test_F1"] = reduced_test_F1
        results["reduced_solution_large_majority_voting_size"] = np.sum(reduced_solution_large_majority_voting)
        results["reduced_solution_large_majority_voting_DB_index"] = reduced_db_index
        results["reduced_solution_large_majority_voting_silhouette_index"] = reduced_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(reduced_solution_large_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(reduced_solution_large_majority_voting)
        results["reduced_solution_large_majority_voting_val_sensitivity"] = val_sensitivity
        results["reduced_solution_large_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['fitness_weighted_majority_voting']:
        activated_features = np.zeros(len(population[0]), dtype=float)
        total_fitness = 0
        for idx, candidate in enumerate(population):
            activated_features += population_kfold_fitness[idx] * np.array(candidate).astype(int)
            total_fitness += population_kfold_fitness[idx]

        reduced_fitness_weighted_majority_voting = (activated_features >= total_fitness // 2) * 1

        feature_vector = np.array(reduced_fitness_weighted_majority_voting).astype(bool)  # bool array from integer
        if any(feature_vector):
            reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            reduced_validation_fitness = 0
            reduced_test_CAC = 0
            reduced_test_F1 = 0

        reduced_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        reduced_silhouette_index = get_silhouette_score(X, y, feature_vector)

        reduced_fitness_weighted_majority_voting = reduced_fitness_weighted_majority_voting * 1
        results["reduced_fitness_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in reduced_fitness_weighted_majority_voting])
        results["reduced_fitness_weighted_majority_voting_validation_fitness"] = reduced_validation_fitness
        results["reduced_fitness_weighted_majority_voting_test_CAC"] = reduced_test_CAC
        results["reduced_fitness_weighted_majority_voting_test_F1"] = reduced_test_F1
        results["reduced_fitness_weighted_majority_voting_size"] = np.sum(reduced_fitness_weighted_majority_voting)
        results["reduced_fitness_weighted_majority_voting_DB_index"] = reduced_db_index
        results["reduced_fitness_weighted_majority_voting_silhouette_index"] = reduced_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            reduced_fitness_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(reduced_fitness_weighted_majority_voting)
        results["reduced_fitness_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["reduced_fitness_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['fitness_rank_weighted_majority_voting']:
        activated_features = np.zeros(len(population[0]), dtype=int)
        total_votes = 0
        for idx, candidate in enumerate(population):
            rank = population_kfold_fitness[population_kfold_fitness <= population_kfold_fitness[idx]].shape[0]
            activated_features += rank * np.array(candidate).astype(int)
            total_votes += rank

        reduced_fitness_rank_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        feature_vector = np.array(reduced_fitness_rank_weighted_majority_voting).astype(
            bool)  # bool array from integer
        if any(feature_vector):
            reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            reduced_validation_fitness = 0
            reduced_test_CAC = 0
            reduced_test_F1 = 0

        reduced_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        reduced_silhouette_index = get_silhouette_score(X, y, feature_vector)

        reduced_fitness_rank_weighted_majority_voting = reduced_fitness_rank_weighted_majority_voting * 1
        results["reduced_fitness_rank_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in reduced_fitness_rank_weighted_majority_voting])
        results["reduced_fitness_rank_weighted_majority_voting_validation_fitness"] = reduced_validation_fitness
        results["reduced_fitness_rank_weighted_majority_voting_test_CAC"] = reduced_test_CAC
        results["reduced_fitness_rank_weighted_majority_voting_test_F1"] = reduced_test_F1
        results["reduced_fitness_rank_weighted_majority_voting_size"] = np.sum(
            reduced_fitness_rank_weighted_majority_voting)
        results["reduced_fitness_rank_weighted_majority_voting_DB_index"] = reduced_db_index
        results["reduced_fitness_rank_weighted_majority_voting_silhouette_index"] = reduced_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            reduced_fitness_rank_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(reduced_fitness_rank_weighted_majority_voting)
        results["reduced_fitness_rank_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["reduced_fitness_rank_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['size_weighted_majority_voting']:
        activated_features = np.zeros(len(population[0]), dtype=int)
        total_votes = 0
        for idx, candidate in enumerate(population):
            rank = len(population[np.sum(population, axis=1) >= np.sum(population[idx])])
            activated_features += rank * np.array(candidate).astype(int)
            total_votes += rank

        reduced_size_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        feature_vector = np.array(reduced_size_weighted_majority_voting).astype(bool)  # bool array from integer
        if any(feature_vector):
            reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            reduced_validation_fitness = 0
            reduced_test_CAC = 0
            reduced_test_F1 = 0

        reduced_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        reduced_silhouette_index = get_silhouette_score(X, y, feature_vector)

        reduced_size_weighted_majority_voting = reduced_size_weighted_majority_voting * 1
        results["reduced_size_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in reduced_size_weighted_majority_voting])
        results["reduced_size_weighted_majority_voting_validation_fitness"] = reduced_validation_fitness
        results["reduced_size_weighted_majority_voting_test_CAC"] = reduced_test_CAC
        results["reduced_size_weighted_majority_voting_test_F1"] = reduced_test_F1
        results["reduced_size_weighted_majority_voting_size"] = np.sum(reduced_size_weighted_majority_voting)
        results["reduced_size_weighted_majority_voting_DB_index"] = reduced_db_index
        results["reduced_size_weighted_majority_voting_silhouette_index"] = reduced_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            reduced_size_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(reduced_size_weighted_majority_voting)
        results["reduced_size_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["reduced_size_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['fitness_size_rank_weighted_majority_voting_solutions']:
        activated_features = np.zeros(len(population[0]), dtype=int)
        total_votes = 0
        for idx, candidate in enumerate(population):
            rank = population_kfold_fitness[population_kfold_fitness <= population_kfold_fitness[idx]].shape[0]
            rank += len(population[np.sum(population, axis=1) >= np.sum(population[idx])])
            activated_features += rank * np.array(candidate).astype(int)
            total_votes += rank

        reduced_fitness_size_rank_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        feature_vector = np.array(reduced_fitness_size_rank_weighted_majority_voting).astype(
            bool)  # bool array from integer
        if any(feature_vector):
            reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            reduced_validation_fitness = 0
            reduced_test_CAC = 0
            reduced_test_F1 = 0

        reduced_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        reduced_silhouette_index = get_silhouette_score(X, y, feature_vector)

        reduced_fitness_size_rank_weighted_majority_voting = reduced_fitness_size_rank_weighted_majority_voting * 1
        results["reduced_fitness_size_rank_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in reduced_fitness_size_rank_weighted_majority_voting])
        results[
            "reduced_fitness_size_rank_weighted_majority_voting_validation_fitness"] = reduced_validation_fitness
        results["reduced_fitness_size_rank_weighted_majority_voting_test_CAC"] = reduced_test_CAC
        results["reduced_fitness_size_rank_weighted_majority_voting_test_F1"] = reduced_test_F1
        results["reduced_fitness_size_rank_weighted_majority_voting_size"] = np.sum(
            reduced_fitness_size_rank_weighted_majority_voting)
        results["reduced_fitness_size_rank_weighted_majority_voting_DB_index"] = reduced_db_index
        results["reduced_fitness_size_rank_weighted_majority_voting_silhouette_index"] = reduced_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            reduced_fitness_size_rank_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(reduced_fitness_size_rank_weighted_majority_voting)
        results["reduced_fitness_size_rank_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["reduced_fitness_size_rank_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['fitness_size_weighted_majority_voting_solutions']:
        activated_features = np.zeros(len(population[0]), dtype=float)
        total_votes = 0
        for idx, candidate in enumerate(population):
            weight = population_kfold_fitness[idx] + ((len(candidate) - np.sum(candidate)) / len(candidate))
            activated_features += weight * np.array(candidate).astype(int)
            total_votes += weight

        reduced_fitness_size_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        feature_vector = np.array(reduced_fitness_size_weighted_majority_voting).astype(
            bool)  # bool array from integer
        if any(feature_vector):
            reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            reduced_validation_fitness = 0
            reduced_test_CAC = 0
            reduced_test_F1 = 0

        reduced_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        reduced_silhouette_index = get_silhouette_score(X, y, feature_vector)

        reduced_fitness_size_weighted_majority_voting = reduced_fitness_size_weighted_majority_voting * 1
        results["reduced_fitness_size_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in reduced_fitness_size_weighted_majority_voting])
        results["reduced_fitness_size_weighted_majority_voting_validation_fitness"] = reduced_validation_fitness
        results["reduced_fitness_size_weighted_majority_voting_test_CAC"] = reduced_test_CAC
        results["reduced_fitness_size_weighted_majority_voting_test_F1"] = reduced_test_F1
        results["reduced_fitness_size_weighted_majority_voting_size"] = np.sum(
            reduced_fitness_size_weighted_majority_voting)
        results["reduced_fitness_size_weighted_majority_voting_DB_index"] = reduced_db_index
        results["reduced_fitness_size_weighted_majority_voting_silhouette_index"] = reduced_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            reduced_fitness_size_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(reduced_fitness_size_weighted_majority_voting)
        results["reduced_fitness_size_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["reduced_fitness_size_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['db_rank_weighted_majority_voting_solutions']:
        activated_features = np.zeros(len(population[0]), dtype=int)
        total_votes = 0
        for idx, candidate in enumerate(population):
            rank = population_clustering_index[population_clustering_index >= population_clustering_index[idx]].shape[0]
            activated_features += rank * np.array(candidate).astype(int)
            total_votes += rank

        reduced_db_rank_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        feature_vector = np.array(reduced_db_rank_weighted_majority_voting).astype(
            bool)  # bool array from integer
        if any(feature_vector):
            reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            reduced_validation_fitness = 0
            reduced_test_CAC = 0
            reduced_test_F1 = 0

        reduced_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        reduced_silhouette_index = get_silhouette_score(X, y, feature_vector)

        reduced_db_rank_weighted_majority_voting = reduced_db_rank_weighted_majority_voting * 1
        results["reduced_db_rank_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in reduced_db_rank_weighted_majority_voting])
        results["reduced_db_rank_weighted_majority_voting_validation_fitness"] = reduced_validation_fitness
        results["reduced_db_rank_weighted_majority_voting_test_CAC"] = reduced_test_CAC
        results["reduced_db_rank_weighted_majority_voting_test_F1"] = reduced_test_F1
        results["reduced_db_rank_weighted_majority_voting_size"] = np.sum(
            reduced_db_rank_weighted_majority_voting)
        results["reduced_db_rank_weighted_majority_voting_DB_index"] = reduced_db_index
        results["reduced_db_rank_weighted_majority_voting_silhouette_index"] = reduced_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            reduced_db_rank_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(reduced_db_rank_weighted_majority_voting)
        results["reduced_db_rank_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["reduced_db_rank_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['db_weighted_majority_voting_solutions']:
        activated_features = np.zeros(len(population[0]), dtype=float)
        total_votes = 0
        for idx, candidate in enumerate(population):
            activated_features += (1 / population_clustering_index[idx]) * np.array(candidate).astype(int)
            total_votes += (1 / population_clustering_index[idx])

        reduced_db_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

        feature_vector = np.array(reduced_db_weighted_majority_voting).astype(
            bool)  # bool array from integer
        if any(feature_vector):
            reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            reduced_validation_fitness = 0
            reduced_test_CAC = 0
            reduced_test_F1 = 0

        reduced_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        reduced_silhouette_index = get_silhouette_score(X, y, feature_vector)

        reduced_db_weighted_majority_voting = reduced_db_weighted_majority_voting * 1
        results["reduced_db_weighted_majority_voting"] = ','.join(
            [str(feature) for feature in reduced_db_weighted_majority_voting])
        results["reduced_db_weighted_majority_voting_validation_fitness"] = reduced_validation_fitness
        results["reduced_db_weighted_majority_voting_test_CAC"] = reduced_test_CAC
        results["reduced_db_weighted_majority_voting_test_F1"] = reduced_test_F1
        results["reduced_db_weighted_majority_voting_size"] = np.sum(
            reduced_db_weighted_majority_voting)
        results["reduced_db_weighted_majority_voting_DB_index"] = reduced_db_index
        results["reduced_db_weighted_majority_voting_silhouette_index"] = reduced_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            reduced_db_weighted_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(reduced_db_weighted_majority_voting)
        results["reduced_db_weighted_majority_voting_val_sensitivity"] = val_sensitivity
        results["reduced_db_weighted_majority_voting_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['pairwise_feature_frequency_majority_voting_solutions']:
        pairwise_feature_frequency_matrix = get_feature_pairwise_frequency_matrix(population)
        activated_features = np.sum(pairwise_feature_frequency_matrix, axis=0)
        total_votes = np.sum(activated_features)

        reduced_pairwise_feature_frequency_majority_voting = (activated_features >= total_votes / dimensionality) * 1

        feature_vector = np.array(reduced_pairwise_feature_frequency_majority_voting).astype(
            bool)  # bool array from integer
        if any(feature_vector):
            reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            reduced_validation_fitness = 0
            reduced_test_CAC = 0
            reduced_test_F1 = 0

        reduced_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        reduced_silhouette_index = get_silhouette_score(X, y, feature_vector)

        reduced_pairwise_feature_frequency_majority_voting = reduced_pairwise_feature_frequency_majority_voting * 1
        results["reduced_pairwise_feature_frequency_majority_voting"] = ','.join(
            [str(feature) for feature in reduced_pairwise_feature_frequency_majority_voting])
        results[
            "reduced_pairwise_feature_frequency_majority_voting_validation_fitness"] = reduced_validation_fitness
        results["reduced_pairwise_feature_frequency_majority_voting_test_CAC"] = reduced_test_CAC
        results["reduced_pairwise_feature_frequency_majority_voting_test_F1"] = reduced_test_F1
        results["reduced_pairwise_feature_frequency_majority_voting_size"] = np.sum(
            reduced_pairwise_feature_frequency_majority_voting)
        results["reduced_pairwise_feature_frequency_majority_voting_DB_index"] = reduced_db_index
        results["reduced_pairwise_feature_frequency_majority_voting_silhouette_index"] = reduced_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            reduced_pairwise_feature_frequency_majority_voting)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(reduced_pairwise_feature_frequency_majority_voting)
        results["reduced_pairwise_feature_frequency_majority_voting_val_sensitivity"] = val_sensitivity
        results["reduced_pairwise_feature_frequency_majority_voting_kfold_fitness"] = kfold_fitness

    reduced_solution_union = np.array([False] * len(population[0]), dtype=bool)
    reduced_solution_intersection = np.array([True] * len(population[0]), dtype=bool)
    for solution in population:
        reduced_solution_intersection = reduced_solution_intersection * np.array(solution).astype(bool)
        reduced_solution_union = reduced_solution_union + np.array(solution).astype(bool)

    if aggregation_principles_to_evaluate['union']:
        feature_vector = np.array(reduced_solution_union).astype(bool)  # bool array from integer
        reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)

        reduced_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        reduced_silhouette_index = get_silhouette_score(X, y, feature_vector)

        reduced_solution_union = reduced_solution_union * 1
        results["reduced_solution_union"] = ','.join([str(feature) for feature in reduced_solution_union])
        results["reduced_solution_union_validation_fitness"] = reduced_validation_fitness
        results["reduced_solution_union_test_CAC"] = reduced_test_CAC
        results["reduced_solution_union_test_F1"] = reduced_test_F1
        results["reduced_solution_union_size"] = np.sum(reduced_solution_union)
        results["reduced_solution_union_DB_index"] = reduced_db_index
        results["reduced_solution_union_index"] = reduced_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            reduced_solution_union)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(reduced_solution_union)
        results["reduced_solution_union_val_sensitivity"] = val_sensitivity
        results["reduced_solution_union_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['intersection']:
        feature_vector = np.array(reduced_solution_intersection).astype(bool)  # bool array from integer
        if any(feature_vector):
            reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            reduced_validation_fitness = 0
            reduced_test_CAC = 0
            reduced_test_F1 = 0

        reduced_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        reduced_silhouette_index = get_silhouette_score(X, y, feature_vector)

        reduced_solution_intersection = reduced_solution_intersection * 1
        results["reduced_solution_intersection"] = ','.join(
            [str(feature) for feature in reduced_solution_intersection])
        results["reduced_solution_intersection_validation_fitness"] = reduced_validation_fitness
        results["reduced_solution_intersection_test_CAC"] = reduced_test_CAC
        results["reduced_solution_intersection_test_F1"] = reduced_test_F1
        results["reduced_solution_intersection_size"] = np.sum(reduced_solution_intersection)
        results["reduced_solution_intersection_DB_index"] = reduced_db_index
        results["reduced_solution_intersection_index"] = reduced_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            reduced_solution_intersection)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(reduced_solution_intersection)
        results["reduced_solution_intersection_val_sensitivity"] = val_sensitivity
        results["reduced_solution_intersection_kfold_fitness"] = kfold_fitness

    if aggregation_principles_to_evaluate['multiintersection']:
        intersections = []
        for candidate1 in population:
            for candidate2 in population:
                if not np.array_equal(candidate1, candidate2):
                    intersections.append(np.array(candidate1).astype(bool) * np.array(candidate2).astype(bool))

        reduced_solution_multiintersection = np.array([False] * len(population[0]), dtype=bool)

        activated_features = np.sum(intersections, axis=0)
        activated_features = activated_features >= (len(intersections) // 2)
        reduced_solution_multiintersection = activated_features * 1

        # for intersect in intersections:
        #     reduced_solution_multiintersection = reduced_solution_multiintersection + np.array(intersect).astype(bool)

        feature_vector = np.array(reduced_solution_multiintersection).astype(bool)  # bool array from integer
        if any(feature_vector):
            reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
            reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
            reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
        else:
            reduced_validation_fitness = 0
            reduced_test_CAC = 0
            reduced_test_F1 = 0

        reduced_db_index = get_Davies_Bouldin_score(X, y, feature_vector)
        reduced_silhouette_index = get_silhouette_score(X, y, feature_vector)

        reduced_solution_multiintersection = reduced_solution_multiintersection * 1
        results["reduced_solution_multiintersection"] = ','.join(
            [str(feature) for feature in reduced_solution_multiintersection])
        results["reduced_solution_multiintersection_validation_fitness"] = reduced_validation_fitness
        results["reduced_solution_multiintersection_test_CAC"] = reduced_test_CAC
        results["reduced_solution_multiintersection_test_F1"] = reduced_test_F1
        results["reduced_solution_multiintersection_size"] = np.sum(reduced_solution_multiintersection)
        results["reduced_solution_multiintersection_DB_index"] = reduced_db_index
        results["reduced_solution_multiintersection_index"] = reduced_silhouette_index
        val_sensitivity = fitness_function_sensitivity.evaluate_on_validation(
            reduced_solution_multiintersection)
        kfold_fitness = fitness_function_CV.evaluate_on_validation(reduced_solution_multiintersection)
        results["reduced_solution_multiintersection_val_sensitivity"] = val_sensitivity
        results["reduced_solution_multiintersection_kfold_fitness"] = kfold_fitness

    print(results)
    return results


def get_stability_report(summary_results, dataset, stability_function):
    stability_report = {}
    stability_report["dataset"] = dataset
    best_solutions = []
    majority_voting_solutions = []
    minority_voting_solutions = []
    large_majority_voting_solutions = []
    fitness_weighted_majority_voting_solutions = []
    fitness_rank_weighted_majority_voting_solutions = []
    size_weighted_majority_voting_solutions = []
    fitness_size_rank_weighted_majority_voting_solutions = []
    fitness_size_weighted_majority_voting_solutions = []
    db_weighted_majority_voting_solutions = []
    db_rank_weighted_majority_voting_solutions = []
    pairwise_feature_frequency_majority_voting_solutions = []
    union_solutions = []
    intersection_solutions = []
    multiintersection_solutions = []

    coalition_majority_voting_solutions = []
    coalition_minority_voting_solutions = []
    coalition_large_majority_voting_solutions = []
    coalition_fitness_weighted_majority_voting_solutions = []
    coalition_fitness_rank_weighted_majority_voting_solutions = []
    coalition_size_weighted_majority_voting_solutions = []
    coalition_fitness_size_rank_weighted_majority_voting_solutions = []
    coalition_fitness_size_weighted_majority_voting_solutions = []
    coalition_db_weighted_majority_voting_solutions = []
    coalition_db_rank_weighted_majority_voting_solutions = []
    coalition_pairwise_feature_frequency_majority_voting_solutions = []
    coalition_union_solutions = []
    coalition_intersection_solutions = []
    coalition_multiintersection_solutions = []

    reduced_majority_voting_solutions = []
    reduced_minority_voting_solutions = []
    reduced_large_majority_voting_solutions = []
    reduced_fitness_weighted_majority_voting_solutions = []
    reduced_fitness_rank_weighted_majority_voting_solutions = []
    reduced_size_weighted_majority_voting_solutions = []
    reduced_fitness_size_rank_weighted_majority_voting_solutions = []
    reduced_fitness_size_weighted_majority_voting_solutions = []
    reduced_db_weighted_majority_voting_solutions = []
    reduced_db_rank_weighted_majority_voting_solutions = []
    reduced_pairwise_feature_frequency_majority_voting_solutions = []
    reduced_union_solutions = []
    reduced_intersection_solutions = []
    reduced_multiintersection_solutions = []

    for run_summary in summary_results:
        if run_summary["dataset"] == dataset:
            best_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in run_summary["best_solution"].split(',')]).astype(
                    bool))
            stability_report["best_solution"] = stability_function(best_solutions)

            if aggregation_principles_to_evaluate['union']:
                union_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["union_solution"].split(',')]).astype(
                        bool))
                coalition_union_solutions.append(
                    np.array([True if feature == '1' else False for feature in
                              run_summary["coalition_solution_union"].split(',')]).astype(bool))
                reduced_union_solutions.append(
                    np.array([True if feature == '1' else False for feature in
                              run_summary["reduced_solution_union"].split(',')]).astype(bool))

                stability_report["union_solution"] = stability_function(union_solutions)
                stability_report["coalition_union_solution"] = stability_function(coalition_union_solutions)
                stability_report["reduced_union_solution"] = stability_function(reduced_union_solutions)

            if aggregation_principles_to_evaluate['intersection']:
                intersection_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["intersection_solution"].split(',')]).astype(
                        bool))
                coalition_intersection_solutions.append(
                    np.array([True if feature == '1' else False for feature in
                              run_summary["coalition_solution_intersection"].split(',')]).astype(bool))
                reduced_intersection_solutions.append(
                    np.array([True if feature == '1' else False for feature in
                              run_summary["reduced_solution_intersection"].split(',')]).astype(bool))

                stability_report["intersection_solution"] = stability_function(intersection_solutions)
                stability_report["coalition_intersection_solution"] = stability_function(
                    coalition_intersection_solutions)
                stability_report["reduced_intersection_solution"] = stability_function(reduced_intersection_solutions)

            if aggregation_principles_to_evaluate['multiintersection']:
                multiintersection_solutions.append(
                    np.array([True if feature == '1' else False for feature in
                              run_summary["multiintersection_solution"].split(',')]).astype(bool))

                coalition_multiintersection_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["coalition_solution_multiintersection"].split(',')]).astype(
                        bool))

                reduced_multiintersection_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["reduced_solution_multiintersection"].split(',')]).astype(
                        bool))

                stability_report["multiintersection_solution"] = stability_function(multiintersection_solutions)
                stability_report["coalition_multiintersection_solution"] = stability_function(
                    coalition_multiintersection_solutions)
                stability_report["reduced_multiintersection_solution"] = stability_function(
                    reduced_multiintersection_solutions)

            if aggregation_principles_to_evaluate['majority_voting']:
                majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["majority_voting"].split(',')]).astype(
                        bool))
                coalition_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["coalition_solution_majority_voting"].split(',')]).astype(
                        bool))
                reduced_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["reduced_solution_majority_voting"].split(',')]).astype(
                        bool))
                stability_report["majority_voting"] = stability_function(majority_voting_solutions)
                stability_report["coalition_majority_voting"] = stability_function(coalition_majority_voting_solutions)
                stability_report["reduced_majority_voting"] = stability_function(reduced_majority_voting_solutions)

            if aggregation_principles_to_evaluate['minority_voting']:
                minority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["minority_voting"].split(',')]).astype(
                        bool))
                coalition_minority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["coalition_solution_minority_voting"].split(',')]).astype(
                        bool))
                reduced_minority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["reduced_solution_minority_voting"].split(',')]).astype(
                        bool))
                stability_report["minority_voting"] = stability_function(minority_voting_solutions)
                stability_report["coalition_minority_voting"] = stability_function(coalition_minority_voting_solutions)
                stability_report["reduced_minority_voting"] = stability_function(reduced_minority_voting_solutions)

            if aggregation_principles_to_evaluate['large_majority_voting']:
                large_majority_voting_solutions.append(
                    np.array([True if feature == '1' else False for feature in
                              run_summary["large_majority_voting"].split(',')]).astype(bool))

                coalition_large_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["coalition_solution_large_majority_voting"].split(',')]).astype(
                        bool))
                reduced_large_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["reduced_solution_large_majority_voting"].split(',')]).astype(
                        bool))

                stability_report["large_majority_voting"] = stability_function(large_majority_voting_solutions)
                stability_report["coalition_large_majority_voting"] = stability_function(
                    coalition_large_majority_voting_solutions)
                stability_report["reduced_large_majority_voting"] = stability_function(
                    reduced_large_majority_voting_solutions)

            if aggregation_principles_to_evaluate['fitness_weighted_majority_voting']:
                fitness_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["fitness_weighted_majority_voting"].split(',')]).astype(
                        bool))

                coalition_fitness_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["coalition_fitness_weighted_majority_voting"].split(',')]).astype(
                        bool))

                reduced_fitness_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["reduced_fitness_weighted_majority_voting"].split(',')]).astype(
                        bool))

                stability_report["fitness_weighted_majority_voting"] = stability_function(
                    fitness_weighted_majority_voting_solutions)
                stability_report["coalition_fitness_weighted_majority_voting"] = stability_function(
                    coalition_fitness_weighted_majority_voting_solutions)
                stability_report["reduced_fitness_weighted_majority_voting"] = stability_function(
                    reduced_fitness_weighted_majority_voting_solutions)

            if aggregation_principles_to_evaluate['fitness_rank_weighted_majority_voting']:
                fitness_rank_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["fitness_rank_weighted_majority_voting"].split(',')]).astype(
                        bool))

                coalition_fitness_rank_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["coalition_fitness_rank_weighted_majority_voting"].split(',')]).astype(
                        bool))

                reduced_fitness_rank_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["reduced_fitness_rank_weighted_majority_voting"].split(',')]).astype(
                        bool))

                stability_report["fitness_rank_weighted_majority_voting"] = stability_function(
                    fitness_rank_weighted_majority_voting_solutions)
                stability_report["coalition_fitness_rank_weighted_majority_voting"] = stability_function(
                    coalition_fitness_rank_weighted_majority_voting_solutions)
                stability_report["reduced_fitness_rank_weighted_majority_voting"] = stability_function(
                    reduced_fitness_rank_weighted_majority_voting_solutions)

            if aggregation_principles_to_evaluate['size_weighted_majority_voting']:
                size_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["size_weighted_majority_voting"].split(',')]).astype(
                        bool))

                coalition_size_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["coalition_size_weighted_majority_voting"].split(',')]).astype(
                        bool))

                reduced_size_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["reduced_size_weighted_majority_voting"].split(',')]).astype(
                        bool))

                stability_report["size_weighted_majority_voting"] = stability_function(
                    size_weighted_majority_voting_solutions)
                stability_report["coalition_size_weighted_majority_voting"] = stability_function(
                    coalition_size_weighted_majority_voting_solutions)
                stability_report["reduced_size_weighted_majority_voting"] = stability_function(
                    reduced_size_weighted_majority_voting_solutions)

            if aggregation_principles_to_evaluate['fitness_size_rank_weighted_majority_voting_solutions']:
                fitness_size_rank_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["fitness_size_rank_weighted_majority_voting"].split(',')]).astype(
                        bool))
                coalition_fitness_size_rank_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["coalition_fitness_size_rank_weighted_majority_voting"].split(',')]).astype(
                        bool))

                reduced_fitness_size_rank_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["reduced_fitness_size_rank_weighted_majority_voting"].split(',')]).astype(
                        bool))
                stability_report["fitness_size_rank_weighted_majority_voting"] = stability_function(
                    fitness_size_rank_weighted_majority_voting_solutions)
                stability_report["coalition_fitness_size_rank_weighted_majority_voting"] = stability_function(
                    coalition_fitness_size_rank_weighted_majority_voting_solutions)
                stability_report["reduced_fitness_size_rank_weighted_majority_voting"] = stability_function(
                    reduced_fitness_size_rank_weighted_majority_voting_solutions)

            if aggregation_principles_to_evaluate['fitness_size_weighted_majority_voting_solutions']:
                fitness_size_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["fitness_size_weighted_majority_voting"].split(',')]).astype(
                        bool))

                coalition_fitness_size_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["coalition_fitness_size_weighted_majority_voting"].split(',')]).astype(
                        bool))

                reduced_fitness_size_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["reduced_fitness_size_weighted_majority_voting"].split(',')]).astype(
                        bool))
                stability_report["fitness_size_weighted_majority_voting"] = stability_function(
                    fitness_size_weighted_majority_voting_solutions)
                stability_report["coalition_fitness_size_weighted_majority_voting"] = stability_function(
                    coalition_fitness_size_weighted_majority_voting_solutions)
                stability_report["reduced_fitness_size_weighted_majority_voting"] = stability_function(
                    reduced_fitness_size_weighted_majority_voting_solutions)

            if aggregation_principles_to_evaluate['db_rank_weighted_majority_voting_solutions']:
                db_rank_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["db_rank_weighted_majority_voting"].split(',')]).astype(
                        bool))

                coalition_db_rank_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["coalition_db_rank_weighted_majority_voting"].split(',')]).astype(
                        bool))

                reduced_db_rank_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["reduced_db_rank_weighted_majority_voting"].split(',')]).astype(
                        bool))

                stability_report["db_rank_weighted_majority_voting"] = stability_function(
                    db_rank_weighted_majority_voting_solutions)
                stability_report["coalition_db_rank_weighted_majority_voting"] = stability_function(
                    coalition_db_rank_weighted_majority_voting_solutions)
                stability_report["reduced_db_rank_weighted_majority_voting"] = stability_function(
                    reduced_db_rank_weighted_majority_voting_solutions)

            if aggregation_principles_to_evaluate['db_weighted_majority_voting_solutions']:
                db_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["db_weighted_majority_voting"].split(',')]).astype(
                        bool))

                coalition_db_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["coalition_db_weighted_majority_voting"].split(',')]).astype(
                        bool))

                reduced_db_weighted_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["reduced_db_weighted_majority_voting"].split(',')]).astype(
                        bool))

                stability_report["db_weighted_majority_voting"] = stability_function(
                    db_weighted_majority_voting_solutions)
                stability_report["coalition_db_weighted_majority_voting"] = stability_function(
                    coalition_db_weighted_majority_voting_solutions)
                stability_report["reduced_db_weighted_majority_voting"] = stability_function(
                    reduced_db_weighted_majority_voting_solutions)
            if aggregation_principles_to_evaluate['pairwise_feature_frequency_majority_voting_solutions']:
                pairwise_feature_frequency_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["pairwise_feature_frequency_majority_voting"].split(',')]).astype(
                        bool))

                coalition_pairwise_feature_frequency_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["coalition_pairwise_feature_frequency_majority_voting"].split(',')]).astype(
                        bool))

                reduced_pairwise_feature_frequency_majority_voting_solutions.append(
                    np.array(
                        [True if feature == '1' else False for feature in
                         run_summary["reduced_pairwise_feature_frequency_majority_voting"].split(',')]).astype(
                        bool))
                stability_report["pairwise_feature_frequency_majority_voting"] = stability_function(
                    pairwise_feature_frequency_majority_voting_solutions)
                stability_report["coalition_pairwise_feature_frequency_majority_voting"] = stability_function(
                    coalition_pairwise_feature_frequency_majority_voting_solutions)
                stability_report["reduced_pairwise_feature_frequency_majority_voting"] = stability_function(
                    reduced_pairwise_feature_frequency_majority_voting_solutions)

    print(stability_report)
    return stability_report


for optimiser in fs_wrappers:
    for procedure in evaluation_procedures:
        for classifier in classifiers:
            summary_results[optimiser.name + procedure + classifier.name] = []
            asm_stability_results[optimiser.name + procedure + classifier.name] = []
            anhd_stability_results[optimiser.name + procedure + classifier.name] = []
            for file in datasets:
                dataset = pd.read_csv(file.path, header=0, index_col=0)
                data = dataset.iloc[:, :-1].values
                target = dataset.iloc[:, -1]
                dimensionality = data.shape[1]

                X_rest, X_test, y_rest, y_test = train_test_split(data, target, test_size=experiment_setup["test_size"],
                                                                  random_state=42,
                                                                  shuffle=True,
                                                                  stratify=target)

                feature_scores = calculate_chi2(X_rest, y_rest)

                population_quality_filename = '_'.join(
                    [procedure, optimiser.name, classifier.name, file.name])
                print("Processing file {0}".format(population_quality_filename))

                val_header, val_data = CsvProcessor().read_file(
                    filename='logs/populationQuality/' + population_quality_filename)
                test_header, test_data = CsvProcessor().read_file(
                    filename='logs/outputQuality/' + population_quality_filename)

                search_results = []
                if val_header is not None and val_data is not None and test_header is not None and test_data is not None:
                    val_data = [row for row in val_data if row]
                    test_data = [row for row in test_data if row]

                    for idx, data in enumerate(val_data):
                        if idx % 14 == 13:  # 13 jer se gleda best solution na kraju runa
                            search_results.append(
                                [float(data[0]), float(test_data[idx // 14][0]), float(test_data[idx // 14][1]),
                                 test_data[idx // 14][-1]])  # 0 je f1:macro, 1 CAC

                filename = '_'.join(
                    [procedure, optimiser.name + "_", classifier.name, file.name,
                     'archive'])
                print("Processing file {0}".format(filename))
                test_header, test_data = CsvProcessor().read_file(
                    filename='logs/archive/' + filename + "_test")

                if test_header is not None and test_data is not None:
                    test_data = [row for row in test_data if row]
                    test_data = np.array(test_data)

                if test_header is not None and test_data is not None:
                    run = 0
                    index = 0
                    population = np.empty([population_size, dimensionality], dtype=bool)
                    population_fitness = np.zeros(population_size)
                    population_test_fitness = np.zeros(population_size)
                    population_kfold_fitness = np.zeros(population_size)
                    population_clustering_index = np.zeros(population_size, dtype=float)

                    best_test_fitness = 0
                    best_test_db_index = 0
                    best_test_silhouette_index = 0
                    fitness_function = ClassificationProblem(file, classifier,
                                                             random_state=42,  # dodati run + za variable
                                                             test_size=experiment_setup["test_size"],
                                                             validation_size=experiment_setup[
                                                                 "validation_size"],
                                                             wrapper_fitness_metric=fitness,
                                                             metrics=classification_metrics)

                    fitness_function_CV = ClassificationProblemCV(file, classifier,
                                                                  random_state=42,  # dodati run + za variable
                                                                  test_size=experiment_setup["test_size"],
                                                                  k_folds=5,
                                                                  wrapper_fitness_metric=fitness,
                                                                  metrics=classification_metrics)
                    for idx, candidate in enumerate(test_data):
                        if idx > 0 and idx % population_size == 0:
                            summary_results[optimiser.name + procedure + classifier.name].append(
                                feature_subset_aggregation(file.name, X_rest, y_rest, run, copy.deepcopy(population),
                                                           copy.deepcopy(population_fitness),
                                                           copy.deepcopy(population_kfold_fitness),
                                                           copy.deepcopy(population_clustering_index), search_results,
                                                           best_test_fitness, feature_scores))

                            for i in range(population_size):
                                population[i] = population[i].astype(int) * 1
                                data = [file.name, run, ','.join([str(feature * 1) for feature in population[i]]),
                                        population_fitness[i], population_kfold_fitness[i], population_test_fitness[i],
                                        population_clustering_index[i], best_test_fitness, best_test_db_index,
                                        best_test_silhouette_index]
                                CsvProcessor().save_summary_results(filename='_'.join(
                                    [procedure, optimiser.name, classifier.name,
                                     "_clustering_classification_performance"]),
                                    header=['dataset', 'run', 'solution', fitness, fitness + "_5fold",
                                            fitness + "_test", 'davies-bouldin', 'best_test_fitness',
                                            'best_test_fitness_db', 'best_test_fitness_silhouette'],
                                    data=data)

                            run += 1
                            index = 0
                            population = np.empty([population_size, dimensionality], dtype=bool)
                            population_fitness = np.zeros(population_size, dtype=float)
                            population_test_fitness = np.zeros(population_size, dtype=float)
                            population_kfold_fitness = np.zeros(population_size)
                            fitness_function = ClassificationProblem(file, classifier,
                                                                     random_state=42,  # dodati run + za variable
                                                                     test_size=experiment_setup["test_size"],
                                                                     validation_size=experiment_setup[
                                                                         "validation_size"],
                                                                     wrapper_fitness_metric=fitness,
                                                                     metrics=classification_metrics)

                            fitness_function_CV = ClassificationProblemCV(file, classifier,
                                                                          random_state=42,  # dodati run + za variable
                                                                          test_size=experiment_setup["test_size"],
                                                                          k_folds=5,
                                                                          wrapper_fitness_metric=fitness,
                                                                          metrics=classification_metrics)
                            best_test_fitness = 0
                            best_test_db_index = 0
                            best_test_silhouette_index = 0

                        population[index] = [True if bit == '1' else False for bit in
                                             candidate[header_idx['vector_val']]]
                        population_clustering_index[index] = get_Davies_Bouldin_score(X_rest, y_rest,
                                                                                      population[index])

                        # population_fitness[index] = fitness_function.evaluate_on_validation(
                        #     np.array(population[index]).astype(bool))
                        # population_test_fitness[index] = fitness_function.evaluate_on_test(
                        #     np.array(population[index]).astype(bool))

                        population_fitness[index] = float(candidate[header_idx[fitness + "_val"]])
                        population_test_fitness[index] = float(candidate[header_idx[fitness + "_test"] + 1])
                        population_kfold_fitness[index] = fitness_function_CV.evaluate_on_validation(population[index])

                        if population_test_fitness[index] > best_test_fitness:
                            best_test_fitness = population_test_fitness[index]
                            best_test_db_index = get_Davies_Bouldin_score(X_rest, y_rest, population[index])
                            best_test_silhouette_index = get_silhouette_score(X_rest, y_rest, population[index])
                        index += 1

                        if idx == len(test_data) - 1:
                            summary_results[optimiser.name + procedure + classifier.name].append(
                                feature_subset_aggregation(file.name, X_rest, y_rest, run, copy.deepcopy(population),
                                                           copy.deepcopy(population_fitness),
                                                           copy.deepcopy(population_kfold_fitness),
                                                           population_clustering_index,
                                                           search_results,
                                                           best_test_fitness, feature_scores))

                    asm_stability_results[optimiser.name + procedure + classifier.name].append(
                        get_stability_report(
                            summary_results[optimiser.name + procedure + classifier.name], file.name, calculate_ASM))

                    # anhd_stability_results[optimiser.name + procedure + classifier.name].append(
                    #     get_stability_report(
                    #         summary_results[optimiser.name + procedure + classifier.name], file.name,
                    #         calculate_ANHD))

            for row in summary_results[optimiser.name + procedure + classifier.name]:
                CsvProcessor().save_summary_results(filename='_'.join(
                    [procedure, optimiser.name, classifier.name, "_archive_combinations"]),
                    header=list(row.keys()),
                    data=list(row.values()))

            for row in asm_stability_results[optimiser.name + procedure + classifier.name]:
                CsvProcessor().save_summary_results(filename='_'.join(
                    [procedure, optimiser.name, classifier.name, "_archive_asm_stabilities"]),
                    header=list(row.keys()),
                    data=list(row.values()))

            for row in anhd_stability_results[optimiser.name + procedure + classifier.name]:
                CsvProcessor().save_summary_results(filename='_'.join(
                    [procedure, optimiser.name, classifier.name, "_archive_anhd_stabilities"]),
                    header=list(row.keys()),
                    data=list(row.values()))
