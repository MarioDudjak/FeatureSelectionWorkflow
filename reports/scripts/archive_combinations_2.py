from src.experiment.setup import fitnesses, classifiers, experiment_setup, classification_metrics
from src.features.wrappers import fs_wrappers
from src.models.classification.classificationProblem import ClassificationProblem
from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor

import numpy as np
import pandas as pd
import copy


def get_Hamming_distance(candidate1, candidate2):
    diff = [0 if candidate1[idx] == candidate2[idx] else 1 for idx in range(len(candidate1))]
    return sum(diff)


def calculate_ASM(population):
    for i in range(len(population)):
        population[i] = np.array(population[i]).astype(bool)
    sa = 0
    m = len(population[0])
    for i in range(len(population) - 1):
        for j in range(i + 1, len(population)):
            if (sum(population[i]) == 0 and sum(population[j]) == 0) or (sum(population[i]) == m and sum(population[j]) == m):
                sa += 1
            else:
                s = sum(population[i] * population[j]) - (
                            sum(population[i]) * sum(population[j]) / m)
                s /= (min(sum(population[i]), sum(population[j])) - max(0, sum(population[i]) + sum(population[j]) - m))
                sa += s

    asm = (2 * sa) / (len(population) * (len(population) - 1))

    return asm


def calculate_ANHD(population):
    nhi = 0
    for i in range(len(population) - 1):
        for j in range(i + 1, len(population)):
            hd = get_Hamming_distance(population[i], population[j])
            nhi += 1 - hd / len(population[i])

    anhd = (2 * nhi) / (len(population) * (len(population) - 1))
    return anhd


def feature_subset_aggregation(dataset, run, population, population_fitness, search_results,
                               best_test_fitness):
    population = np.array(population)
    population_fitness = np.array(population_fitness)
    best_solution = ','.join([str(feature) for feature in search_results[run][-1]])
    best_solution_size = np.sum([str(feature) == '1' for feature in search_results[run][-1]])
    results = {"dataset": dataset, "run": run, "best_solution": best_solution, "best_solution_size": best_solution_size,
               "validation_fitness": float(search_results[run][0]),
               "F1_test": float(search_results[run][1]), "CAC_test": float(search_results[run][2]),
               "best_test_fitness": float(best_test_fitness)}

    intersection = np.array([True] * len(population[0]), dtype=bool)
    union = np.array([False] * len(population[0]), dtype=bool)
    multiintersection = np.array([False] * len(population[0]), dtype=bool)
    activated_features = np.sum(population, axis=0)
    activated_features = activated_features >= (len(population) // 2)
    majority_voting = activated_features * 1

    activated_features = np.sum(population, axis=0)
    activated_features = activated_features >= (len(population) // 3)
    minority_voting = activated_features * 1

    activated_features = np.sum(population, axis=0)
    activated_features = activated_features >= (2 * len(population) // 3)
    large_majority_voting = activated_features * 1

    activated_features = np.zeros(len(population[0]), dtype=float)
    total_fitness = 0
    for idx, candidate in enumerate(population):
        activated_features += population_fitness[idx] * np.array(candidate).astype(int)
        total_fitness += population_fitness[idx]

    fitness_weighted_majority_voting = (activated_features >= total_fitness // 2) * 1

    activated_features = np.zeros(len(population[0]), dtype=int)
    total_votes = 0
    for idx, candidate in enumerate(population):
        rank = len(population_fitness[population_fitness <= population_fitness[idx]])
        activated_features += rank * np.array(candidate).astype(int)
        total_votes += rank

    fitness_rank_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

    activated_features = np.zeros(len(population[0]), dtype=int)
    total_votes = 0
    for idx, candidate in enumerate(population):
        rank = len(population[np.sum(population, axis=1) >= np.sum(population[idx])])
        activated_features += rank * np.array(candidate).astype(int)
        total_votes += rank

    size_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

    activated_features = np.zeros(len(population[0]), dtype=int)
    total_votes = 0
    for idx, candidate in enumerate(population):
        rank = len(population_fitness[population_fitness <= population_fitness[idx]])
        rank += len(population[np.sum(population, axis=1) >= np.sum(population[idx])])
        activated_features += rank * np.array(candidate).astype(int)
        total_votes += rank

    fitness_size_rank_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

    # intersections = []
    # for candidate1 in population:
    #     intersections += [np.array(candidate1).astype(bool) * np.array(candidate2).astype(bool) for candidate2 in population if not np.array_equal(candidate1, candidate2)]
    #
    #
    # for intersect in intersections:
    #     multiintersection = multiintersection + np.array(intersect).astype(bool)

    for solution in population:
        intersection = intersection * np.array(solution).astype(bool)
        union = union + np.array(solution).astype(bool)

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

    union_validation_fitness = fitness_function.evaluate_on_validation(union)
    union_test_CAC = fitness_function_CAC.evaluate_on_test(union)
    union_test_F1 = fitness_function_F1.evaluate_on_test(union)

    majority_voting_fitness = fitness_function.evaluate_on_validation(majority_voting)
    majority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(majority_voting)
    majority_voting_test_F1 = fitness_function_F1.evaluate_on_test(majority_voting)

    minority_voting_fitness = fitness_function.evaluate_on_validation(minority_voting)
    minority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(minority_voting)
    minority_voting_test_F1 = fitness_function_F1.evaluate_on_test(minority_voting)
    minority_voting = minority_voting * 1

    large_majority_voting_fitness = fitness_function.evaluate_on_validation(large_majority_voting)
    large_majority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(large_majority_voting)
    large_majority_voting_test_F1 = fitness_function_F1.evaluate_on_test(large_majority_voting)
    large_majority_voting = large_majority_voting * 1

    fitness_weighted_majority_voting_fitness = fitness_function.evaluate_on_validation(
        fitness_weighted_majority_voting)
    fitness_weighted_majority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(
        fitness_weighted_majority_voting)
    fitness_weighted_majority_voting_test_F1 = fitness_function_F1.evaluate_on_test(
        fitness_weighted_majority_voting)

    fitness_rank_weighted_majority_voting_fitness = fitness_function.evaluate_on_validation(
        fitness_rank_weighted_majority_voting)
    fitness_rank_weighted_majority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(
        fitness_rank_weighted_majority_voting)
    fitness_rank_weighted_majority_voting_test_F1 = fitness_function_F1.evaluate_on_test(
        fitness_rank_weighted_majority_voting)

    size_weighted_majority_voting_fitness = fitness_function.evaluate_on_validation(size_weighted_majority_voting)
    size_weighted_majority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(size_weighted_majority_voting)
    size_weighted_majority_voting_test_F1 = fitness_function_F1.evaluate_on_test(size_weighted_majority_voting)

    fitness_size_rank_weighted_majority_voting_fitness = fitness_function.evaluate_on_validation(
        fitness_size_rank_weighted_majority_voting)
    fitness_size_rank_weighted_majority_voting_test_CAC = fitness_function_CAC.evaluate_on_test(
        fitness_size_rank_weighted_majority_voting)
    fitness_size_rank_weighted_majority_voting_test_F1 = fitness_function_F1.evaluate_on_test(
        fitness_size_rank_weighted_majority_voting)

    if any(intersection):
        intersection_validation_fitness = fitness_function.evaluate_on_validation(intersection)
        intersection_test_CAC = fitness_function_CAC.evaluate_on_test(intersection)
        intersection_test_F1 = fitness_function_F1.evaluate_on_test(intersection)
    else:
        intersection_validation_fitness = 0
        intersection_test_CAC = 0
        intersection_test_F1 = 0

    if any(multiintersection):
        multiintersection_validation_fitness = fitness_function.evaluate_on_validation(multiintersection)
        multiintersection_test_CAC = fitness_function_CAC.evaluate_on_test(multiintersection)
        multiintersection_test_F1 = fitness_function_F1.evaluate_on_test(multiintersection)
    else:
        multiintersection_validation_fitness = 0
        multiintersection_test_CAC = 0
        multiintersection_test_F1 = 0

    majority_voting = majority_voting * 1
    results["majority_voting"] = ','.join([str(feature) for feature in majority_voting])
    results["majority_voting_validation_fitness"] = majority_voting_fitness
    results["majority_voting_test_CAC"] = majority_voting_test_CAC
    results["majority_voting_test_F1"] = majority_voting_test_F1
    results["majority_voting_size"] = np.sum(majority_voting)

    results["minority_voting"] = ','.join([str(feature) for feature in minority_voting])
    results["minority_voting_validation_fitness"] = minority_voting_fitness
    results["minority_voting_test_CAC"] = minority_voting_test_CAC
    results["minority_voting_test_F1"] = minority_voting_test_F1
    results["minority_voting_size"] = np.sum(minority_voting)

    results["large_majority_voting"] = ','.join([str(feature) for feature in large_majority_voting])
    results["large_majority_voting_validation_fitness"] = large_majority_voting_fitness
    results["large_majority_voting_test_CAC"] = large_majority_voting_test_CAC
    results["large_majority_voting_test_F1"] = large_majority_voting_test_F1
    results["large_majority_voting_size"] = np.sum(large_majority_voting)

    results["fitness_weighted_majority_voting"] = ','.join(
        [str(feature) for feature in fitness_weighted_majority_voting])
    results["fitness_weighted_majority_voting_validation_fitness"] = fitness_weighted_majority_voting_fitness
    results["fitness_weighted_majority_voting_test_CAC"] = fitness_weighted_majority_voting_test_CAC
    results["fitness_weighted_majority_voting_test_F1"] = fitness_weighted_majority_voting_test_F1
    results["fitness_weighted_majority_voting_size"] = np.sum(fitness_weighted_majority_voting)

    results["fitness_rank_weighted_majority_voting"] = ','.join(
        [str(feature) for feature in fitness_rank_weighted_majority_voting])
    results["fitness_rank_weighted_majority_voting_validation_fitness"] = fitness_rank_weighted_majority_voting_fitness
    results["fitness_rank_weighted_majority_voting_test_CAC"] = fitness_rank_weighted_majority_voting_test_CAC
    results["fitness_rank_weighted_majority_voting_test_F1"] = fitness_rank_weighted_majority_voting_test_F1
    results["fitness_rank_weighted_majority_voting_size"] = np.sum(fitness_rank_weighted_majority_voting)

    results["size_weighted_majority_voting"] = ','.join(
        [str(feature) for feature in size_weighted_majority_voting])
    results["size_weighted_majority_voting_validation_fitness"] = size_weighted_majority_voting_fitness
    results["size_weighted_majority_voting_test_CAC"] = size_weighted_majority_voting_test_CAC
    results["size_weighted_majority_voting_test_F1"] = size_weighted_majority_voting_test_F1
    results["size_weighted_majority_voting_size"] = np.sum(size_weighted_majority_voting)

    results["fitness_size_rank_weighted_majority_voting"] = ','.join(
        [str(feature) for feature in fitness_size_rank_weighted_majority_voting])
    results[
        "fitness_size_rank_weighted_majority_voting_validation_fitness"] = fitness_size_rank_weighted_majority_voting_fitness
    results["fitness_size_rank_weighted_majority_voting_test_CAC"] = fitness_size_rank_weighted_majority_voting_test_CAC
    results["fitness_size_rank_weighted_majority_voting_test_F1"] = fitness_size_rank_weighted_majority_voting_test_F1
    results["fitness_size_rank_weighted_majority_voting_size"] = np.sum(fitness_size_rank_weighted_majority_voting)

    union = union * 1
    results["union_solution"] = ','.join([str(feature) for feature in union])
    results["union_validation_fitness"] = union_validation_fitness
    results["unions_test_CAC"] = union_test_CAC
    results["unions_test_F1"] = union_test_F1
    results["union_size"] = np.sum(union)

    intersection = intersection * 1
    results["intersection_solution"] = ','.join([str(feature) for feature in intersection])
    results["intersection_validation_fitness"] = intersection_validation_fitness
    results["intersection_test_CAC"] = intersection_test_CAC
    results["intersection_test_F1"] = intersection_test_F1
    results["intersection_size"] = np.sum(intersection)

    multiintersection = multiintersection * 1
    results["multiintersection_solution"] = ','.join([str(feature) for feature in multiintersection])
    results["multiintersection_validation_fitness"] = multiintersection_validation_fitness
    results["multiintersection_test_CAC"] = multiintersection_test_CAC
    results["multiintersection_test_F1"] = multiintersection_test_F1
    results["multiintersection_size"] = np.sum(multiintersection)

    coalition = copy.deepcopy(population)
    coalition_fitness = copy.deepcopy(population_fitness)
    partner_solution_indices = [(idx1, idx2) for idx1 in range(len(population)) for idx2 in range(len(population)) if get_Hamming_distance(population[idx1], population[idx2]) == 1]
    for idx1 in range(len(population)):
        neighbour_solutions_indices = [idx2 for idx2 in range(len(population)) if get_Hamming_distance(population[idx1], population[idx2]) == 1]
        best_partner = copy.deepcopy(population[idx1])
        best_partner_fitness = population_fitness[idx1]
        for idx2 in range(len(neighbour_solutions_indices)):
            if population_fitness[idx2] > best_partner_fitness:
                best_partner = copy.deepcopy(population[idx2])
                best_partner_fitness = population_fitness[idx2]

            elif (population_fitness[idx2] == best_partner_fitness) and np.sum(population[idx2]) < np.sum(
                    best_partner):
                best_partner = copy.deepcopy(population[idx2])
                best_partner_fitness = population_fitness[idx2]

        coalition[idx1] = copy.deepcopy(best_partner)
        coalition_fitness[idx1] = best_partner_fitness

    activated_features = np.sum(coalition, axis=0)
    activated_features = activated_features >= (len(coalition) // 2)
    coalition_solution_majority_voting = activated_features * 1

    activated_features = np.sum(coalition, axis=0)
    activated_features = activated_features >= (len(coalition) // 3)
    coalition_solution_minority_voting = activated_features * 1

    activated_features = np.sum(coalition, axis=0)
    activated_features = activated_features >= (2 * len(coalition) // 3)
    coalition_solution_large_majority_voting = activated_features * 1

    activated_features = np.zeros(len(coalition[0]), dtype=float)
    total_fitness = 0
    for idx, candidate in enumerate(coalition):
        activated_features += coalition_fitness[idx] * np.array(candidate).astype(int)
        total_fitness += coalition_fitness[idx]

    coalition_fitness_weighted_majority_voting = (activated_features >= total_fitness // 2) * 1

    activated_features = np.zeros(len(coalition[0]), dtype=int)
    total_votes = 0
    for idx, candidate in enumerate(coalition):
        rank = len(coalition_fitness[coalition_fitness <= coalition_fitness[idx]])
        activated_features += rank * np.array(candidate).astype(int)
        total_votes += rank

    coalition_fitness_rank_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

    activated_features = np.zeros(len(coalition[0]), dtype=int)
    total_votes = 0
    for idx, candidate in enumerate(coalition):
        rank = len(coalition[np.sum(coalition, axis=1) >= np.sum(coalition[idx])])
        activated_features += rank * np.array(candidate).astype(int)
        total_votes += rank

    coalition_size_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

    activated_features = np.zeros(len(coalition[0]), dtype=int)
    total_votes = 0
    for idx, candidate in enumerate(coalition):
        rank = len(coalition_fitness[coalition_fitness <= coalition_fitness[idx]])
        rank += len(coalition[np.sum(coalition, axis=1) >= np.sum(coalition[idx])])
        activated_features += rank * np.array(candidate).astype(int)
        total_votes += rank

    coalition_fitness_size_rank_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

    feature_vector = np.array(coalition_solution_majority_voting).astype(bool)  # bool array from integer
    if any(feature_vector):
        coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        coalition_validation_fitness = 0
        coalition_test_CAC = 0
        coalition_test_F1 = 0

    coalition_solution_majority_voting = coalition_solution_majority_voting * 1
    results["coalition_solution_majority_voting"] = ','.join(
        [str(feature) for feature in coalition_solution_majority_voting])
    results["coalition_solution_majority_voting_validation_fitness"] = coalition_validation_fitness
    results["coalition_solution_majority_voting_test_CAC"] = coalition_test_CAC
    results["coalition_solution_majority_voting_test_F1"] = coalition_test_F1
    results["coalition_solution_majority_voting_size"] = np.sum(coalition_solution_majority_voting)

    feature_vector = np.array(coalition_solution_minority_voting).astype(bool)  # bool array from integer
    if any(feature_vector):
        coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        coalition_validation_fitness = 0
        coalition_test_CAC = 0
        coalition_test_F1 = 0

    coalition_solution_minority_voting = coalition_solution_minority_voting * 1
    results["coalition_solution_minority_voting"] = ','.join(
        [str(feature) for feature in coalition_solution_minority_voting])
    results["coalition_solution_minority_voting_validation_fitness"] = coalition_validation_fitness
    results["coalition_solution_minority_voting_test_CAC"] = coalition_test_CAC
    results["coalition_solution_minority_voting_test_F1"] = coalition_test_F1
    results["coalition_solution_minority_voting_size"] = np.sum(coalition_solution_minority_voting)

    feature_vector = np.array(coalition_solution_minority_voting).astype(bool)  # bool array from integer
    if any(feature_vector):
        coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        coalition_validation_fitness = 0
        coalition_test_CAC = 0
        coalition_test_F1 = 0

    coalition_solution_large_majority_voting = coalition_solution_large_majority_voting * 1
    results["coalition_solution_large_majority_voting"] = ','.join(
        [str(feature) for feature in coalition_solution_large_majority_voting])
    results["coalition_solution_large_majority_voting_validation_fitness"] = coalition_validation_fitness
    results["coalition_solution_large_majority_voting_test_CAC"] = coalition_test_CAC
    results["coalition_solution_large_majority_voting_test_F1"] = coalition_test_F1
    results["coalition_solution_large_majority_voting_size"] = np.sum(coalition_solution_large_majority_voting)

    feature_vector = np.array(coalition_fitness_weighted_majority_voting).astype(bool)  # bool array from integer
    if any(feature_vector):
        coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        coalition_validation_fitness = 0
        coalition_test_CAC = 0
        coalition_test_F1 = 0

    coalition_fitness_weighted_majority_voting = coalition_fitness_weighted_majority_voting * 1
    results["coalition_fitness_weighted_majority_voting"] = ','.join(
        [str(feature) for feature in coalition_fitness_weighted_majority_voting])
    results["coalition_fitness_weighted_majority_voting_validation_fitness"] = coalition_validation_fitness
    results["coalition_fitness_weighted_majority_voting_test_CAC"] = coalition_test_CAC
    results["coalition_fitness_weighted_majority_voting_test_F1"] = coalition_test_F1
    results["coalition_fitness_weighted_majority_voting_size"] = np.sum(coalition_fitness_weighted_majority_voting)

    feature_vector = np.array(coalition_fitness_rank_weighted_majority_voting).astype(bool)  # bool array from integer
    if any(feature_vector):
        coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        coalition_validation_fitness = 0
        coalition_test_CAC = 0
        coalition_test_F1 = 0

    coalition_fitness_rank_weighted_majority_voting = coalition_fitness_rank_weighted_majority_voting * 1
    results["coalition_fitness_rank_weighted_majority_voting"] = ','.join(
        [str(feature) for feature in coalition_fitness_rank_weighted_majority_voting])
    results["coalition_fitness_rank_weighted_majority_voting_validation_fitness"] = coalition_validation_fitness
    results["coalition_fitness_rank_weighted_majority_voting_test_CAC"] = coalition_test_CAC
    results["coalition_fitness_rank_weighted_majority_voting_test_F1"] = coalition_test_F1
    results["coalition_fitness_rank_weighted_majority_voting_size"] = np.sum(
        coalition_fitness_rank_weighted_majority_voting)

    feature_vector = np.array(coalition_size_weighted_majority_voting).astype(bool)  # bool array from integer
    if any(feature_vector):
        coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        coalition_validation_fitness = 0
        coalition_test_CAC = 0
        coalition_test_F1 = 0

    coalition_size_weighted_majority_voting = coalition_size_weighted_majority_voting * 1
    results["coalition_size_weighted_majority_voting"] = ','.join(
        [str(feature) for feature in coalition_size_weighted_majority_voting])
    results["coalition_size_weighted_majority_voting_validation_fitness"] = coalition_validation_fitness
    results["coalition_size_weighted_majority_voting_test_CAC"] = coalition_test_CAC
    results["coalition_size_weighted_majority_voting_test_F1"] = coalition_test_F1
    results["coalition_size_weighted_majority_voting_size"] = np.sum(coalition_size_weighted_majority_voting)

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

    coalition_fitness_size_rank_weighted_majority_voting = coalition_fitness_size_rank_weighted_majority_voting * 1
    results["coalition_fitness_size_rank_weighted_majority_voting"] = ','.join(
        [str(feature) for feature in coalition_fitness_size_rank_weighted_majority_voting])
    results["coalition_fitness_size_rank_weighted_majority_voting_validation_fitness"] = coalition_validation_fitness
    results["coalition_fitness_size_rank_weighted_majority_voting_test_CAC"] = coalition_test_CAC
    results["coalition_fitness_size_rank_weighted_majority_voting_test_F1"] = coalition_test_F1
    results["coalition_fitness_size_rank_weighted_majority_voting_size"] = np.sum(
        coalition_fitness_size_rank_weighted_majority_voting)

    coalition_solution_union = np.array([False] * len(coalition[0]), dtype=bool)
    coalition_solution_intersection = np.array([True] * len(coalition[0]), dtype=bool)
    for solution in coalition:
        coalition_solution_intersection = coalition_solution_intersection * np.array(solution).astype(bool)
        coalition_solution_union = coalition_solution_union + np.array(solution).astype(bool)

    # intersections = []
    # for candidate1 in coalition:
    #     for candidate2 in coalition:
    #         if not np.array_equal(candidate1, candidate2):
    #             intersections.append(np.array(candidate1).astype(bool) * np.array(candidate2).astype(bool))

    coalition_solution_multiintersection = np.array([False] * len(coalition[0]), dtype=bool)

    # for intersect in intersections:
    #     coalition_solution_multiintersection = coalition_solution_multiintersection + np.array(intersect).astype(bool)

    feature_vector = np.array(coalition_solution_union).astype(bool)  # bool array from integer
    coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
    coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
    coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)

    coalition_solution_union = coalition_solution_union * 1
    results["coalition_solution_union"] = ','.join([str(feature) for feature in coalition_solution_union])
    results["coalition_solution_union_validation_fitness"] = coalition_validation_fitness
    results["coalition_solution_union_test_CAC"] = coalition_test_CAC
    results["coalition_solution_union_test_F1"] = coalition_test_F1
    results["coalition_solution_union_size"] = np.sum(coalition_solution_union)

    feature_vector = np.array(coalition_solution_intersection).astype(bool)  # bool array from integer
    if any(feature_vector):
        coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        coalition_validation_fitness = 0
        coalition_test_CAC = 0
        coalition_test_F1 = 0

    coalition_solution_intersection = coalition_solution_intersection * 1
    results["coalition_solution_intersection"] = ','.join(
        [str(feature) for feature in coalition_solution_intersection])
    results["coalition_solution_intersection_validation_fitness"] = coalition_validation_fitness
    results["coalition_solution_intersection_test_CAC"] = coalition_test_CAC
    results["coalition_solution_intersection_test_F1"] = coalition_test_F1
    results["coalition_solution_intersection_size"] = np.sum(coalition_solution_intersection)

    feature_vector = np.array(coalition_solution_multiintersection).astype(bool)  # bool array from integer
    if any(feature_vector):
        coalition_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        coalition_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        coalition_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        coalition_validation_fitness = 0
        coalition_test_CAC = 0
        coalition_test_F1 = 0

    coalition_solution_multiintersection = coalition_solution_multiintersection * 1
    results["coalition_solution_multiintersection"] = ','.join(
        [str(feature) for feature in coalition_solution_multiintersection])
    results["coalition_solution_multiintersection_validation_fitness"] = coalition_validation_fitness
    results["coalition_solution_multiintersection_test_CAC"] = coalition_test_CAC
    results["coalition_solution_multiintersection_test_F1"] = coalition_test_F1
    results["coalition_solution_multiintersection_size"] = np.sum(coalition_solution_multiintersection)

    similar_counts = []
    indices_to_throw = []
    for idx1 in range(len(population) - 1):
        count = 0
        for idx2 in range(idx1 + 1, len(population)):
            if get_Hamming_distance(population[idx1], population[idx2]) == 1:
                count += 1
                if population_fitness[idx1] > population_fitness[idx2]:
                    if idx2 not in indices_to_throw:
                        indices_to_throw.append(idx2)
                elif population_fitness[idx1] == population_fitness[idx2]:
                    if sum(population[idx1]) < sum(population[idx2]):
                        if idx2 not in indices_to_throw:
                            indices_to_throw.append(idx2)
                    else:
                        if idx1 not in indices_to_throw:
                            indices_to_throw.append(idx1)
                else:
                    if idx2 not in indices_to_throw:
                        indices_to_throw.append(idx2)
        similar_counts.append(count)

    print(similar_counts)

    reduced_population = copy.deepcopy(population)
    reduced_population = np.delete(reduced_population, indices_to_throw, axis=0)
    reduced_population_fitness = np.delete(population_fitness, indices_to_throw, axis=0)

    population = copy.deepcopy(reduced_population)
    population_fitness = copy.deepcopy(reduced_population_fitness)
    results["reduced_population_size"] = len(population)
    activated_features = np.sum(population, axis=0)
    activated_features = activated_features >= (len(population) // 2)
    reduced_solution_majority_voting = activated_features * 1

    activated_features = np.sum(population, axis=0)
    activated_features = activated_features >= (len(population) // 3)
    reduced_solution_minority_voting = activated_features * 1

    activated_features = np.sum(population, axis=0)
    activated_features = activated_features >= (2 * len(population) // 3)
    reduced_solution_large_majority_voting = activated_features * 1

    activated_features = np.zeros(len(population[0]), dtype=float)
    total_fitness = 0
    for idx, candidate in enumerate(population):
        activated_features += population_fitness[idx] * np.array(candidate).astype(int)
        total_fitness += population_fitness[idx]

    reduced_fitness_weighted_majority_voting = (activated_features >= total_fitness // 2) * 1

    activated_features = np.zeros(len(population[0]), dtype=int)
    total_votes = 0
    for idx, candidate in enumerate(population):
        rank = len(population_fitness[population_fitness <= population_fitness[idx]])
        activated_features += rank * np.array(candidate).astype(int)
        total_votes += rank

    reduced_fitness_rank_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

    activated_features = np.zeros(len(population[0]), dtype=int)
    total_votes = 0
    for idx, candidate in enumerate(population):
        rank = len(population[np.sum(population, axis=1) >= np.sum(population[idx])])
        activated_features += rank * np.array(candidate).astype(int)
        total_votes += rank

    reduced_size_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

    activated_features = np.zeros(len(population[0]), dtype=int)
    total_votes = 0
    for idx, candidate in enumerate(population):
        rank = len(population_fitness[population_fitness <= population_fitness[idx]])
        rank += len(population[np.sum(population, axis=1) >= np.sum(population[idx])])
        activated_features += rank * np.array(candidate).astype(int)
        total_votes += rank

    reduced_fitness_size_rank_weighted_majority_voting = (activated_features >= total_votes // 2) * 1

    feature_vector = np.array(reduced_solution_majority_voting).astype(bool)  # bool array from integer
    if any(feature_vector):
        reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        reduced_validation_fitness = 0
        reduced_test_CAC = 0
        reduced_test_F1 = 0

    reduced_solution_majority_voting = reduced_solution_majority_voting * 1
    results["reduced_solution_majority_voting"] = ','.join(
        [str(feature) for feature in reduced_solution_majority_voting])
    results["reduced_solution_majority_voting_validation_fitness"] = reduced_validation_fitness
    results["reduced_solution_majority_voting_test_CAC"] = reduced_test_CAC
    results["reduced_solution_majority_voting_test_F1"] = reduced_test_F1
    results["reduced_solution_majority_voting_size"] = np.sum(reduced_solution_majority_voting)

    feature_vector = np.array(reduced_solution_minority_voting).astype(bool)  # bool array from integer
    if any(feature_vector):
        reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        reduced_validation_fitness = 0
        reduced_test_CAC = 0
        reduced_test_F1 = 0

    reduced_solution_minority_voting = reduced_solution_minority_voting * 1
    results["reduced_solution_minority_voting"] = ','.join(
        [str(feature) for feature in reduced_solution_minority_voting])
    results["reduced_solution_minority_voting_validation_fitness"] = reduced_validation_fitness
    results["reduced_solution_minority_voting_test_CAC"] = reduced_test_CAC
    results["reduced_solution_minority_voting_test_F1"] = reduced_test_F1
    results["reduced_solution_minority_voting_size"] = np.sum(reduced_solution_minority_voting)

    feature_vector = np.array(reduced_solution_minority_voting).astype(bool)  # bool array from integer
    if any(feature_vector):
        reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        reduced_validation_fitness = 0
        reduced_test_CAC = 0
        reduced_test_F1 = 0

    reduced_solution_large_majority_voting = reduced_solution_large_majority_voting * 1
    results["reduced_solution_large_majority_voting"] = ','.join(
        [str(feature) for feature in reduced_solution_large_majority_voting])
    results["reduced_solution_large_majority_voting_validation_fitness"] = reduced_validation_fitness
    results["reduced_solution_large_majority_voting_test_CAC"] = reduced_test_CAC
    results["reduced_solution_large_majority_voting_test_F1"] = reduced_test_F1
    results["reduced_solution_large_majority_voting_size"] = np.sum(reduced_solution_large_majority_voting)

    feature_vector = np.array(reduced_fitness_weighted_majority_voting).astype(bool)  # bool array from integer
    if any(feature_vector):
        reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        reduced_validation_fitness = 0
        reduced_test_CAC = 0
        reduced_test_F1 = 0

    reduced_fitness_weighted_majority_voting = reduced_fitness_weighted_majority_voting * 1
    results["reduced_fitness_weighted_majority_voting"] = ','.join(
        [str(feature) for feature in reduced_fitness_weighted_majority_voting])
    results["reduced_fitness_weighted_majority_voting_validation_fitness"] = reduced_validation_fitness
    results["reduced_fitness_weighted_majority_voting_test_CAC"] = reduced_test_CAC
    results["reduced_fitness_weighted_majority_voting_test_F1"] = reduced_test_F1
    results["reduced_fitness_weighted_majority_voting_size"] = np.sum(reduced_fitness_weighted_majority_voting)

    feature_vector = np.array(reduced_fitness_rank_weighted_majority_voting).astype(bool)  # bool array from integer
    if any(feature_vector):
        reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        reduced_validation_fitness = 0
        reduced_test_CAC = 0
        reduced_test_F1 = 0

    reduced_fitness_rank_weighted_majority_voting = reduced_fitness_rank_weighted_majority_voting * 1
    results["reduced_fitness_rank_weighted_majority_voting"] = ','.join(
        [str(feature) for feature in reduced_fitness_rank_weighted_majority_voting])
    results["reduced_fitness_rank_weighted_majority_voting_validation_fitness"] = reduced_validation_fitness
    results["reduced_fitness_rank_weighted_majority_voting_test_CAC"] = reduced_test_CAC
    results["reduced_fitness_rank_weighted_majority_voting_test_F1"] = reduced_test_F1
    results["reduced_fitness_rank_weighted_majority_voting_size"] = np.sum(
        reduced_fitness_rank_weighted_majority_voting)

    feature_vector = np.array(reduced_size_weighted_majority_voting).astype(bool)  # bool array from integer
    if any(feature_vector):
        reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        reduced_validation_fitness = 0
        reduced_test_CAC = 0
        reduced_test_F1 = 0

    reduced_size_weighted_majority_voting = reduced_size_weighted_majority_voting * 1
    results["reduced_size_weighted_majority_voting"] = ','.join(
        [str(feature) for feature in reduced_size_weighted_majority_voting])
    results["reduced_size_weighted_majority_voting_validation_fitness"] = reduced_validation_fitness
    results["reduced_size_weighted_majority_voting_test_CAC"] = reduced_test_CAC
    results["reduced_size_weighted_majority_voting_test_F1"] = reduced_test_F1
    results["reduced_size_weighted_majority_voting_size"] = np.sum(reduced_size_weighted_majority_voting)

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

    reduced_fitness_size_rank_weighted_majority_voting = reduced_fitness_size_rank_weighted_majority_voting * 1
    results["reduced_fitness_size_rank_weighted_majority_voting"] = ','.join(
        [str(feature) for feature in reduced_fitness_size_rank_weighted_majority_voting])
    results["reduced_fitness_size_rank_weighted_majority_voting_validation_fitness"] = reduced_validation_fitness
    results["reduced_fitness_size_rank_weighted_majority_voting_test_CAC"] = reduced_test_CAC
    results["reduced_fitness_size_rank_weighted_majority_voting_test_F1"] = reduced_test_F1
    results["reduced_fitness_size_rank_weighted_majority_voting_size"] = np.sum(
        reduced_fitness_size_rank_weighted_majority_voting)

    reduced_solution_union = np.array([False] * len(population[0]), dtype=bool)
    reduced_solution_intersection = np.array([True] * len(population[0]), dtype=bool)
    for solution in population:
        reduced_solution_intersection = reduced_solution_intersection * np.array(solution).astype(bool)
        reduced_solution_union = reduced_solution_union + np.array(solution).astype(bool)

    # intersections = []
    # for candidate1 in population:
    #     for candidate2 in population:
    #         if not np.array_equal(candidate1, candidate2):
    #             intersections.append(np.array(candidate1).astype(bool) * np.array(candidate2).astype(bool))

    reduced_solution_multiintersection = np.array([False] * len(population[0]), dtype=bool)

    # for intersect in intersections:
    #     reduced_solution_multiintersection = reduced_solution_multiintersection + np.array(intersect).astype(bool)

    feature_vector = np.array(reduced_solution_union).astype(bool)  # bool array from integer
    reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
    reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
    reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)

    reduced_solution_union = reduced_solution_union * 1
    results["reduced_solution_union"] = ','.join([str(feature) for feature in reduced_solution_union])
    results["reduced_solution_union_validation_fitness"] = reduced_validation_fitness
    results["reduced_solution_union_test_CAC"] = reduced_test_CAC
    results["reduced_solution_union_test_F1"] = reduced_test_F1
    results["reduced_solution_union_size"] = np.sum(reduced_solution_union)

    feature_vector = np.array(reduced_solution_intersection).astype(bool)  # bool array from integer
    if any(feature_vector):
        reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        reduced_validation_fitness = 0
        reduced_test_CAC = 0
        reduced_test_F1 = 0

    reduced_solution_intersection = reduced_solution_intersection * 1
    results["reduced_solution_intersection"] = ','.join(
        [str(feature) for feature in reduced_solution_intersection])
    results["reduced_solution_intersection_validation_fitness"] = reduced_validation_fitness
    results["reduced_solution_intersection_test_CAC"] = reduced_test_CAC
    results["reduced_solution_intersection_test_F1"] = reduced_test_F1
    results["reduced_solution_intersection_size"] = np.sum(reduced_solution_intersection)

    feature_vector = np.array(reduced_solution_multiintersection).astype(bool)  # bool array from integer
    if any(feature_vector):
        reduced_validation_fitness = fitness_function.evaluate_on_validation(feature_vector)
        reduced_test_CAC = fitness_function_CAC.evaluate_on_test(feature_vector)
        reduced_test_F1 = fitness_function_F1.evaluate_on_test(feature_vector)
    else:
        reduced_validation_fitness = 0
        reduced_test_CAC = 0
        reduced_test_F1 = 0

    reduced_solution_multiintersection = reduced_solution_multiintersection * 1
    results["reduced_solution_multiintersection"] = ','.join(
        [str(feature) for feature in reduced_solution_multiintersection])
    results["reduced_solution_multiintersection_validation_fitness"] = reduced_validation_fitness
    results["reduced_solution_multiintersection_test_CAC"] = reduced_test_CAC
    results["reduced_solution_multiintersection_test_F1"] = reduced_test_F1
    results["reduced_solution_multiintersection_size"] = np.sum(reduced_solution_multiintersection)

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
    reduced_union_solutions = []
    reduced_intersection_solutions = []
    reduced_multiintersection_solutions = []

    for run_summary in summary_results:
        if run_summary["dataset"] == dataset:
            best_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in run_summary["best_solution"].split(',')]).astype(
                    bool))
            union_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in run_summary["union_solution"].split(',')]).astype(
                    bool))
            intersection_solutions.append(
                np.array([True if feature == '1' else False for feature in
                          run_summary["intersection_solution"].split(',')]).astype(bool))
            multiintersection_solutions.append(
                np.array([True if feature == '1' else False for feature in
                          run_summary["multiintersection_solution"].split(',')]).astype(bool))
            majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["majority_voting"].split(',')]).astype(
                    bool))
            minority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["minority_voting"].split(',')]).astype(
                    bool))
            large_majority_voting_solutions.append(
                np.array([True if feature == '1' else False for feature in
                          run_summary["large_majority_voting"].split(',')]).astype(bool))
            fitness_weighted_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["fitness_weighted_majority_voting"].split(',')]).astype(
                    bool))
            fitness_rank_weighted_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["fitness_rank_weighted_majority_voting"].split(',')]).astype(
                    bool))
            size_weighted_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["size_weighted_majority_voting"].split(',')]).astype(
                    bool))
            fitness_size_rank_weighted_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["fitness_size_rank_weighted_majority_voting"].split(',')]).astype(
                    bool))
            coalition_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["coalition_solution_majority_voting"].split(',')]).astype(
                    bool))
            coalition_minority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["coalition_solution_minority_voting"].split(',')]).astype(
                    bool))
            coalition_large_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["coalition_solution_large_majority_voting"].split(',')]).astype(
                    bool))
            coalition_fitness_weighted_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["coalition_fitness_weighted_majority_voting"].split(',')]).astype(
                    bool))
            coalition_fitness_rank_weighted_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["coalition_fitness_rank_weighted_majority_voting"].split(',')]).astype(
                    bool))
            coalition_size_weighted_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["coalition_size_weighted_majority_voting"].split(',')]).astype(
                    bool))
            coalition_fitness_size_rank_weighted_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["coalition_fitness_size_rank_weighted_majority_voting"].split(',')]).astype(
                    bool))
            coalition_union_solutions.append(
                np.array([True if feature == '1' else False for feature in
                          run_summary["coalition_solution_union"].split(',')]).astype(bool))
            coalition_intersection_solutions.append(
                np.array([True if feature == '1' else False for feature in
                          run_summary["coalition_solution_intersection"].split(',')]).astype(
                    bool))
            coalition_multiintersection_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["coalition_solution_multiintersection"].split(',')]).astype(
                    bool))
            reduced_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["reduced_solution_majority_voting"].split(',')]).astype(
                    bool))
            reduced_minority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["reduced_solution_minority_voting"].split(',')]).astype(
                    bool))
            reduced_large_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["reduced_solution_large_majority_voting"].split(',')]).astype(
                    bool))
            reduced_fitness_weighted_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["reduced_fitness_weighted_majority_voting"].split(',')]).astype(
                    bool))
            reduced_fitness_rank_weighted_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["reduced_fitness_rank_weighted_majority_voting"].split(',')]).astype(
                    bool))
            reduced_size_weighted_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["reduced_size_weighted_majority_voting"].split(',')]).astype(
                    bool))
            reduced_fitness_size_rank_weighted_majority_voting_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["reduced_fitness_size_rank_weighted_majority_voting"].split(',')]).astype(
                    bool))
            reduced_union_solutions.append(
                np.array([True if feature == '1' else False for feature in
                          run_summary["reduced_solution_union"].split(',')]).astype(bool))
            reduced_intersection_solutions.append(
                np.array([True if feature == '1' else False for feature in
                          run_summary["reduced_solution_intersection"].split(',')]).astype(
                    bool))
            reduced_multiintersection_solutions.append(
                np.array(
                    [True if feature == '1' else False for feature in
                     run_summary["reduced_solution_multiintersection"].split(',')]).astype(
                    bool))

    stability_report["best_solution"] = stability_function(best_solutions)
    stability_report["union_solution"] = stability_function(union_solutions)
    stability_report["intersection_solution"] = stability_function(intersection_solutions)
    stability_report["multiintersection_solution"] = stability_function(multiintersection_solutions)
    stability_report["majority_voting"] = stability_function(majority_voting_solutions)
    stability_report["minority_voting"] = stability_function(minority_voting_solutions)
    stability_report["large_majority_voting"] = stability_function(large_majority_voting_solutions)
    stability_report["fitness_weighted_majority_voting"] = stability_function(
        fitness_weighted_majority_voting_solutions)
    stability_report["fitness_rank_weighted_majority_voting"] = stability_function(
        fitness_rank_weighted_majority_voting_solutions)
    stability_report["size_weighted_majority_voting"] = stability_function(size_weighted_majority_voting_solutions)
    stability_report["fitness_size_rank_weighted_majority_voting"] = stability_function(fitness_size_rank_weighted_majority_voting_solutions)


    stability_report["coalition_union_solution"] = stability_function(coalition_union_solutions)
    stability_report["coalition_intersection_solution"] = stability_function(coalition_intersection_solutions)
    stability_report["coalition_multiintersection_solution"] = stability_function(coalition_multiintersection_solutions)
    stability_report["coalition_majority_voting"] = stability_function(coalition_majority_voting_solutions)
    stability_report["coalition_minority_voting"] = stability_function(coalition_minority_voting_solutions)
    stability_report["coalition_large_majority_voting"] = stability_function(coalition_large_majority_voting_solutions)
    stability_report["coalition_fitness_weighted_majority_voting"] = stability_function(
        coalition_fitness_weighted_majority_voting_solutions)
    stability_report["coalition_fitness_rank_weighted_majority_voting"] = stability_function(
        coalition_fitness_rank_weighted_majority_voting_solutions)
    stability_report["coalition_size_weighted_majority_voting"] = stability_function(
        coalition_size_weighted_majority_voting_solutions)
    stability_report["coalition_fitness_size_rank_weighted_majority_voting"] = stability_function(coalition_fitness_size_rank_weighted_majority_voting_solutions)


    stability_report["reduced_union_solution"] = stability_function(reduced_union_solutions)
    stability_report["reduced_intersection_solution"] = stability_function(reduced_intersection_solutions)
    stability_report["reduced_multiintersection_solution"] = stability_function(reduced_multiintersection_solutions)
    stability_report["reduced_majority_voting"] = stability_function(reduced_majority_voting_solutions)
    stability_report["reduced_minority_voting"] = stability_function(reduced_minority_voting_solutions)
    stability_report["reduced_large_majority_voting"] = stability_function(reduced_large_majority_voting_solutions)
    stability_report["reduced_fitness_weighted_majority_voting"] = stability_function(
        reduced_fitness_weighted_majority_voting_solutions)
    stability_report["reduced_fitness_rank_weighted_majority_voting"] = stability_function(
        reduced_fitness_rank_weighted_majority_voting_solutions)
    stability_report["reduced_size_weighted_majority_voting"] = stability_function(
        reduced_size_weighted_majority_voting_solutions)
    stability_report["reduced_fitness_size_rank_weighted_majority_voting"] = stability_function(reduced_fitness_size_rank_weighted_majority_voting_solutions)


    print(stability_report)
    return stability_report


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
fitness = 'f1_macro'
summary_results = {}
asm_stability_results = {}
anhd_stability_results = {}

for optimiser in fs_wrappers:
    for procedure in evaluation_procedures:
        for classifier in classifiers:
            summary_results[optimiser.name + procedure + classifier.name] = []
            asm_stability_results[optimiser.name + procedure + classifier.name] = []
            anhd_stability_results[optimiser.name + procedure + classifier.name] = []
            for file in datasets:

                dataset = pd.read_csv(file.path, header=0, index_col=0)
                data = dataset.iloc[:, :-1].values
                dimensionality = data.shape[1]

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
                                [float(data[0]), float(test_data[idx // 14][0]), float(test_data[idx // 14][1]), test_data[idx // 14][-1]])  # 0 je f1:macro, 1 CAC

                filename = '_'.join(
                    [procedure, optimiser.name + "_", classifier.name, file.name,
                     'archive'])
                print("Processing file {0}".format(filename))
                test_header, test_data = CsvProcessor().read_file(filename='logs/archive/' + filename + "_full_test")

                if test_header is not None and test_data is not None:
                    test_data = [row for row in test_data if row]
                    test_data = np.array(test_data)

                if test_header is not None and test_data is not None:
                    run = 0
                    index = 0
                    population = []
                    population_fitness = []
                    population_test_fitness = []
                    best_test_fitness = 0
                    fitness_function = ClassificationProblem(file, classifier,
                                                             random_state=42,  # dodati run + za variable
                                                             test_size=experiment_setup["test_size"],
                                                             validation_size=experiment_setup[
                                                                 "validation_size"],
                                                             wrapper_fitness_metric=fitness,
                                                             metrics=classification_metrics)
                    for idx, candidate in enumerate(test_data):
                        if idx > 0 and candidate[0] == 'stop':
                            summary_results[optimiser.name + procedure + classifier.name].append(
                                feature_subset_aggregation(file.name, run, copy.deepcopy(population),
                                                           copy.deepcopy(population_fitness), search_results, best_test_fitness))
                            run += 1
                            index = 0
                            population = []
                            population_fitness = []
                            population_test_fitness = []
                            fitness_function = ClassificationProblem(file, classifier,
                                                                     random_state=42,  # dodati run + za variable
                                                                     test_size=experiment_setup["test_size"],
                                                                     validation_size=experiment_setup[
                                                                         "validation_size"],
                                                                     wrapper_fitness_metric=fitness,
                                                                     metrics=classification_metrics)
                            best_test_fitness = 0

                        population.append([True if bit == '1' else False for bit in
                                             candidate[header_idx['vector_val']]])

                        # population_fitness.append(fitness_function.evaluate_on_validation(
                        #     np.array(population[index]).astype(bool)))
                        # population_test_fitness.append(fitness_function.evaluate_on_test(
                        #     np.array(population[index]).astype(bool)))

                        population_fitness.append(float(candidate[header_idx[fitness + "_val"]]))
                        population_test_fitness.append(float(candidate[header_idx[fitness + "_test"] + 1]))

                        if idx == 0 or population_test_fitness[index] > best_test_fitness:
                            best_test_fitness = population_test_fitness[index]
                        index += 1

                        if idx == len(test_data) - 1:
                            summary_results[optimiser.name + procedure + classifier.name].append(
                                feature_subset_aggregation(file.name, run, copy.deepcopy(population),
                                                           copy.deepcopy(population_fitness), search_results, best_test_fitness))

                    asm_stability_results[optimiser.name + procedure + classifier.name].append(
                        get_stability_report(
                            summary_results[optimiser.name + procedure + classifier.name], file.name, calculate_ASM))

                    anhd_stability_results[optimiser.name + procedure + classifier.name].append(
                        get_stability_report(
                            summary_results[optimiser.name + procedure + classifier.name], file.name,
                            calculate_ANHD))

            for row in summary_results[optimiser.name + procedure + classifier.name]:
                CsvProcessor().save_summary_results(filename='_'.join(
                    [procedure, optimiser.name, classifier.name, "_full_archive_combinations"]),
                    header=list(row.keys()),
                    data=list(row.values()))

            for row in asm_stability_results[optimiser.name + procedure + classifier.name]:
                CsvProcessor().save_summary_results(filename='_'.join(
                    [procedure, optimiser.name, classifier.name, "_full_archive_asm_stabilities"]),
                    header=list(row.keys()),
                    data=list(row.values()))

            for row in anhd_stability_results[optimiser.name + procedure + classifier.name]:
                CsvProcessor().save_summary_results(filename='_'.join(
                    [procedure, optimiser.name, classifier.name, "_full_archive_anhd_stabilities"]),
                    header=list(row.keys()),
                    data=list(row.values()))
