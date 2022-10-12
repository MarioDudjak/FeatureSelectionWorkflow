test_size = 0.2

training_size = {
    '1': int(80 / (1-test_size)),
    '2': int(400 / (1-test_size)),
    '3': int(1600 / (1-test_size))
}

imbalance_degree = {
    '1': 0.5,  # IR = 1
    '2': 0.6,  # IR = 1.5
    '3': 0.7,  # IR = 2.33
    '4': 0.75,  # IR = 3
    '5': 0.8,  # IR = 4
    '6': 0.85,  # IR = 5.67
    '7': 0.9,  # IR = 9
    '8': 0.95,  # IR = 19
    '9': 0.975,  # IR = 39
    '10': 0.99  # IR = 99
}

noise_level = {
    '1': 0.05,
    '2': 0.1,
    '3': 0.2
}

small_disjuncts_complexity = {
    '1': 1,
    '2': 2,
    '3': 3
}

class_overlapping_degree = {
    '1': 0.1,
    '2': 0.2,
    '3': 0.3
}

experiment_setup = {
    'runs': 30,
    'random_state': 42,
    'test_size': 0.2,
    'training_size': 0.8,

    'parameter_tuning':
        {
            'folds': 5,
            'repeats': 3,
            'scoring_metric': 'roc_auc_ovo',
            'n_iter': 1000
        },
    'metrics': ['acc', 'mcr', 'kappa', 'roc_auc', 'f_score', 'gmean', 'precision', 'recall', 'tn', 'fp', 'fn', 'tp',
                'duration']
}
