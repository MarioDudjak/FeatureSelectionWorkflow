from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.models.classification.classifiers.baseClassifier import BaseClassifier


class ExperimentSetup:
    def __init__(self, runs, random_states, wrapper_fitness_metric, test_size, validation_size=None, k_folds=None):
        self.runs = runs
        self.random_states = random_states
        self.wrapper_fitness_metric = wrapper_fitness_metric
        self.test_size = test_size
        self.validation_size = validation_size
        self.k_folds = k_folds


classification_metrics = ['accuracy', 'f1_macro']

experiment_setup = {
    "runs": 30,
    "validation_size": 0.3,
    "test_size": 0.3,
    "k_folds1": 5,
    "k_folds2": 3,
    "k_folds3": 10
}

fitness = 'db_index'

classifiers = [
    #BaseClassifier(SVC(probability=True, random_state=42), "SVM"),
    BaseClassifier(KNeighborsClassifier(n_neighbors=5), "KNN5")
]
