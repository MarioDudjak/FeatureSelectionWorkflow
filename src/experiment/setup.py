from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from src.models.classification.classifiers.baseClassifier import BaseClassifier


class ExperimentSetup:
    def __init__(self, runs, random_states, wrapper_fitness_metric, test_size, validation_size=None, k_folds=None):
        self.runs = runs
        self.random_states = random_states
        self.wrapper_fitness_metric = wrapper_fitness_metric
        self.test_size = test_size
        self.validation_size = validation_size
        self.k_folds = k_folds


classification_metrics = ['f1_macro', 'accuracy', 'recall', 'tp', 'tn', 'fp', 'fn']

experiment_setup = {
    "runs": 30,
    "validation_size": 0.25,
    "test_size": 0.25
}

fitness = 'f1_macro'

classifiers = [
    BaseClassifier(KNeighborsClassifier(n_neighbors=1), "1NN"),
    BaseClassifier(KNeighborsClassifier(n_neighbors=5), "5NN"),
    BaseClassifier(GaussianNB(), "GNB"),
    BaseClassifier(SVC(probability=True, random_state=42), "SVM")
]
