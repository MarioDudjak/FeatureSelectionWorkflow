import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, davies_bouldin_score, silhouette_score
from src.models.classification.classificationScorer import ClassificationScorer


class ClassificationProblem:

    def __init__(self, file, classifier, random_state, test_size, validation_size, wrapper_fitness_metric, metrics):
        self.dataset = pd.read_csv(file.path, header=0, index_col=0)
        self.data = self.dataset.iloc[:, :-1].values
        self.target = self.dataset.iloc[:, -1]
        self.name = '_'.join([classifier.name, file.name])
        self.dimensionality = self.data.shape[1]
        self.model = classifier.alg
        self.random_state = random_state

        self.X_rest, self.X_test, self.y_rest, self.y_test = train_test_split(self.data, self.target, test_size=test_size,
                                                                    random_state=random_state,
                                                                    shuffle=True,
                                                                    stratify=self.target)

        if validation_size != 0:
            self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(self.X_rest, self.y_rest,
                                                                                                test_size=validation_size / (
                                                                                                        1 - test_size),
                                                                                                random_state=random_state,
                                                                                                shuffle=True,
                                                                                                stratify=self.y_rest)
        else:
            self.X_train = self.X_rest
            self.y_train = self.y_rest

        self.fitness = wrapper_fitness_metric
        self.metrics = metrics
        self.scorer = ClassificationScorer()

    def evaluate_on_validation(self, binary_mask=None, comprehensive=False):
        if binary_mask is None:
            X_train = self.X_train[:, :]
            X_validation = self.X_validation[:, :]
        else:
            binary_mask = binary_mask.astype(bool)
            if np.any(binary_mask):
                X_train = self.X_train[:, binary_mask]
                X_validation = self.X_validation[:, binary_mask]
            else:
                return 0

        self.model.fit(X_train, self.y_train)
        predictions = self.model.predict(X_validation)
        if comprehensive:
            classification_scores = self.scorer.get_scores(self.y_validation, predictions,
                                                           [metric for metric in self.metrics if
                                                            metric != self.fitness and metric != 'roc_auc'])

            if 'roc_auc' in self.metrics:
                classification_scores['roc_auc'] = self._get_roc_auc_score(X_validation, self.y_validation)

            if 'db_index' in self.metrics:
                classification_scores['db_index'] = davies_bouldin_score(self.X_rest, self.y_rest)
            if 'silhouette' in self.metrics:
                classification_scores['silhouette'] = silhouette_score(self.X_rest, self.y_rest)
            return classification_scores
        else:
            if self.fitness == 'roc_auc':
                return self._get_roc_auc_score(X_validation, self.y_validation)
            elif self.fitness == 'db_index':
                return 1 / davies_bouldin_score(self.X_rest, self.y_rest)
            elif self.fitness == 'silhouette':
                return silhouette_score(self.X_rest, self.y_rest)
            else:
                return self.scorer.evaluate_feature_subset(self.y_validation, predictions, self.fitness)

    def evaluate_on_test(self, binary_mask=None, comprehensive=False):
        if binary_mask is None:
            X_train = self.X_train[:, :]
            X_test = self.X_test[:, :]
        else:
            binary_mask = binary_mask.astype(bool)
            if np.any(binary_mask):
                X_train = self.X_train[:, binary_mask]
                X_test = self.X_test[:, binary_mask]
            else:
                return 0

        self.model.fit(X_train, self.y_train)
        predictions = self.model.predict(X_test)
        if comprehensive:
            classification_scores = self.scorer.get_scores(self.y_test, predictions,
                                                           [metric for metric in self.metrics if
                                                            metric != self.fitness and metric != 'roc_auc'])

            if 'roc_auc' in self.metrics:
                classification_scores['roc_auc'] = self._get_roc_auc_score(X_test, self.y_test)
            if 'db_index' in self.metrics:
                classification_scores['db_index'] = davies_bouldin_score(X_test, self.y_test)
            if 'silhouette' in self.metrics:
                classification_scores['silhouette'] = silhouette_score(X_test, self.y_test)
            return classification_scores
        else:
            if self.fitness == 'roc_auc':
                return self._get_roc_auc_score(X_test, self.y_test)
            elif self.fitness == 'db_index':
                return 1 / davies_bouldin_score(X_test, self.y_test)
            elif self.fitness == 'silhouette':
                return silhouette_score(X_test, self.y_test)
            else:
                return self.scorer.evaluate_feature_subset(self.y_test, predictions, self.fitness)

    def evaluate_final_solution(self, binary_mask=None):
        if binary_mask is None:
            X_train = self.X_train
            X_test = self.X_test
        else:
            binary_mask = binary_mask.astype(bool)
            X_train = self.X_train[:, binary_mask]
            X_test = self.X_test[:, binary_mask]

        self.model.fit(X_train, self.y_train)
        predictions = self.model.predict(X_test)

        classification_scores = self.scorer.get_scores(self.y_test, predictions,
                                                       metrics=[metric for metric in self.metrics if
                                                                metric != 'roc_auc'])
        if 'roc_auc' in self.metrics:
            classification_scores['roc_auc'] = self._get_roc_auc_score(X_test, self.y_test)
        classification_scores['size'] = sum(binary_mask)
        classification_scores['vector'] = "".join([str(int(element)) for element in binary_mask])
        # confusion_matrix = self.scorer.get_scores(self.y_test, predictions, metrics=['confusion_matrix'])
        return classification_scores

    def evaluate_final_solution_on_validation(self, binary_mask=None):
        if binary_mask is None:
            X_train = self.X_train
            X_validation = self.X_validation
        else:
            X_train = self.X_train[:, binary_mask]
            X_validation = self.X_validation[:, binary_mask]

        self.model.fit(X_train, self.y_train)
        predictions = self.model.predict(X_validation)

        classification_scores = self.scorer.get_scores(self.y_validation, predictions,
                                                       metrics=[metric for metric in self.metrics if
                                                                metric != 'roc_auc'])

        if 'roc_auc' in self.metrics:
            classification_scores['roc_auc'] = self._get_roc_auc_score(X_validation, self.y_validation)
        classification_scores['size'] = sum(binary_mask)
        classification_scores['vector'] = "".join([str(int(element)) for element in binary_mask])
        # confusion_matrix = self.scorer.get_scores(self.y_validation, predictions, metrics=['confusion_matrix'])
        return classification_scores

    def _get_roc_auc_score(self, X, y):
        if len(set(y)) == 2:
            auc = roc_auc_score(y, self.model.predict_proba(X)[:, 1])
        else:
            auc = roc_auc_score(y, self.model.predict_proba(X), multi_class='ovr')

        return auc


class ClassificationProblemCV:

    def __init__(self, file, classifier, random_state, test_size, k_folds, wrapper_fitness_metric, metrics):
        self.dataset = pd.read_csv(file.path, header=0, index_col=0)
        self.data = self.dataset.iloc[:, :-1].values
        self.target = self.dataset.iloc[:, -1]
        self.name = '_'.join([classifier.name, file.name])
        self.dimensionality = self.data.shape[1]
        self.model = classifier.alg
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.target,
                                                                                test_size=test_size,
                                                                                random_state=random_state,
                                                                                shuffle=True,
                                                                                stratify=self.target)

        self.cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

        self.fitness = wrapper_fitness_metric
        self.metrics = metrics
        self.scorer = ClassificationScorer()

    def evaluate_on_validation(self, binary_mask=None, comprehensive=None):
        if binary_mask is None:
            X_train = self.X_train[:, :]
        else:
            binary_mask = binary_mask.astype(bool)
            if np.any(binary_mask):
                X_train = self.X_train[:, binary_mask]
            else:
                return 0

        if comprehensive:
            return self.scorer.get_scores_cross_val(self.model, X_train, self.y_train,
                                                    [metric for metric in self.metrics if metric != self.fitness],
                                                    cv=self.cv)
        else:
            return self.scorer.evaluate_feature_subset_cross_val(self.model, X_train, self.y_train, self.fitness,
                                                                 self.cv)

    def evaluate_on_test(self, binary_mask=None, comprehensive=None):
        if binary_mask is None:
            X_train = self.X_train[:, :]
            X_test = self.X_test[:, :]
        else:
            binary_mask = binary_mask.astype(bool)
            if np.any(binary_mask):
                X_train = self.X_train[:, binary_mask]
                X_test = self.X_test[:, binary_mask]
            else:
                return 0

        self.model.fit(X_train, self.y_train)
        predictions = self.model.predict(X_test)
        if comprehensive:
            return self.scorer.get_scores(self.y_test, predictions,
                                          [metric for metric in self.metrics if metric != self.fitness])
        else:
            return self.scorer.evaluate_feature_subset(self.y_test, predictions, self.fitness)

    def evaluate_final_solution(self, binary_mask=None):
        if binary_mask is None:
            X_train = self.X_train
            X_test = self.X_test
        else:
            binary_mask = binary_mask.astype(bool)
            X_train = self.X_train[:, binary_mask]
            X_test = self.X_test[:, binary_mask]

        self.model.fit(X_train, self.y_train)
        predictions = self.model.predict(X_test)
        classification_scores = self.scorer.get_scores(self.y_test, predictions, metrics=self.metrics)
        classification_scores['size'] = sum(binary_mask)
        classification_scores['vector'] = "".join([str(int(element)) for element in binary_mask])
        return classification_scores
