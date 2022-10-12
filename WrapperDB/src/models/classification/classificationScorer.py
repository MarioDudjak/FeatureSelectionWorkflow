from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, SCORERS, make_scorer, \
    precision_score, recall_score, cohen_kappa_score, davies_bouldin_score
from sklearn.model_selection import cross_validate, cross_val_score


class ClassificationScorer:
    prediction_results = {}
    scorers = {}

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ClassificationScorer, cls).__new__(cls)
            cls.prediction_results = {
                'f1_macro': cls.get_f1_score,
                'accuracy': cls.get_accuracy,
                'mcr': cls.get_mcr,
                'roc_auc': cls.get_roc_auc,
                'confusion_matrix': cls.get_confusion_matrix,
                'tn': cls.get_tn,
                'fp': cls.get_fp,
                'fn': cls.get_fn,
                'tp': cls.get_tp,
                'sensitivity': cls.get_sensitivity,
                'db_index': cls.get_db_index
            }

            cls.scorers = {
                'f1_macro': make_scorer(f1_score, average='macro'),
                'accuracy': make_scorer(accuracy_score),
                'roc_auc_ovo': make_scorer(roc_auc_score, average='macro'),
            }

        return cls.instance

    @classmethod
    def get_f1_score(cls, y_test, predictions):
        return f1_score(y_test, predictions, average='macro')

    @classmethod
    def get_accuracy(cls, y_test, predictions):
        return accuracy_score(y_test, predictions)

    @classmethod
    def get_db_index(cls, y_test, predictions):
        return davies_bouldin_score(y_test, predictions)

    @classmethod
    def get_mcr(cls, y_test, predictions):
        return 1 - cls.get_accuracy(y_test, predictions)

    @classmethod
    def get_roc_auc(cls, y_test, model):
        predictions = model.predict_proba(y_test)
        if len(set(y_test)) > 2 or len(set(predictions)) > 2:
            auc = 0
            for label in set(y_test):
                binary_y_test = [1 if test_label == label else 0 for test_label in y_test]
                binary_predictions = [1 if pred_label == label else 0 for pred_label in predictions]
                auc += roc_auc_score(binary_y_test, binary_predictions, average='macro')
            return auc / (len(set(y_test)))
        return roc_auc_score(y_test, predictions, average='macro')

    @classmethod
    def get_confusion_matrix(cls, y_test, predictions):
        return confusion_matrix(y_test, predictions)

    @classmethod
    def get_tn(cls, y_test, predictions):
        return confusion_matrix(y_test, predictions).ravel()[0]

    @classmethod
    def get_fp(cls, y_test, predictions):
        return confusion_matrix(y_test, predictions).ravel()[1]

    @classmethod
    def get_fn(cls, y_test, predictions):
        return confusion_matrix(y_test, predictions).ravel()[2]

    @classmethod
    def get_tp(cls, y_test, predictions):
        return confusion_matrix(y_test, predictions).ravel()[3]

    @classmethod
    def get_sensitivity(cls, y_test, predictions):
        tp = cls.get_tp(y_test, predictions)
        fn = cls.get_fn(y_test, predictions)
        return tp / (tp + fn)

    @classmethod
    def get_scores(cls, y_test, predictions, metrics):
        return {metric: score_function(y_test, predictions) for (metric, score_function) in
                cls.prediction_results.items() if metric in metrics}

    @classmethod
    def evaluate_feature_subset(cls, y_test, predictions, metric):
        return cls.prediction_results[metric](y_test, predictions)

    @classmethod
    def get_scores_cross_val(cls, model, X_train, y_train, metrics, cv):
        metrics = {metric: metric for metric in metrics}

        for metric in [metric for metric in metrics.keys() if metric not in SCORERS.keys()]:
            if metric not in cls.scorers.keys():
                metrics.pop(metric, None)
            else:
                metrics[metric] = cls.scorers[metric]

        scores = cross_validate(model, X_train, y_train, scoring=metrics, cv=cv)
        for score_key, score_values in scores.items():
            scores[score_key] = sum(score_values) / len(score_values)

        if 'test_accuracy' in scores.keys():
            scores['test_mcr'] = 1 - scores['test_accuracy']
        return scores

    @classmethod
    def evaluate_feature_subset_cross_val(cls, model, X_train, y_train, fitness, cv):
        if fitness not in SCORERS.keys():
            scores = cross_val_score(model, X_train, y_train, scoring=cls.scorers[fitness], cv=cv)
        else:
            scores = cross_val_score(model, X_train, y_train, scoring=fitness, cv=cv)

        return sum(scores) / len(scores)


