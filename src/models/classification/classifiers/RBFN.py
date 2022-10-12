from src.models.classification.classifiers.baseClassifier import BaseClassifier


class RBFN(BaseClassifier):

    def __init__(self, alg, alg_name):
        super().__init__(alg, alg_name)
        

    def fit(self):
        return self.alg.fit()

    def predict(self):
        return self.alg.predict()

    def get_params(self):
        return self.get_params()

    def predict_proba(self):
        return self.alg.predict_proba()