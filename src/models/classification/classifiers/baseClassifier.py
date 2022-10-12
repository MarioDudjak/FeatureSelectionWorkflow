class BaseClassifier:

    def __init__(self, alg, alg_name):
        self.alg = alg
        self.name = alg_name

    def fit(self):
        return self.alg.fit()

    def predict(self):
        return self.alg.predict()

    def get_params(self):
        return self.get_params()

    def predict_proba(self):
        return self.alg.predict_proba()
