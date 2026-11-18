"""Documented model API only. Full QP training/prediction is not included."""
class LDGBNuTSVM:
    def __init__(self, *, nu1=0.1, nu2=0.1, delta=0.05, gamma="scale"):
        self.nu1, self.nu2 = nu1, nu2
        self.delta, self.gamma = delta, gamma

    def fit(self, X_train, y_train):
        """Expected to preprocess training data, construct balls, and solve QPs."""
        raise NotImplementedError("Core ν-TSVM QP solver is not part of this public release")

    def predict(self, X_test):
        """Expected to use twin-hyperplane decision functions."""
        raise NotImplementedError("Core predictor is not part of this public release")
