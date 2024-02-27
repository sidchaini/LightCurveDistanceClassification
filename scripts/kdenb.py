"""
Adapted from https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class KDENBClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE

    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    metric : str or callable
        the distance metric, passed to KernelDensity
    """

    def __init__(self, bandwidth="scott", kernel="gaussian", metric="euclidean"):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        if callable(self.metric):
            self.models_ = [
                KernelDensity(
                    bandwidth=self.bandwidth,
                    kernel=self.kernel,
                    metric="pyfunc",
                    metric_params={"func": self.metric},
                ).fit(Xi)
                for Xi in training_sets
            ]
        else:
            self.models_ = [
                KernelDensity(
                    bandwidth=self.bandwidth, kernel=self.kernel, metric=self.metric
                ).fit(Xi)
                for Xi in training_sets
            ]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0]) for Xi in training_sets]
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)

        logprobs = np.array([model.score_samples(X) for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)

        return self.classes_[np.argmax(self.predict_proba(X), 1)]
