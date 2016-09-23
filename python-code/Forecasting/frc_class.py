import numpy as np
from sklearn.base import  BaseEstimator



class IdenitityFrc(BaseEstimator):

    def __init__(self, name=None):
        self.name = name

    def fit(self, X, Y):
        if not np.all(X == Y):
            print "Warning! X and Y are not identical in identity model"


    def predict(self, X):
        return X


class MartingalFrc(BaseEstimator):
    """Forecasts n_out Y values by last n_out values of X"""

    def __init__(self, name=None):
        self.name = name

    def fit(self, X, Y):
        self.n_out = Y.shape[1]

    def predict(self, X):
        return X[:, -self.n_out:]

