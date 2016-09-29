from __future__ import print_function

import numpy as np
from sklearn.base import  BaseEstimator


class IdentityModel(BaseEstimator):
    # Base class for feature selection

    def __init__(self, name=None):
        self.name = name

    def fit(self, X, Y):
        #override this module
        return self

    def predict(self, X):
        # override this module
        return X

    def transform(self, X):
        return self.predict(X)

    def print_pars(self):
        params = self.get_params()
        print(self.name)
        for k, v in params.items():
            print(k, ":", v)

class IdentityFrc(IdentityModel):
    # Base class for feature selection

    def fit(self, X, Y):
        # check that X and Y are the same:
        if not np.all(X == Y):
            print("X and Y should be the same for identity frc")
        return self

    def predict(self, X):
        # override this module
        # feature selection does not depemd on Y
        return X





class IdentityGenerator(IdentityModel):
    # Base class for feature selection

    def __init__(self, name=None, replace=True):
        super(IdentityModel, self).__init__()
        self.name = name
        self.replace = replace






class MartingalFrc(IdentityModel):
    """Forecasts n_out Y values by last n_out values of X"""

    def fit(self, X, Y):
        self.n_out = Y.shape[1]
        return self

    def predict(self, X):
        return X[:, -self.n_out:]



class CustomModel(IdentityModel):

    def __init__(self, name=None, fitfunc=None, predictfunc=None, **kwargs):
        super(IdentityModel, self).__init__()
        self.name = name
        self.fitfunc = fitfunc
        self.predictfunc = predictfunc
        for k, v in kwargs.items():
            self.__setattr__(k, v)


    def fit(self, X, Y):
        if self.fitfunc is None:
            return self

        return self.fitfunc(self, X, Y)

    def predict(self, X):
        if self.predictfunc is None:
            return X

        return self.predictfunc(self, X)



def print_pipeline_pars(model):
    # model.steps is a list of  tuples ('stepname', stepmodel)
    for _, step_model in model.steps:
        if hasattr(step_model, 'print_pars'):
            step_model.print_pars()
        else:
            print(step_model.get_params())






