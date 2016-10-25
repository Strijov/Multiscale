from __future__ import print_function

import numpy as np
from sklearn.base import BaseEstimator


class IdentityModel(BaseEstimator):
    """ Base class for prediction, feature selection and generation """

    def __init__(self, name=None):
        self.name = name
        self.is_fitted = False

    def fit(self, X, Y):
        """
        Training method, override it in your model

        :param X: training data, features
        :type X: ndarray
        :param Y: trainning data, targets
        :type Y: ndarray
        :return: self
        """
        return self

    def predict(self, X):
        """
        Prediction method, override it in your model

        :param X: input data
        :type X: ndarray
        :return: forecasted data
        :rtype: ndarray
        """

        return X

    def transform(self, X):
        """
        Duplicates self.predict

        :param X:
        :type X:
        :return:
        :rtype:
        """
        return self.predict(X)

    def print_pars(self):
        params = self.get_params()
        print(self.name)
        for k, v in params.items():
            print(k, ":", v)

class IdentityFrc(IdentityModel):
    """ Helper class, used for testing purposes"""

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
    """ Helper class for feature generation"""

    def __init__(self, name=None, replace=True):
        super(IdentityGenerator, self).__init__(name)
        self.replace = replace


class MartingalFrc(IdentityModel):
    """Forecasts n_out Y values by last n_out values of X"""

    def fit(self, X, Y):
        self.n_out = Y.shape[1]
        return self

    def predict(self, X):
        return X[:, -self.n_out:]


def print_pipeline_pars(model):
    """ Formatted print for pipeline model  """

    # model.steps is a list of  tuples ('stepname', stepmodel)
    for _, step_model in model.steps:
        if hasattr(step_model, 'print_pars'):
            step_model.print_pars()
        else:
            print(step_model.get_params())


def CustomModel(parent, *args, **kwargs):
    """
    Defines a new class with double inheritance

    :param parent: Parent class
    :param args: Sequential arguments
    :param kwargs: Optional keyword arguments
    :return: instance of CustomModel class
    """
    class CustomModel(parent, IdentityModel):

        def __init__(self):
            if 'name' in kwargs.keys():
                name = kwargs['name']
                del kwargs['name']
            else:
                name = None
            IdentityModel.__init__(self, name)
            parent.__init__(self, *args, **kwargs)
            # for k, v in kwargs.items():
            #     self.__setattr__(k, v)

        def fit(self, X, Y):
            IdentityModel.fit(self, X, Y)
            parent.fit(self, X, Y)

            return self

        def predict(self, X):
            IdentityModel.predict(self, X)
            Y = parent.predict(self, X)

            return Y


    return CustomModel()

# class parent():
#
#     def __init__(self):
#         self.a = "a"
#         self.b = 2
#
#     def fit(self, X, Y):
#         self.is_fitted = "here"
#         self.b += 2
#
#     def predict(self, X):
#         self.my_print()
#
#     def my_print(self):
#         print(self.a)

# my_model = CustomModel(parent, c=1)
# my_model.fit([], [])
# my_model.predict([])


