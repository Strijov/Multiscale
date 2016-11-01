""" Created on 23 September 2016. Author: Anastasia Motrenko"""
from __future__ import print_function

import os
import pickle

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline




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
        params = self.__dict__
        print(self.name)
        for par in params:
            #if not "__" in par:
            print(par, ":", self.__getattribute__(par))

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



def CustomModel(parent, *args, **kwargs):
    """
    Defines a new class with double inheritance

    :param parent: Parent class
    :param args: Sequential arguments
    :param kwargs: Optional keyword arguments
    :return: instance of CustomModel class
    """
    class CustomModel_(parent, IdentityModel):

        def __init__(self):
            # self.__objclass__ = parent # do smth about this
            # self.__name__ = parent.__name__
            if 'name' in kwargs.keys():
                name = kwargs['name']
                del kwargs['name']
            else:
                name = None
            IdentityModel.__init__(self, name)
            parent.__init__(self, *args, **kwargs)


        def fit(self, X, Y):
            IdentityModel.fit(self, X, Y)
            parent.fit(self, X, Y)

            return self

        def predict(self, X):
            IdentityModel.predict(self, X)
            Y = parent.predict(self, X)

            return Y


    return CustomModel_()


def _reduce_wrapper_descriptor(m): # for serialization
    return getattr, (m.__objclass__, m.__name__)

class PipelineModel(Pipeline):

    def __init__(self, steps=None):
        if not steps == None:
            Pipeline.__init__(self, steps)


    def save_model(self, file_name="", folder="models"):

        import cloudpickle
        try:
            import copy_reg
        except ImportError:
            import copyreg as copy_reg

        if not os.path.exists(folder):
            os.makedirs(folder)

        # copy_reg.pickle(type(self.named_steps['frc']), _reduce_wrapper_descriptor)

        file_name += "_".join(list(self.named_steps.keys()))
        file_name = os.path.join(folder, file_name + ".pkl")
        with open(file_name, "wb") as f:
            cloudpickle.dump(self, f)



        return file_name


    def load_model(self, file_name):

        with open(file_name, "rb") as f:
            self = pickle.load(f)

        return self

    def print_pipeline_pars(self):
        """ Formatted print for pipeline model  """

        # model.steps is a list of  tuples ('stepname', stepmodel)
        for _, step_model in self.steps:
            if hasattr(step_model, 'print_pars'):
                step_model.print_pars()
            else:
                print(step_model.get_params())



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


