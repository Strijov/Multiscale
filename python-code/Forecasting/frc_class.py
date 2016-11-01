""" Created on 23 September 2016. Author: Anastasia Motrenko"""
from __future__ import print_function

import os
import dill
# import cloudpickle

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
            if 'name' in kwargs.keys():
                name = kwargs['name']
                del kwargs['name']
            else:
                name = None
            IdentityModel.__init__(self, name)
            parent.__init__(self, *args, **kwargs)



        def fit(self, X, Y):
            """ Runs parent.fit after IndentityModel.fit"""
            IdentityModel.fit(self, X, Y)
            parent.fit(self, X, Y)

            return self

        def predict(self, X):
            """ Runs parent.predict after IndentityModel.predict"""
            IdentityModel.predict(self, X)
            Y = parent.predict(self, X)
            # if hasattr(parent, "predict"):
            #     Y = parent.predict(self, X)
            # elif hasattr(parent, "forecast"):
            #     Y = parent.forecast(self, X)
            # elif hasattr(parent, "transform"):
            #     Y = parent.transform(self, X)
            # else:
            #     print("{} class has neither of predict, forecast or transform methods".format(parent.__name__))
            #     raise AttributeError

            return Y

        def __reduce__(self):
            """
            A way to reproduce the the class instance

            :return: callable, arguments and object parameters
            :rtype: tuple
            """
            state = self.__dict__.copy()
            return (CustomModel, (parent, ), state)


    return CustomModel_()



class PipelineModel(Pipeline):
    """ A pipeline forecasting model"""

    def __init__(self, steps=None):
        if not steps == None:
            Pipeline.__init__(self, steps)


    def save_model(self, file_name="", folder="models"):
        """
        Saves the model to filename with specified prefix

        :param file_name: prefix for the file name, optional
        :type file_name: str
        :param folder: folder to strore the model in, optional; default = "models"
        :type folder: str
        :return: name of the saved file
        :rtype: str
        """

        if not os.path.exists(folder):
            os.makedirs(folder)

        file_name += "_".join(list(self.named_steps.keys()))
        file_name = os.path.join(folder, file_name + ".pkl")
        with open(file_name, "wb") as f:
            dill.dump(self, f)

        return file_name


    def load_model(self, file_name):
        """
        Loads the model from the specified file

        :param file_name: name of the file to load the data from
        :type file_name: str
        :return: self
        :rtype: PipelineModel
        """

        with open(file_name, "rb") as f:
            self = dill.load(f)

        return self

    def print_pipeline_pars(self):
        """ Formatted print for pipeline model  """

        # model.steps is a list of  tuples ('stepname', stepmodel)
        for _, step_model in self.steps:
            if hasattr(step_model, 'print_pars'):
                step_model.print_pars()
            else:
                print(step_model.get_params())






