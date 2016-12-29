""" Created on 23 September 2016. Author: Anastasia Motrenko"""
from __future__ import print_function

import os
import dill

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from itertools import product
from Features import generation_models as gnt_class
from Features import quadratic_feature_selection as sel_class


class IdentityModel(BaseEstimator):
    """ Base class for prediction, feature selection and generation """

    def __init__(self, name=None):
        if name is None:
            name = "Identity"
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
            # if not "__" in par:
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
                name = parent.__name__
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

            return Y

        def __reduce__(self):
            """
            A way to reproduce the the class instance

            :return: callable, arguments and object parameters
            :rtype: tuple
            """
            state = self.__dict__.copy()
            return (CustomModel, (parent, args, kwargs,), state)
    return CustomModel_()


class PipelineModel(Pipeline):
    """ A pipeline forecasting model"""

    def __init__(self, steps=None, frc_mdl=None, gen_mdl=None, sel_mdl=None):
        # if steps is None and frc_mdl is None:
        #     raise ValueError("Steps are not defined in Pipeline model")

        if steps is None:
            steps = [('gen', gen_mdl), ('sel', sel_mdl), ('frc', frc_mdl)]

        named_steps = {k: v for k, v in steps}

        if named_steps['frc'] is None:
            frc_mdl = IdentityModel(name="Identity")
        if named_steps['sel'] is None:
            sel_mdl = sel_class.FeatureSelection(name="No feature selection", on=False)
        if named_steps['gen'] is None:
            gen_mdl = gnt_class.FeatureGeneration(name="No feature generation")

        steps = [('gen', gen_mdl), ('sel', sel_mdl), ('frc', frc_mdl)]
        Pipeline.__init__(self, steps)
        self.name = "_".join([str(frc_mdl.name), str(gen_mdl.name), str(sel_mdl.name)])

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

    @staticmethod
    def load_model(file_name):
        """
        Loads the model from the specified file

        :param file_name: name of the file to load the data from
        :type file_name: str
        :return: self
        :rtype: PipelineModel
        """

        with open(file_name, "rb") as f:
            model = dill.load(f)

        return model

    def print_pipeline_pars(self):
        """ Formatted print for pipeline model  """

        # model.steps is a list of  tuples ('stepname', stepmodel)
        for _, step_model in self.steps:
            if hasattr(step_model, 'print_pars'):
                step_model.print_pars()
            else:
                print(step_model.get_params())

    def train_model(self, train_x, train_y, retrain=True, hyperpars=None, n_cvs=5):
        """
        Initializes and train feature generation, selection and forecasting model in a pipeline

        :param train_x: training data, inputs
        :type train_x: numpy.ndarray
        :param train_y: training data, targets
        :type train_y: numpy.ndarray
        :param retrain: Flag that specifies if the model needs retraining
        :type retrain: bool
        :param hyperpars: defines ranges of hyperparameters to tune with cross-validation. If None is specified,
        no hyperparameters will be optimized
        :type hyperpars: dict
        :param n_cvs: number of folds in k-fold cross-validation
        :type n_cvs: int
        :return: trained model, forecasting model, generator and selector
        :rtype: tuple
        """


        # once fitted, the model is retrained only if retrain = True
        if self.named_steps['frc'].is_fitted and not retrain:
            return self, self.named_steps['frc'], self.named_steps['gen'], self.named_steps['sel']

        # if a range of hyperparametr values is specify, tune it via k-fold cross-validation
        if hyperpars is not None:
            best_hyperpars = cv_train(self, train_x, train_y, hyperpars, n_cvs)
            for k, v in zip(hyperpars.keys(), best_hyperpars):
                self.named_steps['frc'].__setattr__(k, v)

        self.fit(train_x, train_y)

        return self, self.named_steps['frc'], self.named_steps['gen'], self.named_steps['sel']


def cv_train(raw_model, X, Y, hyperpars, n_cvs):
    """
    Use cross-validation to choose optimal hyperparameter values from the given range

    :param raw_model: model
    :type raw_model: instance of PipelineModel
    :param X: training data, inputs
    :type X: numpy.ndarray
    :param Y: training data, targets
    :type Y: numpy.ndarray
    :param hyperpars: contains ranges of frc_model hyperparameters
    :type hyperpars: dict
    :param n_cvs: number of cross-validation splits
    :type n_cvs: int
    :return:
    :rtype: list
    """
    from sklearn.model_selection import KFold
    import sys
    if n_cvs is None:
        n_cvs = 5

    kf = KFold(n_splits=n_cvs)
    kf.get_n_splits(X)

    par_names = list(hyperpars.keys())
    par_values_range = list(hyperpars.values())
    scores = []
    for i, hyperpars in enumerate(product(*par_values_range)):
        scores.append(np.zeros(n_cvs))
        pars = {key: val for key, val in zip(par_names, hyperpars)}

        for k, train_val_index in enumerate(kf.split(X)):
            model = raw_model
            for key, val in pars.items():
                model.named_steps["frc"].__setattr__(key, val)
            print("\r{}, kfold = {}".format(pars, k), end="")
            sys.stdout.flush()
            # getting training and validation data
            train_index, val_index = train_val_index[0], train_val_index[1]
            x_train, x_val = X[train_index], X[val_index]
            y_train, y_val = Y[train_index], Y[val_index]
            # train the model and predict the MSE
            try:
                model.fit(x_train, y_train)
                pred_val = model.predict(x_val)
                scores[-1][k] = mean_squared_error_(pred_val, y_val)
            except BaseException as e:
                print(e)
                if k > 0:
                    scores[-1][k] = scores[-1][k-1]
                else:
                    scores[-1][k] = 1

        scores[-1] = np.mean(scores[-1])
    idx = np.argmin(scores)
    best_hyperpars = list(product(*par_values_range))[idx]
    print("Best hyperpars combo: {} with mse {}".format(zip(par_names, best_hyperpars), scores[idx]))

    return best_hyperpars

def mean_squared_error_(f, y):
    return np.mean(np.power(f - y, 2))
