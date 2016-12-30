import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers


def rnn_data(data, time_steps, labels=False):

    data = np.array(data, np.float32)
    if not labels:
        data = data.reshape(data.shape[0], time_steps, data.shape[1])
    return data

def split_data(data, val_size=0.1):
    """
    splits data to training, validation parts
    """
    nval = int(round(len(data) * (1 - val_size)))
    df_train, df_val = data.iloc[:nval], data.iloc[nval:]
    return df_train, df_val

def prepare_data(data, time_steps, labels=False, val_size=0.1):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation for an lstm cell.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    df_train, df_val= split_data(data, val_size)
    return(rnn_data(df_train, time_steps, labels=labels), rnn_data(df_val, time_steps, labels=labels))

def mae(predicted, y, idx_rows):
    error = np.mean(np.abs(y[idx_rows] - predicted[idx_rows]))
    return np.round(error, 6)

def mape(predicted, y, idx_rows):
    denom = np.abs(y[idx_rows])
    denom[denom == 0] = np.mean(np.abs(y[idx_rows]))
    error = np.mean(np.divide(np.abs(y[idx_rows] - predicted[idx_rows]), denom))
    return np.round(error, 6)

def lstm_model(num_units, rnn_layers, learning_rate=0.1, optimizer='Adagrad'):
    """
    Creates a deep model based on:
        * stacked lstm cells
        * an optional dense layers
    :param num_units: the size of the cells.
    :param rnn_layers: list of int or dict
                         * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                         * list of dict: [{steps: int, keep_prob: int}, ...]
    :param dense_layers: list of nodes for each layer
    :return: the model definition
    """

    def lstm_cells(layers):
        if isinstance(layers[0], dict):
            return [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(layer['num_units'],
                                                                               state_is_tuple=True),
                                                  layer['keep_prob'])
                    if layer.get('keep_prob') else tf.nn.rnn_cell.BasicLSTMCell(layer['num_units'],
                                                                                state_is_tuple=True)
                    for layer in layers]
        return [tf.nn.rnn_cell.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]

    def _lstm_model(X, y):
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        x_ = tf.unpack(X, axis=1, num=num_units)
        output, layers = tf.nn.rnn(stacked_lstm, x_, dtype=dtypes.float32)
        prediction, loss = tflearn.models.linear_regression(output[-1], y)
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
            learning_rate=learning_rate)
        return prediction, loss, train_op

    return _lstm_model