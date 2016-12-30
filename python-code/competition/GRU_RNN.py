""" Created on 30 December 2016. Author: Renat Khayrullin"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import GRU
import numpy as np


class GRU_Rnn():

    def __init__(self, epoch_num, batch_size):
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.rnn_model = None
        self.name = 'GRU_RNN'
        self.is_fitted = False

    def init_model(self, input_len, out_len):
        rnn_model = Sequential()
        #rnn_model.add(LSTM(10, input_shape=(input_len, 1)))
        rnn_model.add(GRU(10, input_shape=(input_len, 1)))
        rnn_model.add(Dense(out_len))
        rnn_model.compile(loss='mean_squared_error', optimizer='adam')
        return rnn_model

    def fit(self, trainX, trainY):

        _, len_input = trainX.shape
        _, len_output = trainY.shape

        self.rnn_model = self.init_model(int(len_input), int(len_output))

        X = np.zeros(shape=(trainX.shape[0], trainX.shape[1], 1))
        Y = np.zeros(shape=(trainY.shape[0], trainY.shape[1]))
        X[:, :, 0] = trainX[:, :]
        Y[:] = trainY[:, :]

        self.rnn_model.fit(X, Y, nb_epoch=self.epoch_num, batch_size=self.batch_size, verbose=2)
        self.is_fitted = True
        return self

    def predict(self, predictX):
        predicted = self.rnn_model.predict(predictX[:, :, None].astype('float32'))
        return predicted
