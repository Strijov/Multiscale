# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import lasagne
import pickle
import datetime

class LSTM():


    def __init__(self, name="LSTM", grad_clip=100, batch_size=10, l_out=None):
        self.name = name
        self.grad_clip = 100
        self.batch_size = 10
        self.l_out = l_out

    def fit(self, trainX, trainY, seq_length=240, n_epochs=100, batch_size=500, name=None):
        print("Training ...")
        _, nX = trainX.shape
        _, nY = trainY.shape

        trainX = trainX.ravel()
        trainY = trainY.ravel()

        if self.l_out is None:
            self.init_nn_structure(nX*batch_size, nY*batch_size)

        loss = []
        for epoch in xrange(n_epochs):
            avg_cost = 0
            for batch in iterate_minibatches(trainX, trainY, self.batch_size):
                x, y = batch
                avg_cost += self.trainT(x, y)

            loss.append(avg_cost)

            print("Epoch {} average loss = {}".format(epoch, avg_cost))

        self.weights = lasagne.layers.get_all_params(self.l_out,trainable=True)
        self.checkpoint(name)


    def predict(self, X):
        m, n = X.shape

        Y = self.forecast(X)
        nY = int(Y.size/m)
        return Y.reshape(m, nY)


    def init_nn_structure(self, seq_length, pred_len):
        input_sequence = T.matrix('input sequence')
        target_values = T.matrix('target y')

        l_in = lasagne.layers.InputLayer(shape=(None, seq_length),input_var=input_sequence)

        l1 = lasagne.layers.ExpressionLayer(l_in, lambda X: T.repeat(X.mean(axis=1), pred_len), output_shape=(None, pred_len))
        output_shape = l1._output_shape
        l1 = lasagne.layers.ReshapeLayer(l1, shape=(output_shape, pred_len))

        l2 = lasagne.layers.ExpressionLayer(l_in, lambda X: T.repeat(X.std(axis=1), pred_len), output_shape=(None, pred_len))
        l2 = lasagne.layers.ReshapeLayer(l2, shape=(-1, pred_len))

        l3 = lasagne.layers.ExpressionLayer(l_in, lambda X:
                                            ((X.reshape([1, X.shape[0]*X.shape[1]]) - T.repeat(X.mean(axis=1), seq_length))/
                                             T.repeat(X.std(axis=1), seq_length)).reshape(X.shape),
                                            output_shape=(None, seq_length))

        l4 = lasagne.layers.ReshapeLayer(l3, shape=(-1, seq_length, 1))

        l_rnn = lasagne.layers.LSTMLayer(l4, num_units=50, grad_clipping=self.grad_clip,
                                            nonlinearity=lasagne.nonlinearities.tanh)
        l_rnn = lasagne.layers.LSTMLayer(l_rnn, num_units=50, grad_clipping=self.grad_clip,
                                            nonlinearity=lasagne.nonlinearities.tanh)

        l_out_norm = lasagne.layers.DenseLayer(l_rnn, num_units=pred_len, nonlinearity=lasagne.nonlinearities.linear)

        l_out_mul = lasagne.layers.ElemwiseMergeLayer([l_out_norm, l2], merge_function = T.mul)
        l_out = lasagne.layers.ElemwiseSumLayer([l_out_mul, l1])

        weights = lasagne.layers.get_all_params(l_out,trainable=True)
        network_output = lasagne.layers.get_output(l_out)
        self.l_out = l_out

        network_output_norm = lasagne.layers.get_output(l_out_norm)
        std = lasagne.layers.get_output(l2)

        self.loss = T.mean(lasagne.objectives.squared_error(network_output/std,
                                                            target_values/std))
        self.updates = lasagne.updates.adam(self.loss, weights, learning_rate=2e-4)

        self.trainT = theano.function([input_sequence, target_values], self.loss,
                                      updates=self.updates, allow_input_downcast=True)
        self.compute_cost = theano.function([input_sequence, target_values], self.loss,
                                            allow_input_downcast=True)

        #forecasting next timestep
        self.forecast = theano.function([input_sequence],network_output,
                                        allow_input_downcast=True)




    def checkpoint(self, name):
        params = lasagne.layers.get_all_param_values(self.l_out)

        if name is None:
            name = "last_weights_" + str(datetime.date.today())

        pickle.dump(params, open(name, 'wb'))



def iterate_minibatches(X, Y, batch_size=10):
    
    m = X.shape[0]
    #n_batches = int(m/batch_size)
    for i in range(m-batch_size):
        X_batch = X[i:i+batch_size]
        Y_batch = Y[i:i+batch_size]
        yield (X_batch, Y_batch)

       









#updates = lasagne.updates.nesterov_momentum(loss, weights, learning_rate=2e-4, momentum=0.9)




# # Function for saving trained parameters

# In[99]:








