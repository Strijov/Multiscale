# coding: utf-8
""" Created on 30 September 2016. Author: Maria Popova, Anastasia Motrenko """

import numpy as np
import theano
import theano.tensor as T
import lasagne
import pickle
import datetime
import _special_layers
import my_plots

class ShvetsRNN():
    
    def __init__(self, feature_dict, name="ShvetsRNN", grad_clip=100.0, batch_size=50, l_out=None, n_epochs=100, plot_loss=False,
                 num_lstm_units_large=35, num_lstm_units_small=5, learning_rate=1e-4):
        """

        :param name: reference name
        :type name: str
        :param grad_clip: max absolute gradient value
        :type grad_clip: float
        :param batch_size: number of rows in a batch
        :type batch_size: int
        :param l_out: output layer, optional
        :type l_out: lasagne.Layer
        :param n_epochs: number of training epochs
        :type n_epochs: int
        :param plot_loss: if True, plots loss by training epoch
        :type plot_loss: bool
        """
        self.name = name
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.l_out = l_out
        self.plot_loss = plot_loss
        self.learning_rate = learning_rate
        self.num_lstm_units_large = num_lstm_units_large
        self.num_lstm_units_small = num_lstm_units_small
        self.feature_dict = feature_dict

    def fit(self, trainX, trainY, n_epochs=None, fname=None, verbose=True, save_results=False):
        """
        Train module for LSTM network

        :param trainX: training data, features
        :type trainX: ndarray
        :param trainY: training data, targets
        :type trainY: ndarray
        :param n_epochs: number of training epochs
        :type n_epochs: int
        :param fname: filename for saving model parameters
        :type fname: str
        :param verbose: if True, outputs loss values while training
        :type verbose: bool
        :return: None
        """

        _, nX = trainX.shape
        _, nY = trainY.shape

        if self.l_out is None:
            self.init_nn_structure(int(nX), int(nY))

        print("Training ...")
        loss = []
        loss_msg = ""
        if not n_epochs is None:
            self.n_epochs = n_epochs
        for epoch in xrange(self.n_epochs):
            avg_cost = 0
            for batch in iterate_minibatches(trainX, trainY, self.batch_size):
                x, y = batch
                iter_loss = self.trainT(x.astype('float32'), y.astype('float32'))
                avg_cost += iter_loss
                #print iter_loss


            loss.append(avg_cost)

            msg = "Epoch {} average loss = {}".format(epoch, avg_cost)
            loss_msg += msg + "\n \\\\"

            if verbose:
                print(msg)
                
            if (epoch+1) % 500 == 0:
                self.lr.set_value(np.array(self.lr.get_value() / 2, dtype='float32'))
                print 'Learning rate:', self.lr.get_value()

        self.weights = lasagne.layers.get_all_params(self.l_out,trainable=True)
        if save_results:
            self.checkpoint(fname, loss=loss)
        self.msg = loss_msg

        if self.plot_loss:
            self.fig = my_plots.formatted_plot(loss, xlabel="Epoch", ylabel="Average loss (mse)")

    def predict(self, X):
        Y = self.forecast(X[:,:,None].astype('float32'))
        return Y


    def init_nn_structure(self, seq_length, pred_len):
        """
        Inits network structure

        :param seq_length: number of features
        :type seq_length: int
        :param pred_len: number of predicted values (target dimensionality)
        :type pred_len: int
        :return: None
        """
        self.iteration = 0
        
        theano_input = T.tensor3()
        theano_output = T.matrix()
        
        from lasagne.layers import InputLayer, LSTMLayer, DenseLayer, ExpressionLayer, ConcatLayer
        from lasagne.nonlinearities import tanh
        
        model = {}
        model['input_layer'] = InputLayer((None, seq_length, 1), input_var=theano_input)
        
        lst_concat = []
        for i, key in enumerate(self.feature_dict.keys()):
            if self.feature_dict[key] is None or len(self.feature_dict[key]) == 0:
                continue
            model['input_slice_' + str(i)] = ExpressionLayer(model['input_layer'], lambda X: X[:,self.feature_dict[key],:])
            num_units = self.num_lstm_units_large if len(self.feature_dict[key]) > 10 else self.num_lstm_units_small
            model['hidden_layer_' + str(i) + '_1'] = LSTMLayer(model['input_slice_' + str(i)], 
                               num_units, grad_clipping=self.grad_clip, nonlinearity=tanh)
            model['hidden_layer_' + str(i) + '_2'] = LSTMLayer(model['hidden_layer_' + str(i) + '_1'], 
                               num_units, grad_clipping=self.grad_clip, nonlinearity=tanh, only_return_final=True)
            lst_concat.append(model['hidden_layer_' + str(i) + '_2'])
        model['concatenate_hidden'] = ConcatLayer(lst_concat, axis=1)
        model['output_layer'] = DenseLayer(model['concatenate_hidden'], pred_len, nonlinearity=None)
        
        model_output = lasagne.layers.get_output(model['output_layer'])
        params = lasagne.layers.get_all_params(model['output_layer'], trainable=True)

        self.loss = lasagne.objectives.squared_error(model_output, theano_output).mean()
        self.lr = theano.shared(np.array(self.learning_rate, dtype='float32'))
        self.updates = lasagne.updates.adam(self.loss, params, learning_rate=self.lr)

        self.l_out = model['output_layer']
        self.trainT = theano.function([theano_input, theano_output], self.loss, updates=self.updates)
        self.compute_cost = theano.function([theano_input, theano_output], self.loss)
        self.forecast = theano.function([theano_input], model_output)
        
        '''
        input_sequence = T.matrix('input sequence')
        target_values = T.matrix('target y')

        l_in = lasagne.layers.InputLayer(shape=(None, seq_length),input_var=input_sequence)

        l1 = _special_layers.ExpressionLayer(l_in, lambda X: T.repeat(X.mean(axis=1), pred_len), output_shape=(None, pred_len))
        l1 = lasagne.layers.ReshapeLayer(l1, shape=(-1, pred_len))

        l2 = _special_layers.ExpressionLayer(l_in, lambda X: T.repeat(X.std(axis=1), pred_len), output_shape=(None, pred_len))
        l2 = lasagne.layers.ReshapeLayer(l2, shape=(-1, pred_len))

        l3 = _special_layers.ExpressionLayer(l_in, lambda X:
                                            ((X.reshape([1, X.shape[0]*X.shape[1]]) - T.repeat(X.mean(axis=1), seq_length))/
                                             T.repeat(X.std(axis=1), seq_length)).reshape(X.shape),
                                            output_shape=(None, seq_length))

        l4 = lasagne.layers.ReshapeLayer(l3, shape=(-1, seq_length, 1))

        l_rnn = lasagne.layers.LSTMLayer(l4, num_units=self.num_lstm_units, grad_clipping=self.grad_clip,
                                            nonlinearity=lasagne.nonlinearities.tanh)
        l_rnn = lasagne.layers.LSTMLayer(l_rnn, num_units=self.num_lstm_units, grad_clipping=self.grad_clip,
                                            nonlinearity=lasagne.nonlinearities.tanh)

        l_out_norm = lasagne.layers.DenseLayer(l_rnn, num_units=pred_len, nonlinearity=lasagne.nonlinearities.linear)

        l_out_mul = lasagne.layers.ElemwiseMergeLayer([l_out_norm, l2], merge_function = T.mul)
        l_out = lasagne.layers.ElemwiseSumLayer([l_out_mul, l1])

        weights = lasagne.layers.get_all_params(l_out,trainable=True)
        network_output = lasagne.layers.get_output(l_out)
        

        network_output_norm = lasagne.layers.get_output(l_out_norm)
        std = lasagne.layers.get_output(l2)

        self.loss = T.mean(lasagne.objectives.squared_error(network_output/std,
                                                            target_values/std))
        self.updates = lasagne.updates.adam(self.loss, weights, learning_rate=self.learning_rate)

        self.trainT = theano.function([input_sequence, target_values], self.loss,
                                      updates=self.updates, allow_input_downcast=True)
        self.compute_cost = theano.function([input_sequence, target_values], self.loss,
                                            allow_input_downcast=True)

        #forecasting next timestep
        self.forecast = theano.function([input_sequence],network_output,
                                        allow_input_downcast=True)

        '''


    def checkpoint(self, fname, **kwargs):
        """
        Saves parameters for a file

        :param fname: filename
        :type fname: str
        :param kwargs: named parameters to save
        :return: None
        """
        results = {}
        params_names = lasagne.layers.get_all_params(self.l_out, trainable=True)
        params = lasagne.layers.get_all_param_values(self.l_out)

        for name, par in zip(params_names, params):
            results[name] = par

        for k, v in kwargs.items():
            results[k] = v

        if fname is None:
            fname = "tmp/ShvetsRNN_weights_" + str(datetime.date.today())

        pickle.dump(params, open(fname, 'wb'))



def iterate_minibatches(X, Y, batch_size):
    """
    Generates batch_size batches from X and Y
    """

    m = X.shape[0]

    ind = np.random.permutation(m).tolist()
    k = 0
    X_batch = np.zeros(shape=(batch_size, X.shape[1], 1))
    y_batch = np.zeros(shape=(batch_size, Y.shape[1]))
    for start_index in ind:
        if k == batch_size:
            yield (X_batch, y_batch)
            X_batch = np.zeros(shape=(batch_size, X.shape[1], 1))
            y_batch = np.zeros(shape=(batch_size, Y.shape[1]))
            k = 0
        else:
            X_batch[k, :, 0] = X[start_index, :]
            y_batch[k] = Y[start_index, :]
            k = k + 1




