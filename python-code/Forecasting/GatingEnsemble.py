""" Created on 06 October 2016. Author: Radoslav Neychev """
import lasagne
import theano
import theano.tensor as T
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

class GatingEnsemble:
    
    def __init__(self, estimators, estimator_loss = lambda y,y_pred:((y - y_pred)**2)):
        self.estimators = estimators
        self.estimator_loss = estimator_loss
    
    def fit(self,X,y,n_iter=100):
        
        self.compile_gf((None,)+X.shape[1:])
        params_vec = []
        k = 0
        for i in range(n_iter):
            self.refit_estimators(X,y)
            if i%10 == 0:
                print i, 'iteration done'
            self.refit_gf(X,y,10)

        #print "i want a better implementation"
        
    
    def refit_gf(self,X,y,n_iter=1):
        #for x in [np.sum(self.estimator_loss(y,est.predict(X)), axis = 1)
        #                        for est in self.estimators]:
        #    print x.shape
        best_estimator_ix = np.argmin([np.sum(self.estimator_loss(y,est.predict(X)), axis = 1)
                                 for est in self.estimators],axis=0).astype('int32')
        #print best_estimator_ix.shape
        for i in range(n_iter):
            self.fit_nn_step(X,best_estimator_ix)
        
    def refit_estimators(self,X,y):
        ix = np.random.randint(0,len(X),size=len(X))
        X,y = X[ix],y[ix]
        W = self.get_W(X)
        W = (W == np.max(W,axis=1,keepdims=True)).astype('float32')
        W /= W.sum(axis=0,keepdims=True)
        W[np.isnan(W)] = 1./len(W)
        
        
        self.estimators = [est.fit(X,y,sample_weight=W[:,i]) for i,est in enumerate(self.estimators)]
        self.get_est_params = [est.get_params() for est in self.estimators]
        
    def build_gating_function(self,x_shape,n_gates):
        
        #Input layer (auxilary)
        input_layer = lasagne.layers.InputLayer(shape = x_shape)

        #fully connected layer, that takes input layer and applies 50 neurons to it.
        # nonlinearity here is sigmoid as in logistic regression
        # you can give a name to each layer (optional)
        dense_1 = lasagne.layers.DenseLayer(input_layer,num_units=50, W=lasagne.init.Normal(0.01),
                                          nonlinearity = lasagne.nonlinearities.sigmoid,
                                          name = "hidden_dense_layer")

        #fully connected output layer that takes dense_1 as input and has 10 neurons (1 for each digit)
        #We use softmax nonlinearity to make probabilities add up to 1
        dense_output = lasagne.layers.DenseLayer(dense_1,num_units = n_gates,
                                                nonlinearity = lasagne.nonlinearities.softmax,
                                                name='output')
        
        return dense_output
    
    def compile_gf(self,x_shape):
        
        input_X = T.matrix("X")
        
        target_W_of_x = T.ivector("W(x) target - probability that i-th estimator is best")
        
        nn = self.nn = self.build_gating_function(x_shape,len(self.estimators))
        
        w_predicted = lasagne.layers.get_output(nn,inputs=input_X)
        
        loss = lasagne.objectives.categorical_crossentropy(w_predicted,target_W_of_x).mean()
        
        nn_params = lasagne.layers.get_all_params(self.nn,trainable=True)
        
        updates = lasagne.updates.adamax(loss,nn_params)
        
        self.fit_nn_step = theano.function([input_X,target_W_of_x],loss, updates=updates)
        self.get_W = theano.function([input_X], w_predicted)
    
    def predict(self,X):
        assert hasattr(self,'nn')
        
        
        W = self.get_W(X)
        
        #(sample i, est k)
        base_predictions = np.stack([est.predict(X) for est in self.estimators],axis=1)
        #print base_predictions.shape
        #print W.shape
        return (W[:,:,None]*base_predictions).sum(axis=1)
        
    