# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 13:05:07 2016

@author: pvrancx
"""

import theano
import theano.tensor as T
import lasagne
import numpy as np
import time

from datasets import DataSet



'''
A simple stack of layers
'''
class Sequential(object):
    def __init__(self,input_shape,
                 input_var=T.matrix('X'),
                 output_var = T.vector('y'), layers=[]):
                     
        self.layers = []
        self.input_var = input_var
        self.output_var = output_var
        self.l1_decays = {}
        self.l2_decays = {}
        self.layers.append(
            lasagne.layers.InputLayer(input_shape,self.input_var)
        )
        for l,p in layers:
            self.add(l,**p)
    
    def add_regularization(self,layer_id,lambda1=0.,lambda2=0.):
        if  lambda1 > 0.:
            self.l1_decays[self.layers[layer_id]] = lambda1
        if lambda2 > 0.:
            self.l2_decays[self.layers[layer_id]] = lambda2
        
            
    def add(self,layer_type,lambda1=0.,lambda2=0.,**kwargs):
        l = layer_type(self.layers[-1],**kwargs)
        self.layers.append( l )
        self.add_regularization(-1,lambda1,lambda2)
        return l


        
    def get_output_fn(self,layer_id=-1, 
                      deterministic=True, 
                      prev_layers = False):
        target_layer = self.layers[ layer_id ]
        outputs = None
        if prev_layers:
            outputs = []
            all_layers = lasagne.layers.get_all_layers(target_layer)
            for l in all_layers:
                outputs.append(lasagne.layers.get_output(l,deterministic))
        else:
            outputs = lasagne.layers.get_output(target_layer,
                                                deterministic=deterministic)
        return theano.function([self.input_var],outputs)
        
    def compile(self,loss_fn,optimizer,**kwargs):
        train_output = lasagne.layers.get_output(self.layers[-1],
                                                 deterministic=False)
        test_output = lasagne.layers.get_output(self.layers[-1],
                                                deterministic=True)
        #base loss
        train_loss = loss_fn(train_output,self.output_var)
        test_loss = loss_fn(test_output,self.output_var)
        
        #add regularization
        l1_penalty = lasagne.regularization.regularize_layer_params_weighted(   
                    self.l1_decays,
                    lasagne.regularization.l1)
        l2_penalty = lasagne.regularization.regularize_layer_params_weighted(   
                    self.l2_decays,
                    lasagne.regularization.l2)
        train_loss += l1_penalty + l2_penalty
        train_loss = train_loss.mean()
                    
        test_loss +=  l1_penalty + l2_penalty
        test_loss = test_loss.mean()
                
        
        self.params = lasagne.layers.get_all_params(self.layers[-1], 
                                                    trainable=True)
        updates = optimizer(train_loss, self.params,**kwargs)
        
        self._test_fn = theano.function([self.input_var, self.output_var],
                                        [test_output,test_loss],
                                        )
        self._train_fn = theano.function([self.input_var, self.output_var],
                                         [train_output,train_loss],
                                          updates=updates)
        self._pred_fn = self.get_output_fn()
        
    def predict(self,X):
        return self._pred_fn(X)
        
    def test_batch(self,X,y):
        return self._test_fn(X,y)
        
    def train_batch(self,X,y):
        return self._train_fn(X,y)
        
    def score(self,X,y,batch_size = 128):
        test_data = DataSet(X,y,batch_size,shuffle=True)
        return self.score_iterator(test_data)
        
    def score_iterator(self,iterator):
        val_err = 0
        val_batches = 0
        for batch in iterator:
            inputs, targets = batch
            _,err = self.test_batch(inputs, targets)
            val_err += err
            val_batches += 1
        val_err /= val_batches
        return val_err
                                          
    def fit(self, X,y,batch_size= 128, n_epochs=10, validation_data = None, 
            loglevel =0, validation_split = 0, early_stopping = True,
            initial_patience=5,min_improvement=.995,patience_increase=2):
        
        n_samples = X.shape[0]
        
        val_iterator = None
        if (validation_data is None) and (0. < validation_split < 1.):
            val_size = np.floor(validation_split*n_samples)
            val_idx = np.random.choice(n_samples,
                                       size = val_size,
                                       replace = False
                                       )
            train_idx = np.delete(np.arange(n_samples), val_idx)
            X_val, y_val = X[val_idx,],y[val_idx,]
            X_train, y_train = X[train_idx,],y[train_idx,]
        elif validation_data is not None:
             X_val, y_val = validation_data
             X_train, y_train = X,y
             val_iterator = DataSet(X_val,y_val,shuffle=True)
        else:
             X_train, y_train = X,y
            
        train_iterator = DataSet(X_train,y_train, batch_size, shuffle=True)
        self.fit_iterator(train_iterator,n_epochs,val_iterator,loglevel,
                          early_stopping,initial_patience,min_improvement,
                          patience_increase)
                          
    def fit_iterator(self, train_iterator, 
                      n_epochs=10, 
                      val_iterator = None, 
                      loglevel =0, 
                      early_stopping = True,
                      initial_patience=10,
                      min_improvement=.995,
                      patience_increase=2):                                          

        
        
        if early_stopping:
            assert val_iterator is not None,'early stopping requires validation set'
            
        #based on lassagne docs
        start_training = time.time()
        train_losses = []
        val_losses = []
        
        patience = initial_patience
        total_batches = 0
        best_loss = np.inf

        for epoch in range(n_epochs):
            train_loss = 0
            train_batches = 0
            start_time = time.time()
            
            #full pass through data
            for batch in train_iterator:
                inputs, targets = batch
                _,b_loss = self.train_batch(inputs, targets)
                train_loss += b_loss
                train_batches += 1
                
            train_losses.append(train_loss / train_batches)
                
            total_batches += train_batches
            
            #test on validation data
            if val_iterator is not None:
                val_losses.append(self.score_iterator(val_iterator))


            if early_stopping:
                if val_losses[-1] < best_loss:
                    if val_losses[-1] < min_improvement * best_loss:
                        patience = max(patience,patience_increase
                                *total_batches)
                    best_loss = val_losses[-1]
                if total_batches >= patience:
                    break

                    
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_losses[-1]))
            if val_iterator is not None:
                print("  validation loss:\t\t{:.6f}".format(val_losses[-1]))

                
                
            
            
        