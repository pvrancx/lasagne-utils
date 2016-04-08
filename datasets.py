# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 21:51:29 2016

@author: pvrancx
"""

import sys
import os
import numpy as np
from collections import OrderedDict

from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme

#'''
#from lasagne examples
#'''
#def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
#    assert len(inputs) == len(targets)
#    if shuffle:
#        indices = np.arange(len(inputs))
#        np.random.shuffle(indices)
#    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
#        if shuffle:
#            excerpt = indices[start_idx:start_idx + batchsize]
#        else:
#            excerpt = slice(start_idx, start_idx + batchsize)
#        yield inputs[excerpt], targets[excerpt]
#
#class DataSet2(object):
#    
#    def __init__(self,X,y,batch_size=128,shuffle=False):
#        self.X = X
#        self.y = y
#        self.batch_size = batch_size
#        self.shuffle = shuffle
#        
#    def __iter__(self):
#        return iterate_minibatches(self.X,self.y,self.batch_size,self.shuffle)
        
class FuelDataSet(object):
    def __init__(self,dataset,batch_size=128,shuffle=False):
        self.dataset = dataset
        if shuffle:
            self.datastream = DataStream(self.dataset,
                                     iteration_scheme=ShuffledScheme(
                                     examples=dataset.num_examples,
                                     batch_size=batch_size))
        else:
            self.datastream = DataStream(self.dataset,
                                     iteration_scheme=SequentialScheme(
                                     examples=dataset.num_examples,
                                     batch_size=batch_size))

    def __iter__(self):
        return self.datastream.get_epoch_iterator()
        

class DataSet(FuelDataSet):
    def __init__(self,X,y,batch_size=128,shuffle=False):
        super(DataSet,self).__init__(IndexableDataset(
                                    indexables=OrderedDict([
                                    ('features', X), 
                                    ('targets', y)]))
                                    ,batch_size,shuffle)
    

def load_mnist():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test