"""func.py
~~~~~~~~~~~~~~

Constains modularized functions used for nueral network training

Activation functions (modularized from tf.nn module:
linear, Relu, Sigmoid, tanh, softmax

Miscellaneous functions:
Size: Get number of examples in a dataset
dropout_layer: Create dropout layer using tf.nn.dropout

Convolutional layers functions:
Conv2d: 2d convolution operation
max+pool: 2x2 maxpooling layer

Loading data:
load_mnist_data_shared: Load pickled mnist dataset
NOTE: data is NOT one-hot encoded

"""
import pickle
import gzip
import tensorflow as tf
import numpy as np


# Modularized activation functions for neurons
def linear(z): return z
def ReLU(z): return tf.nn.relu(z) 
def leaky_relu(z): return tf.nn.leaky_relu(z)
def sigmoid(z): return tf.nn.sigmoid(z)
def tanh(z): return tf.nn.tanh(z)
def softmax(z): return tf.nn.softmax(z)


#### Miscellaneaous functions
def size(data): return data[0].shape[0]
def dropout_layer(layer, p_dropout): return tf.nn.dropout(layer, p_dropout)


# Modularized Convolution and pooling functions for convolution layers
def conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'):
    '''Padding is either SAME or VALID. According to TensorFlow docs, default is SAME
    x: Input tensor
    W: Filter tensor. 4D tensor shape of [filter_height, filter_width, in_channels, out_channels]
    Stride length is default 1
    '''
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)


def max_pool_2x2(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    '''Default is 2x2 max pool with stride length = 1
    '''
    return tf.nn.max_pool(x, ksize=ksize,
                          strides=strides, padding=padding)

###Load the MNIST Data
def load_mnist_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    def numpify(data):
        """Cast data into numpy array format
        """
        shared_x = np.asarray(data[0], dtype="float32")
        shared_y = np.asarray(data[1], dtype="int32")
        return shared_x, shared_y
    return [numpify(training_data), numpify(validation_data), numpify(test_data)]


        
###Exception Class to exit training if 100% accuracy is met
class GetOutOfLoop( Exception ):
    pass

