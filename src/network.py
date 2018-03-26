"""network.py
~~~~~~~~~~~~~~
Written by Yong Yu Wen, 2018

(Built using tensorflow-gpu 1.6.0)

A TensorFlow-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

This program incorporates ideas from the Theano-based network from
http://http://neuralnetworksanddeeplearning.com/, by Michael Nielsen,
as well as Tensorflow documentations on convolutional neuralnetworks

"""



#### Libraries
# Standard library
import time

#Third-party libraries
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

#Import self-defined functions
from func import * #Bad practice but for ease of access of all self defined functions



#### Main class used to construct and train networks
class Network(object):
    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = tf.placeholder("float32", name = "x")
        self.y = tf.placeholder("int32", name = "y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout



    def train(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data=None, store_accuracies=False, save_dir=None, calibration=False, lmbda=0.0):
        '''Trains the neural network using Kingma and Ba's Adam algorithm (by default)
        Other optimization algorithms such as stochastic gradient descent can also be used. To do so, edit the


        (REQUIRED) training_data: training data in numpy array format
        (REQUIRED) epochs: number of training epochs
        (REQUIRED) mini_batch_size: size of mini batch for stochastic gradient descent
        (REQUIRED) eta: learning rate (Note: learning rate for GradientDescentOptimizer is ~100x larger than AdamOptimizer)
        (REQUIRED) validation_data: validation data in numpy array format
        test_data: test data in numpy array format
        store_accuracies: If True, stores the train, validation and test accuracies (Can be used for plotting against epoch
                            for model calibration and hyperparameter tuning) NOTE: train and test accuracies will be stored only if calibration = True
        save_dir: Directory to store/load data. If None, stores data in default directory "/tmp/model.ckpt"
        calibration: If True, will calculate train and test accuracy for every epoch. NOTE: Should be disabled unless when calibrating as
                     it will greatly slow down training.
        lmbda: Regularization parameter for l2 regularization

        :return: None
        
        '''

        #Initialize accuracies list
        if store_accuracies:
            self.validation_accuracies = []
            if calibration:
                self.train_accuracies = [] 
                self.test_accuracies=[]


        # compute number of minibatches for training, validation and testing
        num_training_batches = int(size(training_data)/mini_batch_size)
        num_validation_batches = int(size(validation_data)/mini_batch_size)
        if test_data:
            num_test_batches = int(size(test_data)/mini_batch_size)


        # define the (regularized) cost function
        l2_norm_squared = sum(tf.nn.l2_loss(layer.w) for layer in self.layers)
        cost = self.layers[-1].cost(self)+\
               lmbda*l2_norm_squared/tf.cast(num_training_batches, dtype=tf.float32)

        # Define optimizer
        with tf.name_scope('optimizer'):
            #train_step = tf.train.GradientDescentOptimizer(eta).minimize(cost)
            train_step = tf.train.AdamOptimizer(eta).minimize(cost)

        # Define minibatch accuracy operation
        ## Used to get train, validate and test accuracies in session
        mb_accuracy = self.layers[-1].accuracy(self.y)


        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        #Loading of data
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        if test_data:
            test_x, test_y = test_data

        #~~~~~Do the actual training
        with tf.Session() as sess:

            #Initialize best accuracies
            best_validation_accuracy = 0
            best_iteration = 0
            test_accuracy = 0

            if save_dir:
                try:
                    print("\n\nSearching for stored model")
                    saver.restore(sess, save_dir)
                    print("Model restored.\n\n")
                except:
                    print("No model of specified name found")
                    print("Initializing new model...\n\n")
                    sess.run(tf.global_variables_initializer())
            else:
                print("\n\nInitializing new model...\n\n")
                sess.run(tf.global_variables_initializer())

            start_time = time.time() #Track time taken for model

            try:
                for epoch in range(epochs):
                    for minibatch_index in range(num_training_batches):
                        iteration = num_training_batches*epoch+minibatch_index
                        if iteration % 1000 == 0:
                            print("Training mini-batch number {0}".format(iteration))

                        #Training of the model
                        train_step.run(feed_dict={self.x:
                                                      training_x[minibatch_index*self.mini_batch_size: (minibatch_index+1)*self.mini_batch_size],
                                                  self.y:
                                                      training_y[minibatch_index*self.mini_batch_size: (minibatch_index+1)*self.mini_batch_size]})

                        # Calculate and storing of Accuracies
                        if (iteration+1) % num_training_batches == 0:
                            if calibration:
                                train_accuracy = np.mean(
                                    [mb_accuracy.eval(feed_dict={self.x:
                                                          training_x[minibatch_index*self.mini_batch_size: (minibatch_index+1)*self.mini_batch_size],
                                                      self.y:
                                                          training_y[minibatch_index*self.mini_batch_size: (minibatch_index+1)*self.mini_batch_size]}
                                                          ) for j in range(num_training_batches)])
                                print("Epoch {0}: train accuracy {1:.2%}".format(
                                    epoch, train_accuracy))
                                if store_accuracies:
                                    self.train_accuracies.append(train_accuracy)


                            validation_accuracy = np.mean(
                                [mb_accuracy.eval(feed_dict = {self.x:
                                                                   validation_x[j*self.mini_batch_size: (j+1)*self.mini_batch_size],
                                                               self.y:
                                                                   validation_y[j*self.mini_batch_size: (j+1)*self.mini_batch_size]
                                                               }) for j in range(num_validation_batches)])
                            print("Epoch {0}: validation accuracy {1:.2%}".format(
                                epoch, validation_accuracy))
                            if store_accuracies:
                                self.validation_accuracies.append(validation_accuracy)

                            if calibration:
                                if test_data:
                                    test_accuracy = np.mean(
                                        [mb_accuracy.eval(feed_dict = {self.x:
                                                                                test_x[j*self.mini_batch_size: (j+1)*self.mini_batch_size],
                                                                            self.y:
                                                                                test_y[j*self.mini_batch_size: (j+1)*self.mini_batch_size]
                                                                            }) for j in range(num_test_batches)])
                                    print('The corresponding test accuracy is {0:.2%}'.format(
                                        test_accuracy))
                                    if store_accuracies:
                                        self.test_accuracies.append(test_accuracy)



                            if validation_accuracy >= best_validation_accuracy:
                                print("This is the best validation accuracy to date.")
                                best_validation_accuracy = validation_accuracy
                                best_iteration = iteration
                                if test_data:
                                    test_accuracy = np.mean(
                                        [mb_accuracy.eval(feed_dict = {self.x:
                                                                                test_x[j*self.mini_batch_size: (j+1)*self.mini_batch_size],
                                                                            self.y:
                                                                                test_y[j*self.mini_batch_size: (j+1)*self.mini_batch_size]
                                                                            }) for j in range(num_test_batches)])
                                    print('The corresponding test accuracy is {0:.2%}'.format(
                                        test_accuracy))

                                #Saving best weights and biases
                                #save_path = saver.save(sess, "/tmp/best.ckpt")
                                #print("Best variables saved in specified file dir: %s" % save_path)
                                if best_validation_accuracy == 1:
                                    raise GetOutOfLoop

            except GetOutOfLoop:
                print("100% Accuracy achieved. Stopping training...\n\n")
                pass
                                
                                                            

            end_time = time.time()
            total_time = end_time - start_time


            print("Finished training network.")
            print("Time to train network: {}s".format(total_time))
            print("Number of examples trained per sec: {}".format(size(training_data)*epochs/total_time))

            print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
                best_validation_accuracy, str(best_iteration)))
            if test_data:
                print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

            if save_dir:
                save_path = saver.save(sess, save_dir)
                print("Model saved in specified file dir: %s" % save_path)
            else:
                pass

            '''
            print("List of train accuracies are:", self.train_accuracies)
            print("List of validation accuracies are:", self.validation_accuracies)
            print("List of test accuracies are:", self.test_accuracies)
            '''

    def predict(self, data):
        test_x, test_y = data
        predictions = []
        num_test_batches = int(size(data)/self.mini_batch_size)

        #Define predictions operations
        ## This operation will return predictions for a specific minibatch
        ### If output_predictions = True, predictions of each minibatch will be extended into self.test_mb_predictions = []
        mb_predictions = self.layers[-1].y_out

        #Same accuracy operation as in method train
        mb_accuracy = self.layers[-1].accuracy(self.y)
              

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            # Restore variables from disk.
            saver.restore(sess, "/tmp/model.ckpt")
            
            print("Predicting results...")
            
            predictions.extend([mb_predictions.eval(feed_dict = {self.x:
                                                                 test_x[j*self.mini_batch_size: (j+1)*self.mini_batch_size],
                                                                 self.y:
                                                                 test_y[j*self.mini_batch_size: (j+1)*self.mini_batch_size]
                                                                 }) for j in range(num_test_batches)])
                                                                
            predictions = np.concatenate(predictions).ravel().tolist()
            print("Number of predictions = ", len(predictions))
            print("Number of test samples = ", size(data))
            print("Calculating Accuracies")
            accuracy = np.mean(
                [mb_accuracy.eval(feed_dict = {self.x:
                                               test_x[j*self.mini_batch_size: (j+1)*self.mini_batch_size],
                                               self.y:
                                               test_y[j*self.mini_batch_size: (j+1)*self.mini_batch_size]
                                               }) for j in range(num_test_batches)])
            return predictions, accuracy
            


###Define Layer types
class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, 
                 activation_fn=sigmoid, padding='SAME', c_stride=[1,1,1,1], k_stride = [1,2,2,1], k_size = [1, 2, 2, 1]):
        """Filter is of shape [filter_height, filter_width, in_channels, out_channels]
            For the mnist data in the sample, it would be (5, 5, 1, 20)

        Tensorflow input shape is a 4d tensor with dimensions [batch, in_height, in_width, in_channels]
            For the mnist data, it would be (mini_batch_size, 28, 28, 1)
        
        filter_shape: 4D tensor of shape [filter_height, filter_width, in_channels, out_channels]
        image_shape: 4D tensor of shape [batch_size, image_height, image_width, num_channels]
        activation_fn: the activation function to be used in this layer instance
        padding: Type of padding to use for convolution operation. Either SAME or VALID
        c_stride: stride length of convolution sliding window. Default is 1
        k_stride: stride length of max pool window. Default is 2
        k_size: size of max pool window. Default is 2
        Notably: We do not incorporate drop-out for convolutional layer becuase the shared weight and biases
        already generelizes the layer
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = k_size[1:3]
        self.activation_fn=activation_fn
        self.padding = padding
        self.c_stride = c_stride
        self.k_stride = k_stride
        self.k_size = k_size
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(self.poolsize))
        self.w = tf.Variable(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype="float32")
        )
        self.b = tf.Variable(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[3],)),
                dtype="float32")
        )
        self.params = [self.w, self.b]
        

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        
        self.inpt = tf.reshape(inpt, self.image_shape)
        conv_out = self.activation_fn(conv2d(self.inpt, self.w, strides=self.c_stride, padding=self.padding) + self.b)
        pooled_out = max_pool_2x2(conv_out, ksize=self.k_size, strides=self.k_stride, padding=self.padding)
        self.output = pooled_out
        self.output_dropout = self.output # no dropout in the convolutional layers




class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        """
        n_in: number of input channels
        n_out: number of output channels
        activation_fn: activation function to be used in this layer instance
        p_dropout: probability that each element(network's units) is kept
        """
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = tf.Variable(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype="float32"),
            name='w')
        self.b = tf.Variable(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype="float32"),
            name='b')
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = tf.reshape(inpt, (mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*tf.matmul(self.inpt, self.w) + self.b)
        self.y_out = tf.argmax(self.output, axis=1)

        #Set dropout
        if self.p_dropout != 0.0:
            self.inpt_dropout = dropout_layer(
                tf.reshape(inpt_dropout, (mini_batch_size, self.n_in)), self.p_dropout)
        else:
            self.inpt_dropout = self.inpt

        self.output_dropout = self.activation_fn(
            tf.matmul(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        #Return the accuracy for the mini-batch.
        return tf.reduce_mean(tf.equal(y, self.y_out))



class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        """
        n_in: number of input channels
        n_out: number of output channels
        p_dropout: probability that each element(network's units) is kept
        Note:Activation function is automatically softmax
        """
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = tf.Variable(
            np.zeros((n_in, n_out), dtype="float32"),
            name='w')
        self.b = tf.Variable(
            np.zeros((n_out,), dtype="float32"),
            name='b')
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = tf.reshape(inpt, (mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*tf.matmul(self.inpt, self.w) + self.b)
        self.y_out = tf.argmax(self.output, axis=1)

        #Set dropout
        if self.p_dropout != 0.0:
            self.inpt_dropout = dropout_layer(
                tf.reshape(inpt_dropout, (mini_batch_size, self.n_in)), self.p_dropout)
        else:
            self.inpt_dropout = self.inpt

        self.output_dropout = tf.matmul(self.inpt_dropout, self.w) + self.b #Use this when using softmax cross entropy
        #self.output_dropout = softmax(tf.matmul(self.inpt_dropout, self.w) + self.b) #Use this when using log loss

    def cost(self, net):
        #Return the softmax cross entropy cost.
        #return tf.reduce_mean(tf.losses.log_loss(labels=net.y, predictions=self.output_dropout))
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output_dropout, labels=net.y))
        #return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_dropout, labels=net.y)) #For one-hot encoded data

    def accuracy(self, y):
        #Return the accuracy for the mini-batch.
        correct = tf.equal(y, tf.cast(self.y_out, tf.int32))
        return tf.reduce_mean(tf.cast(correct, tf.float32))



