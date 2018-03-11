import func
from network import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer # softmax plus log-likelihood cost is more common in modern image classification networks.
import tensorflow as tf
from func import ReLU
'''To use a different activation function, we need to import it first.
Functions in func: sigmoid, ReLU, leaky_relu, tanh, softmax(used for softmax output layer), linear(Not advisable)
We can use other activation functions from tensorflow libraries ie. tf.nn.<activation name>

'''

# read data:
training_data, validation_data, test_data = func.load_mnist_data_shared()  #"../data/mnist_expanded.pkl.gz" if dataset is expanded

# mini-batch size:
mini_batch_size = 10

#Create Model object
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 28, 28, 1),
                  filter_shape=(5, 5, 1, 20),
                  poolsize=(2, 2),
                  activation_fn=ReLU, padding='SAME'),
    ConvPoolLayer(image_shape=(mini_batch_size, 14, 14, 20),
                  filter_shape=(5, 5, 20, 40),
                  poolsize=(2, 2),
                  activation_fn=ReLU, padding='SAME'),
    FullyConnectedLayer(
        n_in=40*7*7, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
    FullyConnectedLayer(
        n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.0),
    SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)], mini_batch_size)

#Train the model
net.train(training_data, 40, mini_batch_size, 0.0003, validation_data, test_data, store_accuracies=False, save_dir = "/tmp/model.ckpt", calibration=False, lmbda=0.0)

'''
#Use the model to predict and get accuracy of predictions
predictions, accuracy = net.predict(test_data)
print("Predictions are:", predictions)
print("Accuracy is: ", accuracy)

'''
