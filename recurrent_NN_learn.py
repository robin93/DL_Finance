import sys,os,time,theano,lasagne
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T
from lasagne.init import Constant
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
from lasagne.nonlinearities import softmax
from lasagne.layers import InputLayer, DenseLayer, get_output
from lasagne.updates import sgd, apply_momentum
from lasagne.objectives import binary_crossentropy, aggregate

from numpy import genfromtxt
data = genfromtxt('SBI_lag_TI_rise_fall.csv', delimiter=',',usecols=[5,9,13,17,18,19,39],skip_header=4000)
number_of_batches = data.shape[0]-10 - 200
X,Y = np.empty((number_of_batches,10,data.shape[1]-1)),np.empty((number_of_batches,))
for i in range(10,data.shape[0]-200):
    x = data[(i-10):i,0:6]
    X[i-10] = x
    Y[i-10] = data[i-1,6]

number_of_batches_val = 200
X_val,Y_val = np.empty((number_of_batches_val,10,data.shape[1]-1)),np.empty((number_of_batches_val,))
for i in range(data.shape[0]-200,data.shape[0]):
    x = data[(i-10):i,0:6]
    X_val[i-10-data.shape[0]+200] = x
    Y_val[i-10-data.shape[0]+200] = data[i-1,6]


N_BATCH = X.shape[0]
MAX_LENGTH = X.shape[1]
N_FEATURES = X.shape[2]
N_HIDDEN = 50
LEARNING_RATE = 0.001
EPOCH_SIZE = 10
NUM_EPOCHS = 10

l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH,N_FEATURES))
l_forward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh)
l_backward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh,
        backwards=True)
l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
l_out = lasagne.layers.DenseLayer(l_concat, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)
target_values = T.vector('target_output')
network_output = lasagne.layers.get_output(l_out)
predicted_values = network_output.flatten()
cost = T.mean((predicted_values - target_values)**2)
all_params = lasagne.layers.get_all_params(l_out)
print("Computing updates ...")
updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
train = theano.function([l_in.input_var, target_values],cost, updates=updates)
compute_cost = theano.function([l_in.input_var, target_values], cost)
print "Training...."
try:
    for epoch in range(NUM_EPOCHS):
        train(X,Y)
        cost_val = compute_cost(X_val,Y_val)
        print epoch,cost_val
except KeyboardInterrupt:
    pass
