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

def iterate_minibatches(inputs,targets,batch_size,shuffle=True):
    assert len(inputs)==len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0,len(inputs)- batch_size+1,batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx+batch_size]
        else:
            excerpt = slice(start_idx,start_idx+batch_size)
        yield inputs[excerpt],targets[excerpt]

from numpy import genfromtxt
cols = [5,9,13,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
data = genfromtxt('SBI_lag_TI_rise_fall.csv', delimiter=',',usecols=cols,skip_header=4000)
seq_len = 10
number_of_batches = data.shape[0]-seq_len - 200
X,Y = np.empty((number_of_batches,seq_len,data.shape[1]-1)),np.empty((number_of_batches,))
for i in range(seq_len,data.shape[0]-200):
    x = data[(i-seq_len):i,0:25]
    X[i-seq_len] = x
    Y[i-seq_len] = data[i-1,25]

number_of_batches_val = 200
X_val,Y_val = np.empty((number_of_batches_val,seq_len,data.shape[1]-1)),np.empty((number_of_batches_val,))
for i in range(data.shape[0]-200,data.shape[0]):
    x = data[(i-seq_len):i,0:25]
    X_val[i-seq_len-data.shape[0]+200] = x
    Y_val[i-seq_len-data.shape[0]+200] = data[i-1,25]



N_BATCH = X.shape[0]
MAX_LENGTH = seq_len
N_FEATURES = X.shape[2]
N_HIDDEN = 50
LEARNING_RATE = 0.001
EPOCH_SIZE = 10
NUM_EPOCHS = 30

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
print "network output",network_output
predicted_values = network_output.flatten()
print "predicted_values",predicted_values
cost = T.mean((predicted_values - target_values)**2)
# cost = lasagne.objectives.binary_crossentropy(predicted_values,target_values)
all_params = lasagne.layers.get_all_params(l_out)
print("Computing updates ...")
updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
train = theano.function([l_in.input_var, target_values],cost, updates=updates)
compute_cost = theano.function([l_in.input_var, target_values], cost)
print "Training...."
try:
    count = 0
    count_list,train_err_list = list(),list()
    for epoch in range(NUM_EPOCHS):
        train_err = 0
        for batch in iterate_minibatches(X,Y,20,shuffle=True):
            inputs,targets = batch
            batch_error = train(inputs,targets)
            train_err += batch_error
        count += 1
        count_list.append(count)
        train_err_list.append(train_err)
        print train_err

    f_test = theano.function([l_in.input_var],network_output.flatten())

    print "percentiles", [np.percentile(list(f_test(X_val)),10*i) for i in range(10)]

    threshold = 0.05
    prediction_list = [1 if i > threshold else 0 for i in list(f_test(X_val))]

    y_test_list = list()
    for item in list(Y_val):
        y_test_list.append(int(item))
    print "y_test_list",y_test_list
    # Confusion Matrix
    confusion_matrix_array = [0,0,0,0]
    #print "unique count target values",np.bincount(y_test_list)
    for index in range(len(y_test_list)):
        if y_test_list[index]==0:
            if prediction_list[index]==0:
                confusion_matrix_array[0] += 1
            else:
                confusion_matrix_array[2] += 1
        else:
            if prediction_list[index] == 1:
                confusion_matrix_array[1] += 1
            else:
                confusion_matrix_array[3] += 1

    from terminaltables import AsciiTable
    table_data = [
        ["Target Values","Prediction = 1","Prediction = 0"],
        ["Target Values = 1",str(confusion_matrix_array[1]),str(confusion_matrix_array[3])],
        ["Target Values = 0",str(confusion_matrix_array[2]),str(confusion_matrix_array[0])]
        ]
    table = AsciiTable(table_data)
    print (table.table)

except KeyboardInterrupt:
    pass
