import sys,os,time,theano,lasagne
import numpy as np
import theano.tensor as T
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'



from lasagne.nonlinearities import softmax
from lasagne.layers import InputLayer, DenseLayer, get_output
from lasagne.updates import sgd, apply_momentum
from lasagne.objectives import binary_crossentropy, aggregate

from numpy import genfromtxt
data = genfromtxt('SBI_processed_data.csv', delimiter=',',usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],skip_header=10)

X_train, y_train,X_test, y_test = data[10:5500,:16],data[10:5500,16:],data[5500:,:16],data[5500:,16:]
print X_train.shape

batch_size = 20
n_training_batches = len(X_train)//batch_size 
n_test_batches = len(X_test)//batch_size

def build_mlp(input_var):
    l_in = lasagne.layers.InputLayer((1,16), name='INPUT')
    # l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=20, name = 'Hidden1')
    # l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=20, name = 'Hidden2')
    l_out = lasagne.layers.DenseLayer(l_in, num_units=1,nonlinearity=lasagne.nonlinearities.rectify, name = 'OUTPUT')
    return l_out

# input_var = T.dvector('inputs')
# target_var = T.dvector('targets')
input_var = T.matrix('inputs')
target_var = T.matrix('targets')

num_epochs=10

network = build_mlp(input_var)

prediction = lasagne.layers.get_output(network,input_var, deterministic = True)
loss = lasagne.objectives.squared_error(prediction, target_var)
loss = aggregate(loss, mode='mean')

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.sgd(loss, params, learning_rate=0.001)

test_prediction = lasagne.layers.get_output(network,input_var, deterministic=True)
test_loss = lasagne.objectives.squared_error(test_prediction,target_var)
test_loss = test_loss.mean()
# test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)
# test_acc = test_acc.mean()
train_fn = theano.function([input_var, target_var],loss, updates=updates)
# val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
val_fn = theano.function([input_var, target_var],test_loss)

print("Starting training...")
for epoch in range(num_epochs):
    train_err = 0
    start_time = time.time()
    # for i in range(len(X_train)):
    #     train_err = train_err + train_fn(X_train[i],y_train[i])
    #     print train_err

    for i in range(n_training_batches):
        train_err += train_fn(X_train[batch_size*i:batch_size*(i+1),],y_train[batch_size*i:batch_size*(i+1)])
        print train_err
    
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))

test_err = 0
test_acc = 0
# for i in range(len(X_test)):
#     error,acc = val_fn(X_test[i],y_test[i])
#     test_err += error
#     test_acc += acc
#     print test_err,test_acc

for i in range(n_test_batches):
    # error,acc = val_fn(X_test[batch_size*i:batch_size*(i+1)],y_test[batch_size*i:batch_size*(i+1)])
    error = val_fn(X_test[batch_size*i:batch_size*(i+1)],y_test[batch_size*i:batch_size*(i+1)])
    test_err += error
    # test_acc += acc
    print test_err,test_acc

# for batch in iterate_minibatches(X_test, y_test, 20, shuffle=False):
#     inputs, targets = batch
#     err, acc = val_fn(inputs, targets)
#     test_err += err
#     test_acc += acc
#     test_batches += 1
# print("Final results:")
# print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
# print("  test accuracy:\t\t{:.2f} %".format(
#        test_acc / test_batches * 100))