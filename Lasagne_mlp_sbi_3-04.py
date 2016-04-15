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
cols = [i for i in range(1,40)]
raw_data = genfromtxt('SBI_lag_TI_rise_fall.csv', delimiter=',',usecols=cols,skip_header=20)

# print raw_data

#oversampling the data for class balancing
unq,unq_idx = np.unique(raw_data[:,-1],return_inverse=True)
unq_cnt = np.bincount(unq_idx)
cnt = np.max(unq_cnt)
data = np.empty((cnt*len(unq),) + raw_data.shape[1:],raw_data.dtype)
for j in xrange(len(unq)):
    np.random.seed(20*j + 10)
    indices = np.random.choice(np.where(unq_idx==j)[0],cnt)
    data[j*cnt:(j+1)*cnt]=raw_data[indices]
data = data[np.argsort(data[:,0])]


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

X_train, y_train,X_test, y_test = data[6000:9600,1:38],data[6000:9600,38:],data[9600:,1:38],data[9600:,38:]

# print X_train
# print y_train

def build_mlp(input_var):
    l_in = lasagne.layers.InputLayer((1,37),name='INPUT')
    l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=200,name = 'Hidden1')
    # l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=30,nonlinearity=lasagne.nonlinearities.linear, name = 'Hidden1')
    l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=200, name = 'Hidden2')
    l_out = lasagne.layers.DenseLayer(l_hid1, num_units=1,nonlinearity=lasagne.nonlinearities.sigmoid, name = 'OUTPUT')
    # l_out = lasagne.layers.DenseLayer(l_hid1, num_units=1,nonlinearity=lasagne.nonlinearities.leaky_rectify, name = 'OUTPUT')
    # l_out = lasagne.layers.DenseLayer(l_hid1, num_units=1,nonlinearity=lasagne.nonlinearities.leaky_rectify, name = 'OUTPUT')
    return l_out

input_var = T.matrix('inputs')
target_var = T.matrix('targets')

num_epochs=300

network = build_mlp(input_var)

prediction = lasagne.layers.get_output(network,input_var, deterministic = True)
loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
# loss = aggregate(loss, weights=3.2*target_var,mode='mean')
loss = aggregate(loss,mode='mean')

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.sgd(loss, params, learning_rate=0.05)

test_prediction = lasagne.layers.get_output(network,input_var, deterministic=True)
test_loss = lasagne.objectives.binary_crossentropy(test_prediction,target_var)
test_loss = test_loss.mean()
# test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)
# test_acc = test_acc.mean()
train_fn = theano.function([input_var, target_var],loss, updates=updates)
# val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
val_fn = theano.function([input_var, target_var],test_loss)

train_err = 0
print("Starting training...")
count = 0
count_list,train_err_list = list(),list()
for epoch in range(num_epochs):
    for batch in iterate_minibatches(X_train,y_train,50,shuffle=True):
        inputs,targets = batch
        batch_error = train_fn(inputs,targets)
        train_err += batch_error
    count += 1
    count_list.append(count)
    train_err_list.append(batch_error)
    # print "batch count",count
    print batch_error
    # print "epoch number", epoch,"training error",train_err

# for i in range(n_test_batches):
#     # error,acc = val_fn(X_test[batch_size*i:batch_size*(i+1)],y_test[batch_size*i:batch_size*(i+1)])
#     error = val_fn(X_test[batch_size*i:batch_size*(i+1)],y_test[batch_size*i:batch_size*(i+1)])
#     test_err += error
#     # test_acc += acc
# # print "Test error",test_err#,test_acc

# test_err = 0
# for batch in iterate_minibatches(X_test,y_test,100,shuffle=False):
#         inputs,targets = batch
#         test_err += val_fn(inputs,targets)
#         print test_err

# print "Test predictions outputs",test_prediction
# print "parameter values",params

# print "Last layer weights:"
# print lasagne.layers.get_all_param_values(network)[-1]
# print lasagne.layers.get_all_param_values(network)[-2]
# print lasagne.layers.get_all_param_values(network)


f_test = theano.function([input_var],test_prediction)
# print list(f_test(X_test))

threshold = 0.25
prediction_list = [1 if i[0] > threshold else 0 for i in list(f_test(X_test))]
# print prediction_list

y_test_list = list()
for item in list(y_test):
    y_test_list.append(int(item[0]))
# print y_test_list

print "percentiles", [np.percentile(prediction_list,10*i) for i in range(10)]

# print "length of prediction list",len(prediction_list)
# print "length of y_test",len(y_test_list)


confusion_matrix_array = [0,0,0,0]
print "unique count target values",np.bincount(y_test_list)
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

# print "Threshold value",threshold
# print "Target value = 1 & Prediction = 1 : ", confusion_matrix_array[1]
# print "Target value = 1 & Prediction = 0 : ", confusion_matrix_array[3]
# print "Target value = 0 & Prediction = 0 : ", confusion_matrix_array[0]
# print "Target value = 0 & Prediction = 1 : ", confusion_matrix_array[2]

# print "unique count target values",np.bincount(y_test_list)

from terminaltables import AsciiTable
table_data = [
    ["Target Values","Prediction = 1","Prediction = 0"],
    ["Target Values = 1",str(confusion_matrix_array[1]),str(confusion_matrix_array[3])],
    ["Target Values = 0",str(confusion_matrix_array[2]),str(confusion_matrix_array[0])]
]

table = AsciiTable(table_data)
print table.table

plt.plot([i for i in range(0,num_epochs)],train_err_list)
plt.show()

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