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

def cost_sensitive_weights(target_values,a,b):
    weight_matrix = np.empty((target_values.shape[0],target_values.shape[1]),dtype=float)
    count = 0
    for i in target_values:
        if (i[0] == 1):
            weight_array[count][0],weight_array[count][1] = a,b
        elif (i[1]==1):
            weight_array[count][1],weight_array[count][0] = a,b
        else:
            weight_array[count][0],weight_array[count][1] = b,b
        count += 1
    return weight_matrix

"""import data and convert to numpy array, segment into train and test"""
from numpy import genfromtxt
cols = [i for i in range(1,41)]
raw_data = genfromtxt('SBI_lag_TI_rise_fall.csv', delimiter=',',usecols=cols,skip_header=20)
data = raw_data

print raw_data.shape
X_test, y_test = data[5385:,1:38],data[5385:,38:]
raw_data = data[3385:5385,:]
# X_train, y_train,X_test, y_test = data[3385:5385,1:38],data[3385:5385,38:],data[5385:,1:38],data[5385:,38:]

"""oversampling the train dataset to treat class imbalance"""
unq_rise,unq_idx_rise = np.unique(raw_data[:,-2],return_inverse=True)
unq_dip,unq_idx_dip = np.unique(raw_data[:,-1],return_inverse=True)
unq_cnt_rise,unq_cnt_dip = np.bincount(unq_idx_rise),np.bincount(unq_idx_dip)
cnt_rise_dip = np.max(unq_cnt_rise)-np.min(unq_cnt_dip)
data_rise_dip = np.empty((cnt_rise_dip*3,) + raw_data.shape[1:],raw_data.dtype)
intersectArray = np.intersect1d(np.where(unq_idx_dip==0)[0],np.where(unq_idx_rise==0)[0],assume_unique = True)
rise_indices = np.where(unq_idx_rise==1)[0]
dip_indices = np.where(unq_idx_dip==1)[0]
sampled_indices_0 = np.random.choice(intersectArray,cnt_rise_dip)
sampled_indices_rise = np.random.choice(rise_indices,cnt_rise_dip)
sampled_indices_dip = np.random.choice(dip_indices,cnt_rise_dip)
data_rise_dip[0:cnt_rise_dip] = raw_data[sampled_indices_0]
data_rise_dip[cnt_rise_dip:cnt_rise_dip*2] = raw_data[sampled_indices_rise]
data_rise_dip[cnt_rise_dip*2:cnt_rise_dip*3] = raw_data[sampled_indices_dip]
data = data_rise_dip[np.argsort(data_rise_dip[:,0])]

X_train, y_train = data[:,1:38],data[:,38:]



"""building the network"""
def build_mlp(input_var):
    l_in = lasagne.layers.InputLayer((None,37),name='INPUT')
    l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=50,name = 'Hidden1')
    # l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=30,nonlinearity=lasagne.nonlinearities.linear, name = 'Hidden1')
    l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=50, name = 'Hidden2')
    l_out = lasagne.layers.DenseLayer(l_hid2, num_units=2,nonlinearity=lasagne.nonlinearities.sigmoid, name = 'OUTPUT')
    # l_out = lasagne.layers.DenseLayer(l_hid1, num_units=1,nonlinearity=lasagne.nonlinearities.leaky_rectify, name = 'OUTPUT')
    # l_out = lasagne.layers.DenseLayer(l_hid1, num_units=1,nonlinearity=lasagne.nonlinearities.leaky_rectify, name = 'OUTPUT')
    return l_out


input_var,target_var = T.matrix('inputs'),T.matrix('targets')
num_epochs=50
network = build_mlp(input_var)

"""loading weight values from the previous model"""
"""load layer parameters. To be used only when learning walk forward"""
# with np.load('model.npz') as f:
#     param_values = [f['arr_%d'%i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(network,param_values)

prediction = lasagne.layers.get_output(network,input_var, deterministic = True)
loss = lasagne.objectives.binary_crossentropy(prediction, target_var)

"""loss function aggregation only to be used when using cost sensitive training"""
# class_weights = np.empty((50,2),dtype=float)
# class_weights_global = class_weights
# loss = aggregate(loss, weights=theano.shared(class_weights_global),mode='normalized_sum')

"""loss function aggregation to be used without cost sensitive training"""
loss = aggregate(loss,mode='mean')


params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.sgd(loss, params, learning_rate=0.2)
test_prediction = lasagne.layers.get_output(network,input_var, deterministic=True)
test_loss = lasagne.objectives.binary_crossentropy(test_prediction,target_var)
test_loss = test_loss.mean()
# test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)
# test_acc = test_acc.mean()
train_fn = theano.function([input_var, target_var],loss, updates=updates)
# val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
val_fn = theano.function([input_var, target_var],test_loss)

"""Training"""
print("Starting training...")
count = 0
count_list,train_err_list = list(),list()
for epoch in range(num_epochs):
    train_err = 0
    for batch in iterate_minibatches(X_train,y_train,50,shuffle=True):
        inputs,targets = batch
        # class_weights = cost_sensitive_weights(targets,10,1) ##to be used when using cost sensitive training
        batch_error = train_fn(inputs,targets)
        train_err += batch_error
    count += 1
    count_list.append(count)
    train_err_list.append(train_err)
    # print "batch count",count
    print train_err
    # print "epoch number", epoch,"training error",train_err

# print "Last layer weights:"
# print "lasagne.layers.get_all_param_values(network)[-1]",lasagne.layers.get_all_param_values(network)[-1]
# print "shape",(lasagne.layers.get_all_param_values(network)[-1]).shape
# print "lasagne.layers.get_all_param_values(network)[-2]"#,lasagne.layers.get_all_param_values(network)[-2]
# print "shape",(lasagne.layers.get_all_param_values(network)[-2]).shape
# print "lasagne.layers.get_all_param_values(network)[-3]"#,lasagne.layers.get_all_param_values(network)[-3]
# print "shape",(lasagne.layers.get_all_param_values(network)[-3]).shape
# print "lasagne.layers.get_all_param_values(network)[-4]"#,lasagne.layers.get_all_param_values(network)[-4]
# print "shape",(lasagne.layers.get_all_param_values(network)[-4]).shape
# print "lasagne.layers.get_all_param_values(network)[-5]"#,lasagne.layers.get_all_param_values(network)[-3]
# print "shape",(lasagne.layers.get_all_param_values(network)[-5]).shape
# print "lasagne.layers.get_all_param_values(network)[-6]"#,lasagne.layers.get_all_param_values(network)[-4]
# print "shape",(lasagne.layers.get_all_param_values(network)[-6]).shape

# np.savez('model.npz',*lasagne.layers.get_all_param_values(network))

f_test = theano.function([input_var],test_prediction)
print list(f_test(X_test))

threshold = 0.2
prediction_list_rise = [1 if i[0] > threshold else 0 for i in list(f_test(X_test))]
prediction_list_dip = [1 if i[1] > threshold else 0 for i in list(f_test(X_test))]
y_test_list_rise = [int(i[0]) for i in list(y_test)]
y_test_list_dip = [int(i[1]) for i in list(y_test)]


print "percentiles", [np.percentile(prediction_list_rise,10*i) for i in range(10)]

from sklearn.metrics import confusion_matrix
from terminaltables import AsciiTable
confusion_matrix_rise = confusion_matrix(y_test_list_rise,prediction_list_rise)
confusion_matrix_dip = confusion_matrix(y_test_list_dip,prediction_list_dip)

table_data_rise = [["Target Values","Prediction = 1","Prediction = 0"],
    ["Target Values = 1",str(confusion_matrix_rise[1,1]),str(confusion_matrix_rise[1,0])],
    ["Target Values = 0",str(confusion_matrix_rise[0,1]),str(confusion_matrix_rise[0,0])]]

table_data_dip = [["Target Values","Prediction = 1","Prediction = 0"],
    ["Target Values = 1",str(confusion_matrix_dip[1,1]),str(confusion_matrix_dip[1,0])],
    ["Target Values = 0",str(confusion_matrix_dip[0,1]),str(confusion_matrix_dip[0,0])]]

table_rise = AsciiTable(table_data_rise)
table_dip = AsciiTable(table_data_dip)
print "confusion matrix for rise prediction"
print table_rise.table
print "confusion matrix for dip prediction"
print table_dip.table

plt.plot([i for i in range(0,num_epochs)],train_err_list)
plt.show()