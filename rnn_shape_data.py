import numpy as np
from numpy import genfromtxt
data = genfromtxt('SBI_lag_TI_rise_fall.csv', delimiter=',',usecols=[5,9,13,17,18,19,39],skip_header=4000)
number_of_batches = data.shape[0]-10
X_val,Y_val = np.empty((number_of_batches,10,data.shape[1]-1)),np.empty((number_of_batches,))
for i in range(10,data.shape[0]):
	x = data[(i-10):i,0:6]
	X_val[i-10] = x
	Y_val[i-10] = data[i-1,6]

print X_val.shape[0]








