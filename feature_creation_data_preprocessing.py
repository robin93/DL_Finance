import pandas as pd
import csv as csv
import os
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()

print "Reading the raw data file"
raw_data = pd.read_csv(os.path.join(cwd,'SBI_ITC_AXIS_INFO_LT_ONGC_RIL_SIEM_LPC_TATA_SJET_INDIGO_TV18.csv'),header=0,sep=",")
raw_data.drop(['OFFICIAL_CLOSE','EXCH_CALCULATED_VOLATILITY','CHG_NET_1D'],inplace=True,axis=1)
print "Removing unclean data rows"
data = raw_data[(raw_data.HIGH > raw_data.LOW)]
data = data[(data.LAST>0)&(data.HIGH>0)&(data.LOW>0)]
print "Filling Missing Values"
data = data.fillna(method="backfill")
data = data.fillna(method="ffill")


#data.insert(2,'Day',pd.DatetimeIndex(data['Date']).day)
#data.insert(2,'Month',pd.DatetimeIndex(data['Date']).month)
#data.insert(2,'Year',pd.DatetimeIndex(data['Date']).year)
data = data.set_index('Date')
print "Data Tail",data.tail(20)

#pct_change_list = data['CHG_PCT_1D'].tolist()
#print pct_change_list
#smaller_list = [ i for i in pct_change_list if (i> -6 and i <10)]
#print "mean of pct change",np.mean(smaller_list)
#print "standard deviation",np.std(smaller_list)
#histo, bin_edges = np.histogram(smaller_list,density=True)
#print 'histogram values', histo
#print 'bin edges value',bin_edges

mean,std,factor = 0.206,2.47,1.5
up_bound,lower_bound = mean + factor*std,mean - factor*std
data_subset = data[data['Equity']=='SBI']
data_subset['significant_change'] = data_subset['CHG_PCT_1D'].apply(lambda row : 1 if (row>up_bound or row<lower_bound) else 0)
data_subset.insert(2,'LAST_lag1',data_subset['LAST'].shift(+1))
data_subset.insert(3,'LAST_lag2',data_subset['LAST'].shift(+2))
data_subset.insert(4,'LAST_lag3',data_subset['LAST'].shift(+3))
data_subset.insert(6,'VOLUME_lag1',data_subset['VOLUME'].shift(+1))
data_subset.insert(7,'VOLUME_lag2',data_subset['VOLUME'].shift(+2))
data_subset.insert(8,'VOLUME_lag3',data_subset['VOLUME'].shift(+3))
data_subset.insert(10,'HIGH_lag1',data_subset['HIGH'].shift(+1))
data_subset.insert(11,'HIGH_lag2',data_subset['HIGH'].shift(+2))
data_subset.insert(12,'HIGH_lag3',data_subset['HIGH'].shift(+3))
data_subset.insert(14,'LOW_lag1',data_subset['LOW'].shift(+1))
data_subset.insert(15,'LOW_lag2',data_subset['LOW'].shift(+2))
data_subset.insert(16,'LOW_lag3',data_subset['LOW'].shift(+3))
data_subset = data_subset.fillna(method="backfill")
data_subset = data_subset.fillna(method="ffill")
cols_to_norm = ['LAST','LAST_lag1','LAST_lag2','LAST_lag3','VOLUME','VOLUME_lag1','VOLUME_lag2','VOLUME_lag3','HIGH', 'HIGH_lag1', 'HIGH_lag2', 'HIGH_lag3', 'LOW', 'LOW_lag1', 'LOW_lag2', 'LOW_lag3']
data_subset[cols_to_norm] = data_subset[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
print data_subset.head(10)
print data_subset.tail(10)

data_subset.to_csv('SBI_processed_data.csv',sep = ',')
