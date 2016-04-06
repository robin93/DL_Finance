# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 22:32:48 2016

@author: Keshav Sehgal
"""
# Check Current Working Directory
import os
os.getcwd()

# Set Current Working Directory
os.chdir('D:\\PGDBA\\KGP\\fin tech\\MLP\\3 April')

import pandas as pd
import csv as csv
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

#################### Writing the technical indicators#################

print(data_subset.columns.values)
print(data_subset.head(10))

# Simple 5 day Moving Average
def ma(price,n):
    p = pd.rolling_mean(price, window = n)
    return p.fillna(0)
    
data_subset['MA(5)'] = ma(data_subset['LAST'],5)

# Simple 20 day Moving Average
data_subset['MA(20)'] = ma(data_subset['LAST'],20)

# Expoenetially Weighted Moving Average (alpha = 4, span = 9 )
def ewma(price, n):
    return pd.ewma(price, span = n)

data_subset['EWMA(20)'] = ewma(data_subset['LAST'], 9)

######## MACD, Signal and Histogram #############
#  Moving Average Convergence/Divergence (MACD)
def MACD(price, n1, n2):
    return pd.ewma(price, span = n1) - pd.ewma(price, span = n2)

data_subset['MACD'] = MACD(data_subset['LAST'], 12 , 26)

# Signal
def signal(price, n1, n2, span):
    p = MACD(price,n1,n2)
    return ewma(p,span)

data_subset['SIGNAL'] = signal(data_subset['LAST'],12,26,9)

# Histogram
def histogram(price, n1, n2, span):
    return MACD(price, 12 , 26) - signal(price, 12, 26, 9)

data_subset['Histogram'] = histogram (data_subset['LAST'],12,26,9)
####################

# RSI (Relative Strength Index)
def rsi(price, n):
    gain = price.diff()
    gain[np.isnan(gain)] = 0
    up, down = gain.copy(), gain.copy()
    up[up<0] = 0
    down[down>0] = 0
    upEWMA = ewma(up,n)
    downEWMA = ewma(down.abs(),n)
    rs = upEWMA/downEWMA
    return 100 - (100/(1+rs))

data_subset['RSI(14)'] = rsi(data_subset['LAST'], 14)

# Bollinger Band
data_subset['BolBandUpper'] = pd.rolling_mean(data_subset['LAST'], window = 20) + (2 * pd.rolling_std(data_subset['LAST'], window = 20))
data_subset['BolBandLower'] = pd.rolling_mean(data_subset['LAST'], window = 20) - (2 * pd.rolling_std(data_subset['LAST'], window = 20))

# Stochastics (%K)
def STOK(price,high,low,n,d):
    highp = pd.rolling_max(high,n)
    lowp = pd.rolling_min(low,n)
    stok = 100*(price - lowp)/(highp-lowp)
    return stok,ma(stok,d), ma(ma(stok,d),d)

data_subset['FastStochastic(k%)'],data_subset['FastStochastic(d%)'] , data_subset['SlowStochastic(d%)']= STOK(data_subset['LAST'],data_subset['HIGH'],data_subset['LOW'],14,3)

# Momentum and Rate of Cahnge(7 day)
def Momentum(price,n):
    p = price.diff(n)
    return p.fillna(0)

def RateOfChange(price,n):
    p = price.diff(n)/price.shift(n)
    return p.fillna(0)
    
data_subset['Momentum(7)'] = Momentum(data_subset['LAST'],7)
data_subset['RateOfChange(7)'] = RateOfChange(data_subset['LAST'],7)

# Moving variance
def MovingVariance(price,n):
    p = pd.rolling_var(price,n)
    return p.fillna(0)

data_subset['MovingVariance'] = MovingVariance(data_subset['LAST'],7)

#Commodity Channel Index (CCI)iance
def CCI(price, high, low, n, d):
    tp = (price + high + low)/3
    MeanDev = MeanDeviation(tp,d)
    CCI = (tp - ma(tp,n))/(.015 * MeanDev)
    return CCI

def MeanDeviation(price, d):
    mad = pd.rolling_apply(price,d,lambda x: np.abs(x - x.mean()).mean())
    return mad.fillna(0)

data_subset['CCI'] = CCI(data_subset['LAST'],data_subset['HIGH'],data_subset['LOW'],20,20)

# Chaikin Oscillator
def Chaikin(price, high, low, volume):
    MoneyFlowMultiplier = ((price - low) - (high - low))/ (high - low)
    MoneyFlowVolume = MoneyFlowMultiplier * volume
    ADL = price.copy()
    ADL[0] = 0
    for i in range(1,len(price)):
        ADL[i] = ADL[i-1] + MoneyFlowVolume[i] 
    Chaikin = ewma(ADL,3) - ewma(ADL,10)
    return Chaikin

data_subset['Chaikin'] = Chaikin(data_subset['LAST'],data_subset['HIGH'],data_subset['LOW'],data_subset['VOLUME'])

# Disparity Index (10)
def DisparityIndex(price,n) :
    DI = 100*(price - ma(price,n))/ma(price,n)
    return DI

data_subset['DisparityIndex(10)'] = DisparityIndex(data_subset['LAST'],10)

# Williams %R
def WilliamR(price,high,low,n):
    highp = pd.rolling_max(high,n)
    lowp = pd.rolling_min(low,n)
    WR = (highp - price) / (highp - lowp)
    return WR
data_subset['WilliamR(10)'] = WilliamR(data_subset['LAST'],data_subset['HIGH'],data_subset['LOW'],10)

# Volatility
def volatility(price,n):
    volatility = pd.rolling_std(price,window = n)/ma(price,n)
    return volatility
    
data_subset['Volatility(20)'] = volatility(data_subset['LAST'],20)
data_subset['Volatility(10)'] = volatility(data_subset['LAST'],10)


data_subset.to_csv('SBI_processed_data.csv',sep = ',')