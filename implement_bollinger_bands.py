import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

import pandas as pd
import numpy as np
import csv as csv
import os
import datetime as dt
import sys
cwd = os.getcwd()

def getValue(row):
	if pd.isnull(row[''.join([symbol,"_rolling_std"])]):
		mid_value = 0
	else:
		mid_value = (row[symbol] - row[''.join([symbol,"_rolling_mean"])])/row[''.join([symbol,"_rolling_std"])]
	return mid_value


list_of_symbols = ["AAPL","GOOG","IBM","MSFT"]

"""Extract and format the prices data"""
dt_timeofday = dt.timedelta(hours=16)
dt_start = dt.datetime(2010, 1, 1)
dt_end = dt.datetime(2010, 12, 31)
ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
c_dataobj = da.DataAccess('Yahoo')
ls_keys = ['close']
ldf_data = (c_dataobj.get_data(ldt_timestamps,list_of_symbols, ls_keys))[0]
for symbol in list_of_symbols:
	ldf_data[''.join([symbol,"_rolling_mean"])] = pd.rolling_mean(ldf_data[symbol],20)
	ldf_data[''.join([symbol,"_rolling_std"])] = pd.rolling_std(ldf_data[symbol],20)
	ldf_data[''.join([symbol,"_value"])] = ldf_data.apply(getValue,axis=1)
print ldf_data
print ldf_data.tail(10)

ldf_data.to_csv('bollinger bands data.csv',sep=",")