import pandas as pd
import numpy as np
import math
import copy
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
# import QSTK.qstkstudy.EventProfiler as ep
import sys

def find_events(ls_symbols, d_data):
	''' Finding the event dataframe '''
	# df_close = d_data['close']
	# ts_market = df_close['SPY']

	# Time stamps for the event range
	ldt_timestamps = d_data.index

	print "Initiating the loop on symbols"
	number_of_events = 0
	for s_sym in ls_symbols:
		for i in range(1, len(ldt_timestamps)-20):
			# Calculating the returns for this timestamp
			equity_boling_today = d_data[''.join([s_sym,"_value"])].ix[ldt_timestamps[i]]
			equity_boling_yest = d_data[''.join([s_sym,"_value"])].ix[ldt_timestamps[i - 1]]
			market_boling_today = d_data[''.join(["SPY","_value"])].ix[ldt_timestamps[i]]
			# print equity_boling_today
			# f_symprice_today = df_close[s_sym].ix[ldt_timestamps[i]]
			# f_symprice_yest = df_close[s_sym].ix[ldt_timestamps[i - 1]]
			# f_marketprice_today = ts_market.ix[ldt_timestamps[i]]
			# f_marketprice_yest = ts_market.ix[ldt_timestamps[i - 1]]
			# f_symreturn_today = (f_symprice_today / f_symprice_yest) - 1
			# f_marketreturn_today = (f_marketprice_today / f_marketprice_yest) - 1

			# Event is found if the symbol is down more then 3% while the
			# market is up more then 2%
			# if f_symprice_today <10 and f_symprice_yest >= 10:
			# 	number_of_events += 1
			# 	print number_of_events
			if (equity_boling_today < -2.0 and equity_boling_yest >= -2.0 and market_boling_today >= 1.1):
				number_of_events += 1
				print number_of_events

Answer = 232

            


"""Main program begins"""
dt_start = dt.datetime(2008, 1, 1)
dt_end = dt.datetime(2009,12,31)
print "Fetching Time stamps"
ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))

print "Connecting to the dataset"
dataobj = da.DataAccess('Yahoo')
ls_symbols = dataobj.get_symbols_from_list('sp5002012')
ls_symbols.append('SPY')

ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
print "creating the data object"
ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)

print "Zipping the data to dictionary"
d_data = dict(zip(ls_keys, ldf_data))

print "Imputing the dataset"
for s_key in ls_keys:
    d_data[s_key] = d_data[s_key].fillna(method='ffill')
    d_data[s_key] = d_data[s_key].fillna(method='bfill')
    d_data[s_key] = d_data[s_key].fillna(1.0)

def getValue(row):
	if pd.isnull(row[''.join([symbol,"_rolling_std"])]):
		mid_value = -3
	else:
		mid_value = (row[symbol] - row[''.join([symbol,"_rolling_mean"])])/row[''.join([symbol,"_rolling_std"])]
	return mid_value
d_data = d_data['close']
for symbol in ls_symbols:
	d_data[''.join([symbol,"_rolling_mean"])] = pd.rolling_mean(d_data[symbol],20)
	d_data[''.join([symbol,"_rolling_std"])] = pd.rolling_std(d_data[symbol],20)
	d_data[''.join([symbol,"_value"])] = d_data.apply(getValue,axis=1)

# print d_data.tail(10)
print "Initiating the find events function"
df_events = find_events(ls_symbols, d_data)
