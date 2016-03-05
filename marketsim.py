import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

import pandas as pd
import numpy as np
import csv as csv
import os
import sys
import datetime as dt
import matplotlib.pyplot as plt
cwd = os.getcwd()

opening_cash = int(sys.argv[1])
order_data = sys.argv[2]
value_file = sys.argv[3]

def datetimefun(row):
	date = dt.datetime(row['Year'],row['Month'],row['Date'],16,0,0)
	return date

def extract_actual_price(row):
	index = dt.datetime(row['Year'],row['Month'],row['Date'],16,0,0)
	act_price = (ldf_data.loc[[index]])[row['sym']][0]
	return act_price

def delta_cash(row):
	if (row['action'] == 'Buy'):
		return (-1*row['Number_of_shares']*row['adjusted_price'])
	else:
		return (row['Number_of_shares']*row['adjusted_price'])

def CompanySharesDelta(row):
	if (row['sym'] != symbol):
		return 0
	else:
		if (row['action'] == 'Buy'):
			return row['Number_of_shares']
		elif (row['action'] == 'Sell'):
			return (-1*row['Number_of_shares'])

def HeldSharesValue(row):
	total_value = 0
	index = row['timestamps']
	for symbol in list_of_symbols:
		total_value = total_value + (row[symbol])*ldf_data.loc[[index]][symbol][0]
	return total_value

def getOrderInfo(row):
	if any(order_data.DateTime == row['timestamps']):
		return order_data[(order_data.DateTime == row['timestamps'])]['action']

def getCashAmount(row):
	timestamp = row['timestamps']
	subset = (order_data[order_data['DateTime']<=timestamp]).tail(n=1)
	value = list(subset['cash_balance'])[0]
	return value

def fetchShareNumbers(row):
	timestamp = row['timestamps']
	subset = (order_data[order_data['DateTime']<=timestamp]).tail(n=1)
	value = list(subset[symbol])[0]
	return value

"""Main Program Begins"""

"""Extract and read the orders data"""
order_data = pd.read_csv(os.path.join(cwd,order_data),header = None,sep = ",")
order_data = order_data[[0,1,2,3,4,5]]
order_data.columns = ['Year','Month','Date','sym','action','Number_of_shares']
list_of_symbols = order_data['sym'].unique()
order_data['DateTime'] = order_data.apply(datetimefun,axis=1)
order_data = order_data.sort(['DateTime'])


"""Extract and format the prices data"""
dt_timeofday = dt.timedelta(hours=16)
dt_start = min(order_data['DateTime'])
dt_end = max(order_data['DateTime'])
ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
c_dataobj = da.DataAccess('Yahoo')
ls_keys = ['close']
ldf_data = (c_dataobj.get_data(ldt_timestamps,list_of_symbols, ls_keys))[0]

"""Doing Calculations on the Order Data"""
order_data['adjusted_price'] = order_data.apply(extract_actual_price,axis=1)
order_data['delta_cash'] = order_data.apply(delta_cash,axis=1)
order_data.loc[0, 'delta_cash'] = order_data.loc[0, 'delta_cash'] + opening_cash #initiate the cash balance
order_data['cash_balance']  = order_data.delta_cash.cumsum()
order_data.loc[0, 'delta_cash'] = order_data.loc[0, 'delta_cash'] - opening_cash
for symbol in list_of_symbols:
	order_data[symbol] = order_data.apply(CompanySharesDelta,axis=1)
	order_data[symbol] = order_data[symbol].cumsum()
print order_data



"""Extract the trading dates in the time range"""
sym_data_ranges = pd.DataFrame(columns= ['sym','Min_Date','Max_Date'])
df_index = 0
for symbol in list_of_symbols:
	sym_data_ranges.set_value(df_index,'sym',symbol)
	subset = order_data[order_data['sym']==symbol]
	sym_data_ranges.set_value(df_index,'Min_Date',min(subset['DateTime']))
	sym_data_ranges.set_value(df_index,'Max_Date',max(subset['DateTime']))
	df_index += 1
# print sym_data_ranges



"""Create and assign values in the values file"""
values_data = pd.DataFrame(ldt_timestamps)
values_data.columns = ['timestamps']
values_data['cash'] = values_data.apply(getCashAmount,axis=1)
for symbol in list_of_symbols:
 	values_data[symbol] = values_data.apply(fetchShareNumbers,axis=1)
values_data['HeldSharesValue'] = values_data.apply(HeldSharesValue,axis=1)
values_data['Combined_worth'] = values_data['cash'] + values_data['HeldSharesValue']
print values_data



##Evaluate the performance metrics
daily_values = pd.Series(values_data['Combined_worth'])
daily_returns = daily_values.pct_change(1)
std_deviation = np.std(daily_returns)
sharpe_ratio = np.sqrt(250)*np.mean(daily_returns)/std_deviation
print "Average Daily Returns",np.mean(daily_returns)
print "standard Deviation",std_deviation
print 'sharpe ratio',sharpe_ratio

#plot the portfolio value with time during the year
values_data.plot(x='timestamps',y='Combined_worth',kind = 'line')
plt.xlabel('Time of Year');plt.ylabel('Portfolio value');plt.title('Plot of Portfolio Value with Time'),plt.grid(True)
plt.show()
