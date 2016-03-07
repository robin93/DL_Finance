import pandas as pd
import numpy as np
import math
import copy
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep
import sys
import datetime as dt
import matplotlib.pyplot as plt


def find_events(ls_symbols, d_data):
	''' Finding the event dataframe '''
	df_close = d_data['actual_close']
	ts_market = df_close['SPY']

	# Creating an empty dataframe
	df_events = copy.deepcopy(df_close)
	df_events = df_events * np.NAN

	# Time stamps for the event range
	ldt_timestamps = df_close.index

	print "Initiating the loop on symbols"
	number_of_events = 0
	orders_index = 0
	for s_sym in ls_symbols:
		for i in range(1, len(ldt_timestamps)):
			# Calculating the returns for this timestamp
			f_symprice_today = df_close[s_sym].ix[ldt_timestamps[i]]
			f_symprice_yest = df_close[s_sym].ix[ldt_timestamps[i - 1]]
			f_marketprice_today = ts_market.ix[ldt_timestamps[i]]
			f_marketprice_yest = ts_market.ix[ldt_timestamps[i - 1]]
			f_symreturn_today = (f_symprice_today / f_symprice_yest) - 1
			f_marketreturn_today = (f_marketprice_today / f_marketprice_yest) - 1

			# Event is found if the symbol is down more then 3% while the
			# market is up more then 2%
			if f_symprice_today <10 and f_symprice_yest >= 10:
				number_of_events += 1
				print number_of_events
				orders.set_value(orders_index,'sym',s_sym)
				orders.set_value(orders_index+1,'sym',s_sym)
				orders.set_value(orders_index,'Number_of_shares',100)
				orders.set_value(orders_index+1,'Number_of_shares',100)
				orders.set_value(orders_index,'Year',ldt_timestamps[i].year)
				orders.set_value(orders_index,'Month',ldt_timestamps[i].month)
				orders.set_value(orders_index,'Date',ldt_timestamps[i].day)
				orders.set_value(orders_index,'action','Buy')
				orders.set_value(orders_index+1,'action','Sell')
				if i < len(ldt_timestamps)-5:
					orders.set_value(orders_index+1,'Year',ldt_timestamps[i+5].year)
					orders.set_value(orders_index+1,'Month',ldt_timestamps[i+5].month)
					orders.set_value(orders_index+1,'Date',ldt_timestamps[i+5].day)
				else:
					orders.set_value(orders_index+1,'Year',ldt_timestamps[len(ldt_timestamps)-1].year)
					orders.set_value(orders_index+1,'Month',ldt_timestamps[len(ldt_timestamps)-1].month)
					orders.set_value(orders_index+1,'Date',ldt_timestamps[len(ldt_timestamps)-1].day)
				orders_index +=2


            


"""Main program begins"""
dt_start = dt.datetime(2008, 1, 3)
dt_end = dt.datetime(2009,12,28)
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

print "Creating empty dataframe"
orders = pd.DataFrame(columns=['Year','Month','Date','sym','action','Number_of_shares'])
orders_index = 0

print "Initiating the find events function"
df_events = find_events(ls_symbols, d_data)

print orders
# orders.to_csv('orders file output.csv',sep=",")


opening_cash = int(sys.argv[1])
order_data = orders
# value_file = sys.argv[3]

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
	if len(subset)>0:
		value = list(subset['cash_balance'])[0]
	else:
		value = opening_cash
	return value

def fetchShareNumbers(row):
	timestamp = row['timestamps']
	subset = (order_data[order_data['DateTime']<=timestamp]).tail(n=1)
	if len(subset):
		value = list(subset[symbol])[0]
	else:
		value = 0
	return value

"""Main Program Begins"""

"""Extract and read the orders data"""
# order_data = pd.read_csv(os.path.join(cwd,order_data),header = None,sep = ",")
# order_data = order_data[[0,1,2,3,4,5]]
# order_data.columns = ['Year','Month','Date','sym','action','Number_of_shares']
list_of_symbols = order_data['sym'].unique()
order_data['DateTime'] = order_data.apply(datetimefun,axis=1)
order_data = order_data.sort(['DateTime'])
order_data = order_data.reset_index(drop=True)
# print order_data

"""Extract and format the prices data"""
dt_timeofday = dt.timedelta(hours=16)
dt_start = dt.datetime(2008, 1, 3)
dt_end = dt.datetime(2009,12,28)
# dt_start = min(order_data['DateTime'])
# print "Start date",dt_start
# dt_end = max(order_data['DateTime'])
# print "Last date",dt_end
ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
# print ldt_timestamps
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
order_data.to_csv('orders data file output.csv',sep=',')



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

# values_data.to_csv('values data file output.csv',sep =",")

##Evaluate the performance metrics
daily_values = pd.Series(values_data['Combined_worth'])
print daily_values
daily_returns = list()
for i in range(0,len(daily_values)-1):
	x = daily_values[i]
	y = daily_values[i+1]
	print "X is",x, "Y is",y,"return_value is",((y-x)/abs(x))
	if abs(x) > 0:
		daily_returns.append((y-x)/abs(x))
		
print "Number of elements in daily returns are", len(daily_returns)
print daily_returns
# daily_returns = daily_values.pct_change(1)
std_deviation = np.std(daily_returns)
sharpe_ratio = np.sqrt(252)*np.mean(daily_returns)/std_deviation
print "Average Daily Returns",np.mean(daily_returns)
print "standard Deviation",std_deviation
print 'sharpe ratio',sharpe_ratio
