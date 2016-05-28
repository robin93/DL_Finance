import pandas as pd

print "Reading the raw data file"
raw_data = pd.read_csv(("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/S0003505.CSV"),header =0, sep=",")

print len(raw_data)

positive_samples = raw_data[raw_data['Change']==1]
print len(positive_samples)
negative_samples = raw_data[raw_data['Change']==0]
print len(negative_samples)

pos_sampled = positive_samples.sample(n=200)
neg_sampled = negative_samples.sample(n=200)
print len(pos_sampled)
print len(neg_sampled)

dates_for_rnn = list(pos_sampled['Date']) + list(neg_sampled['Date'])

print len(dates_for_rnn)

count = 0
for date in dates_for_rnn:
	row = (raw_data[raw_data['Date']==date]).index.get_values()[0]
	if row>10:
		features_data = raw_data.iloc[row-10:row,1:6]
		labels_data = raw_data.iloc[row-10:row,6:]
		features_data.to_csv(('/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/RNN_Java/myInput_'+str(count)+'.csv'),sep = ',',header=False,index=None)
		(labels_data.tail(1)).to_csv(('/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/RNN_Java/myLabels_'+str(count)+'.csv'),sep = ',',header=False,index=None)
		count += 1
