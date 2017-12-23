import numpy as np
import matplotlib
matplotlib.use('Agg')
from time import time
import matplotlib.pyplot as plt
import pandas as pd 
import math
import keras
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Doc du lieu
colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
df = read_csv('./data/ICCS_sample_resource_617685_10minute.csv', header=None, index_col=False, names=colnames, usecols=[0,1], engine='python')
# df = read_csv('/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thangbk2209/GGClusterTraceDataAnalysis/data/Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv', header=None, index_col=False, names=colnames, usecols=[0,1], engine='python')

dataset = df.values

# normalize the dataset

length = len(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))

RAM = df['mem_usage'].values
CPU = df['cpu_rate'].values

CPU_nomal = scaler.fit_transform(CPU)
RAM_nomal = scaler.fit_transform(RAM)
# create and fit the LSTM network
sliding_widow = [2,3,4,5]
# split into train and test sets
for sliding in sliding_widow:
	print "sliding", sliding
	# Chuan bi du lieu dau vao
	data = []
	for i in range(length-sliding):
		datai=[]
		for j in range(sliding):
			datai.append(CPU_nomal[i+j])
		for j in range(sliding):
			datai.append(RAM_nomal[i+j])
		data.append(datai)
	data = np.array(data)

	train_size = 2880
	test_size = length - train_size
	batch_size_array = [8,16,32,64,128]
	trainX, trainY = data[0:train_size], RAM_nomal[sliding:train_size+sliding]
	testX = data[train_size:length-sliding]
	testY =  RAM[train_size+sliding:length]
	print 'trainX'
	print trainX
	print trainX[0]

	# reshape input to be [samples, time steps, features]

	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
	print 'trainX reshape'
	print trainX
	for batch_size in batch_size_array: 
		print "sliding= ", sliding
		print "batch_size= ", batch_size
		# model 1 layer 4 neural
		model1 = Sequential()
		model1.add(LSTM(4, activation = 'relu',input_shape=(2*sliding, 1)))
		model1.add(Dense(1, activation = 'relu'))

		# model 2 layer 4-2 neural
		model2 = Sequential()
		model2.add(LSTM(4,return_sequences=True, activation = 'relu',input_shape=(2*sliding, 1)))
		model2.add(LSTM(2, activation = 'relu'))
		model2.add(Dense(1, activation = 'relu'))

		# model 2 layer 32-4 neural
		model3 = Sequential()
		model3.add(LSTM(32,return_sequences=True, activation = 'relu',input_shape=(2*sliding, 1)))
		model3.add(LSTM(4, activation = 'relu'))
		model3.add(Dense(1, activation = 'relu'))

		# model 2 layer 5122-4 neural
		model4 = Sequential()
		model4.add(LSTM(512,return_sequences=True, activation = 'relu',input_shape=(2*sliding, 1)))
		model4.add(LSTM(4, activation = 'relu'))
		model4.add(Dense(1, activation = 'relu'))

		# model 3 layer 32-8-2 neural
		model5 = Sequential()
		model5.add(LSTM(32,return_sequences=True, activation = 'relu',input_shape=(2*sliding, 1)))
		model5.add(LSTM(8, activation = 'relu',return_sequences=True))
		model5.add(LSTM(2, activation = 'relu'))
		model5.add(Dense(1, activation = 'relu'))

		# model 3 layer 32-16-4 neural
		model6 = Sequential()
		model6.add(LSTM(32,return_sequences=True, activation = 'relu',input_shape=(2*sliding, 1)))
		model6.add(LSTM(16, activation = 'relu',return_sequences=True))
		model6.add(LSTM(4, activation = 'relu'))
		model6.add(Dense(1, activation = 'relu'))

		for k in range(6):
			if (k==0):
				model = model1
			elif (k == 1):
				model = model2
			elif (k == 2):
				model = model3
			elif (k == 3):
				model = model4
			elif (k == 4):
				model = model5
			elif (k == 5):
				model = model6

			modelName = "model" + str(k+1)
			print modelName
			optimizerArr = ['adam']
			for optimize in optimizerArr:
				print optimize
				model.compile(loss='mean_squared_error' ,optimizer=optimize , metrics=['mean_squared_error'])
				history = model.fit(trainX, trainY, epochs=2000, batch_size=batch_size, verbose=2,validation_split=0.25,
				 							callbacks=[EarlyStopping(monitor='loss', patience=20, verbose=1)])
				# make predictions
				# list all data in history
				print(history.history.keys())
				# summarize history for accuracy
				# summarize history for loss
				plt.plot(history.history['loss'])
				plt.plot(history.history['val_loss'])
				plt.title('model loss')
				plt.ylabel('loss')
				plt.xlabel('epoch')
				plt.legend(['train', 'test'], loc='upper left')
				# plt.show()
				plt.savefig('results/multivariate/memFuzzy/%s/history_sliding=%s_batchsize=%s_optimize=%s.png'%(modelName,sliding,batch_size,optimize))
				testPredict = model.predict(testX)

				print len(testPredict), len(testY)
				# invert predictions
				testPredictInverse = scaler.inverse_transform(testPredict)
				print testPredictInverse
				# calculate root mean squared error

				testScoreRMSE = math.sqrt(mean_squared_error(testY, testPredictInverse[:,0]))
				testScoreMAE = mean_absolute_error(testY, testPredictInverse[:,0])
				print('Test Score: %.6f RMSE' % (testScoreRMSE))
				print('Test Score: %.6f MAE' % (testScoreMAE))
				
				testDf = pd.DataFrame(np.array(testPredictInverse))
				testDf.to_csv('results/multivariate/memFuzzy/%s/testPredictInverse_sliding=%s_batchsize=%s_optimize=%s.csv'%(modelName,sliding,batch_size,optimize), index=False, header=None)
				errorScore=[]
				errorScore.append(testScoreRMSE)
				errorScore.append(testScoreMAE)
				errorDf = pd.DataFrame(np.array(errorScore))
				errorDf.to_csv('results/multivariate/memFuzzy/%s/error_sliding=%s_batchsize=%s_optimize=%s.csv'%(modelName,sliding,batch_size,optimize), index=False, header=None)
