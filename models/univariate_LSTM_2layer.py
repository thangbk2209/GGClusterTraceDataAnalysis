# LSTM for international airline passengers problem with window regression framing
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.callbacks import TensorBoard, EarlyStopping
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)
sliding_window=[4]
# load the dataset
# dataframe = read_csv('/home/nguyen/LSTM_GoogleTraceData/data/Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv', usecols=[0], engine='python', skipfooter=3)
dataframe = read_csv('./data/Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv', usecols=[0], engine='python', skipfooter=3)

dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
for sliding in sliding_window:

	batch_size_arr=[8,16,32,64,128]
	for batch_size in batch_size_arr: 
		# split into train and test sets
		print "sliding = %s, batch_size = %s"%(sliding, batch_size)
		# train_size = 2880
		train_size = int(len(dataset)* 0.67)
		test_size = len(dataset) - train_size
		train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
		# reshape into X=t and Y=t+1
		look_back = sliding
		trainX, trainY = create_dataset(train, look_back)
		testX, testY = create_dataset(test, look_back)
		testY = scaler.inverse_transform([testY])
		# reshape input to be [samples, time steps, features]
		trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
		testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
		print 'train X,testX'
		print trainX
		print testX
		print trainX[0], trainY[0]
		# create and fit the LSTM network

		model1 = Sequential()
		model1.add(LSTM(4, activation = 'relu',input_shape=(sliding, 1)))
		model1.add(Dense(1, activation = 'relu'))

		# model 2 layer 4-2 neural
		model2 = Sequential()
		model2.add(LSTM(4,return_sequences=True, activation = 'relu',input_shape=(sliding, 1)))
		model2.add(LSTM(2, activation = 'relu'))
		model2.add(Dense(1, activation = 'relu'))

		# model 2 layer 32-4 neural
		model3 = Sequential()
		model3.add(LSTM(32,return_sequences=True, activation = 'relu',input_shape=(sliding, 1)))
		model3.add(LSTM(4, activation = 'relu'))
		model3.add(Dense(1, activation = 'relu'))

		# model 2 layer 5122-4 neural
		model4 = Sequential()
		model4.add(LSTM(512,return_sequences=True, activation = 'relu',input_shape=(sliding, 1)))
		model4.add(LSTM(4, activation = 'relu'))
		model4.add(Dense(1, activation = 'relu'))

		# model 3 layer 32-8-2 neural
		model5 = Sequential()
		model5.add(LSTM(32,return_sequences=True, activation = 'relu',input_shape=(sliding, 1)))
		model5.add(LSTM(8, activation = 'relu',return_sequences=True))
		model5.add(LSTM(2, activation = 'relu'))
		model5.add(Dense(1, activation = 'relu'))

		# model 3 layer 32-16-4 neural
		model6 = Sequential()
		model6.add(LSTM(32,return_sequences=True, activation = 'relu',input_shape=(sliding, 1)))
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
			optimizerArr = ['adam','SGD']

			# model = Sequential()
			# model.add(LSTM(4, activation = 'relu',input_shape=(sliding,1)))
			# # model.add(LSTM(4, activation = 'relu'))
			# model.add(Dense(1))
			for optimize in optimizerArr:
				print optimize
				model.compile(loss='mean_squared_error', optimizer=optimize )
				
				history = model.fit(trainX, trainY, epochs=2000, batch_size=batch_size, verbose=2,validation_split=0.25,
				 							callbacks=[EarlyStopping(monitor='loss', patience=20, verbose=1)])
			
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
				plt.savefig('results/univariate/cpu/%s/history_sliding=%s_batchsize=%s_optimize=%s.png'%(modelName,sliding,batch_size,optimize))
				# make predictions
				testPredict = model.predict(testX)
				# invert predictions
				
				testPredictInverse = scaler.inverse_transform(testPredict)
				
				print 'len(testY), len(testPredict)'
				print len(testY[0]), len(testPredict)
				# calculate root mean squared error
				
				testScoreRMSE = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
				testScoreMAE = mean_absolute_error(testY[0], testPredictInverse[:,0])
				print('Test Score: %f RMSE' % (testScoreRMSE))
				print('Test Score: %f MAE' % (testScoreMAE))
			
				testDf = pd.DataFrame(np.array(testPredictInverse))
				testDf.to_csv('results/univariate/cpu/%s/testPredictInverse_sliding=%s_batchsize=%s_optimize=%s.csv'%(modelName,sliding,batch_size,optimize), index=False, header=None)
				errorScore=[]
				errorScore.append(testScoreRMSE)
				errorScore.append(testScoreMAE)
				errorDf = pd.DataFrame(np.array(errorScore))
				errorDf.to_csv('results/univariate/cpu/%s/error_sliding=%s_batchsize=%s_optimize=%s.csv'%(modelName,sliding,batch_size,optimize), index=False, header=None)