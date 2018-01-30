import matplotlib.pyplot as plt
import pandas as pd 
from pandas import read_csv
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
# colnames=['time_stamp','numberOfTaskIndex','numberOfMachineId','meanCPUUsage','CMU','AssignMem','unmapped_cache_usage','page_cache_usage', 'max_mem_usage','mean_diskIO_time','mean_local_disk_space','max_cpu_usage', 'max_disk_io_time', 'cpi', 'mai','sampled_cpu_usage']
# df = read_csv('/home/nguyen/learnRNNs/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
colnames = ['cpu','mem','disk_io_time','disk_space'] 
# colnames = ['cpu','mem','disk_io_time','disk_space'] 
batch_size_array = [8,16,32,64,128]
realFile = ['/home/nguyen/GGTraceAnalysis/data/5_Fuzzy_Mem_sampling_617685_metric_10min_datetime_origin.csv']
folderArr = ["model1"]

sliding_widow = [2]
for folderName in folderArr: 
	for sliding in sliding_widow:
		for batch_size in batch_size_array:
				
			Real_df = read_csv('%s'%(realFile[0]), header=None, index_col=False, names=colnames, engine='python')
			Pred_TestInverse_df = read_csv('%s/testPredictInverse_sliding=%s_batchsize=%s_optimize=SGD.csv'%(folderName, sliding, batch_size), header=None, index_col=False, engine='python')
			
			error_df = read_csv('%s/error_sliding=%s_batchsize=%s_optimize=SGD.csv'%(folderName, sliding, batch_size), header=None, index_col=False, engine='python')
			RealDataset = Real_df['cpu'].values

			train_size = int(len(RealDataset) *0.67 )
			test_size = len(RealDataset) - train_size
			print RealDataset
			Pred_TestInverse = Pred_TestInverse_df.values

			# predictions = []
			# for i in range(100):
			# 	predictions.append(Pred_TestInverse[i])
			# TestPredDataset = Pred_Test_df.values
			RMSE = error_df.values[0][0]
			MAE = error_df.values[1][0]
			print RMSE
			realTestData = []
			# for j in range(train_size+sliding, len(RealDataset),1):
			for j in range(train_size+sliding, len(RealDataset),1):
				realTestData.append(RealDataset[j])
			print len(realTestData)
			print len(Pred_TestInverse)
			# testScoreMAE = mean_absolute_error(Pred_TestInverse, realTestData)
			# print 'test score', testScoreMAE
			ax = plt.subplot()
			ax.plot(realTestData,label="Actual")
			ax.plot(Pred_TestInverse,label="predictions")
			# ax.plrot(TestPred,label="Test")
			plt.xlabel("TimeStamp")
			plt.ylabel("CPU")
			ax.text(0,0, '%s_testScore-sliding=%s-batch_size=%s_optimise=SGD: %s RMSE- %s MAE'%(folderName, sliding,batch_size, RMSE,MAE), style='italic',
			        bbox={'facecolor':'red', 'alpha':0.5, 'pad':8})
			plt.legend()
			# plt.savefig('mem5.png')
			plt.show()

