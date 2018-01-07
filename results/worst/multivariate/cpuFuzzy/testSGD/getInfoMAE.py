import matplotlib.pyplot as plt
import pandas as pd 
from pandas import read_csv
import numpy as np

batch_size_array = [8,16,32,64,128]
modelNameArr = ['model1','model2','model3','model4','model5']
sliding_widow = [2,3,4,5]
optimizerArr=['adam']

MAEerror = list()
for modelName in modelNameArr:	
	MAEAdam = list()
	MAESGD = list()
	MAEerror.append([modelName,'','','',''])
	tach = ['','','','','']
	for sliding in sliding_widow:
		MAEslidingAdam = []
		
		MAEslidingSGD = []
		for batch_size in batch_size_array:
			if sliding==5 and batch_size ==128:
				break
			for optimize in optimizerArr:
				if optimize== 'adam':
					error_df = read_csv('%s/error_sliding=%s_batchsize=%s_optimize=%s.csv'%(modelName, sliding, batch_size, optimize), header=None, index_col=False, engine='python')
					MAE = error_df.values[1][0]
					MAEslidingAdam.append(MAE)
				elif optimize == 'SGD':
					error_df = read_csv('%s/error_sliding=%s_batchsize=%s_optimize=%s.csv'%(modelName, sliding, batch_size, optimize), header=None, index_col=False, engine='python')
					MAE = error_df.values[1][0]
					MAEslidingSGD.append(MAE)
		MAEAdam.append(MAEslidingAdam)
		MAESGD.append(MAEslidingSGD)
	print MAEAdam
	for i in range(len(MAEAdam)):
		MAEerror.append(MAEAdam[i])
	MAEerror.append(['','','','',''])
	for i in range(len(MAESGD)):
		MAEerror.append(MAESGD[i])
	# MAEerror.append(MAEmodel)
	# MAEAdamDf = pd.DataFrame(MAEAdam)
	# MAEAdamDf.to_csv('%sAdam.csv'%(modelName), index=False, header=None)
	# MAESGDDf = pd.DataFrame(MAESGD)
	# MAESGDDf.to_csv('%sSGD.csv'%(modelName), index=False, header=None)
MAEDf = pd.DataFrame(MAEerror)
MAEDf.to_csv('MAE.csv', index=False, header=None)