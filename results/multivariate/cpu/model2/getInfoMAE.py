import matplotlib.pyplot as plt
import pandas as pd 
from pandas import read_csv
import numpy as np

batch_size_array = [8,16,32,64,128]

sliding_widow = [2,3,4,5]
optimizerArr=['adam', 'SGD']
MAEAdam = list()
MAESGD = list()
MAEerror = list()
for sliding in sliding_widow:
	MAEslidingAdam = []
	tach = ['','','','','']
	MAEslidingSGD = []
	for batch_size in batch_size_array:
		if sliding==5 and batch_size ==128:
			break
		for optimize in optimizerArr:
			if optimize== 'adam':
				error_df = read_csv('error_sliding=%s_batchsize=%s_optimize=%s.csv'%(sliding,batch_size, optimize), header=None, index_col=False, engine='python')
				MAE = error_df.values[1][0]
				MAEslidingAdam.append(MAE)
			elif optimize == 'SGD':
				error_df = read_csv('error_sliding=%s_batchsize=%s_optimize=%s.csv'%(sliding,batch_size, optimize), header=None, index_col=False, engine='python')
				MAE = error_df.values[1][0]
				MAEslidingSGD.append(MAE)
	MAEAdam.append(MAEslidingAdam)
	MAESGD.append(MAEslidingSGD)
print MAEAdam
for i in range(len(MAEAdam)):
	MAEerror.append(MAEAdam[i])
MAEerror.append(tach)
for i in range(len(MAESGD)):
	MAEerror.append(MAESGD[i])

MAEAdamDf = pd.DataFrame(MAEAdam)
MAEAdamDf.to_csv('Adam.csv', index=False, header=None)
MAESGDDf = pd.DataFrame(MAESGD)
MAESGDDf.to_csv('SGD.csv', index=False, header=None)
MAEDf = pd.DataFrame(MAEerror)
MAEDf.to_csv('MAE.csv', index=False, header=None)