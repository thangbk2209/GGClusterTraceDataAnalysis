import matplotlib.pyplot as plt
import pandas as pd 
from pandas import read_csv
import numpy as np

batch_size_array = [8,16,32,64,128]
sliding_widow = [2,3,4,5]
folderArr = ["cpu", "mem"]
MAEerror = list()
for folder in folderArr:	
	MAESGD = list()
	MAEerror.append([folder,'','','',''])
	tach = ['','','','','']
	for sliding in sliding_widow:
		MAEsliding = list()
		for batch_size in batch_size_array:
			error_df = read_csv('%s/error_sliding=%s_batchsize=%s_optimize=sgd.csv'%(folder, sliding, batch_size), header=None, index_col=False, engine='python')
			MAE = error_df.values[1][0]
			MAEsliding.append(MAE)
		MAESGD.append(MAEsliding)

	for i in range(len(MAESGD)):
		MAEerror.append(MAESGD[i])
MAEDf = pd.DataFrame(MAEerror)
MAEDf.to_csv('MAE.csv', index=False, header=None)