import matplotlib as mpl
mpl.use('Agg')
import math
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression as LR
import numpy as np
import pandas as pd 


from collections import Counter
# Counter(array1.most_common(1))
# print math.log(2,2)

# a =[1,2,3,4,5,1,2]
# print Counter(a.most_common(1))

def entro(X):
	x = [] # luu lai danh sach cac gia tri X[i] da tinh
	tong_so_lan = 0
	result = 0
	p=[]
	for i in range(len(X)):
		if Counter(x)[X[i]]==0:
			so_lan = Counter(X)[X[i]]
			tong_so_lan += so_lan
			x.append(X[i])
			P = 1.0*so_lan / len(X)
			p.append(P)
			result -= P * math.log(P,2)
		if tong_so_lan == len(X):
			break
	return result
# entropy(X|Y)
def entroXY(X,Y):
	y = []
	result = 0
	pY = []
	tong_so_lan_Y = 0
	for i in range(len(Y)):
		# print Counter(y)[Y[i]]
		if Counter(y)[Y[i]]==0:
			x=[]
			so_lan_Y = Counter(Y)[Y[i]]
			tong_so_lan_Y += so_lan_Y
			y.append(Y[i])
			PY = 1.0* so_lan_Y / len(Y)
			# vi_tri = Y.index(Y[i])
			vi_tri=[]
			for k in range(len(Y)):
				if Y[k] == Y[i]: 
					vi_tri.append(k)
			for j in range(len(vi_tri)):
				x.append(X[vi_tri[j]])
			entro_thanh_phan = entro(x)
			result += PY * entro_thanh_phan
		if tong_so_lan_Y == len(Y):
			break
	return result
def infomation_gain(X,Y):
	return entro(X) - entroXY(X,Y)
def symmetrical_uncertainly(X,Y):
	return 2.0*infomation_gain(X,Y)/(entro(X)+entro(Y))
# Link tham khao: https://math.stackexchange.com/questions/1222200/entropy-for-three-random-variables
# H(X,Y,Z) = Tong(P(x=xi, y=yi, z=zi)log2(P(x=xi, y=yi, z=zi)))
def entropyXYZ(X,Y,Z):
	length = len(X)
	print length
	x = []
	result = 0
	tongSoLanX = 0
	for i in range(len(X)):
		if Counter(x)[X[i]] == 0:
			newY=[]
			newZ=[]
			vi_tri = []
			soLanX = Counter(X)[X[i]]
			tongSoLanX += soLanX
			x.append(X[i])
			for j in range(len(X)):
				if X[j] == X[i]:
					vi_tri.append(j)
			YX = []
			ZX = []
			for k in range(len(vi_tri)):
				YX.append(Y[vi_tri[k]])  #YX la mang chua cac gia tri Y tuong ung voi cac gia tri X dang xet
				ZX.append(Z[vi_tri[k]])	 #ZX la mang chua cac gia tri Z tuong ung voi cac gia tri X dang xet
			y = []
			tongSoLanY = 0
			for t in range(len(YX)):
				if Counter(y)[YX[t]] == 0:
					soLanY = Counter(YX)[YX[t]]
					tongSoLanY += soLanY
					y.append(YX[t])
					vi_triYX=[]
					for m in range(len(YX)):
						if YX[m] == YX[t]:
							vi_triYX.append(m)
					ZYX = []
					for n in range(len(vi_triYX)):
						ZYX.append(ZX[vi_triYX[n]])
					z=[]
					tongSoLanZ = 0
					for p in range(len(ZYX)):
						if Counter(z)[ZYX[p]] == 0:
							soLanZ = Counter(ZYX)[ZYX[p]]
							z.append(ZYX[p])
							tongSoLanZ += soLanZ
							P = 1.0* soLanZ / length
							result -= P * math.log(P,2)
						if tongSoLanZ == len(ZYX):
							break
				if tongSoLanY == len(YX):
					break
		if tongSoLanX == length:
			break
	return result
def information_gainXYZ(X,Y,Z):
	return entro(X) + entro(Y) + entro(Z) - entropyXYZ(X,Y,Z)
def symmetrical_uncertainlyXYZ(X,Y,Z):
	return 3.0*information_gainXYZ(X,Y,Z)/(entro(X)+entro(Y) + entro(Z))

# colnames=['meanCPUUsage' ,'CMU' ,'AssignMem' ,'unmap_page_cache_memory_ussage' ,'page_cache_usage' ,'mean_local_disk_space', 'timeStamp']
# df = read_csv('/home/nguyen/spark-lab/spark-2.1.1-bin-hadoop2.7/google_cluster_analysis/results/my_offical_data_resource_TopJobId.csv', header=None, index_col=False, names=colnames)
# colnames=['cpu_rate','mem_usage','disk_io_time','disk_space']
colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space']
# df = read_csv('data/Fuzzy_data_resource_JobId_6336594489_5minutes.csv', header=None, index_col=False, names=colnames)
df = read_csv('data/5_Fuzzy_Mem_sampling_617685_metric_10min_datetime_origin.csv', header=None, index_col=False, names=colnames)

cpu_rate = df['cpu_rate'].values
mem_usage = df['mem_usage'].values
disk_io_time = df['disk_io_time'].values
disk_space = df['disk_space'].values

su=[]
# entropyGGTrace = []
# # numberOfEntropy = 0
# print infomation_gain(cpu_rate,mem_usage)
# print infomation_gain(mem_usage,cpu_rate)
# print 2.0 * math.log(0.002,2)
# print 'information_gainXYZ(cpu_rate,mem_usage,disk_space)'
# print information_gainXYZ(cpu_rate,mem_usage,disk_space)
# print 'information_gainXYZ(cpu_rate,mem_usage,disk_space)'
# print information_gainXYZ(mem_usage,disk_space,cpu_rate)
# print 'information_gainXYZ(cpu_rate,mem_usage,disk_space)'
# print information_gainXYZ(disk_space,cpu_rate,mem_usage)
 
print 'symmetrical_uncertainlyXYZ(cpu_rate,mem_usage,disk_space)'
print symmetrical_uncertainlyXYZ(cpu_rate,mem_usage,disk_space)
IG = []
for i in range(len(colnames)-2):
	print i
	IGi=[]
	for k in range(i+1, len(colnames)-1):
		print k
		for j in range(k+1,len(colnames)):
			print j
			IG.append(information_gainXYZ(df[colnames[i]].values,df[colnames[k]].values,df[colnames[j]].values))
		
print IG
# su=[[1,2,3],[2,3,4]]
# dataFuzzyDf = pd.DataFrame(np.array(su))
# dataFuzzyDf.to_csv('data/su_5_Fuzzy_Mem_sampling_617685_metric_10min_datetime_origin.csv', index=False, header=None)
