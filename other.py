
import time

import numpy as np
import pandas as pd


AC_BIAO = pd.read_excel('e:/pythonpuod/pythonpuod/2_ok.xlsx', sheet_name="Sheet8")

M=0.1

Y = AC_BIAO.iloc[:, 0:2].values
AC= AC_BIAO.iloc[:,2].values


#######################sos(2012)################################
from others.sos import SOS
a=time.time()
sos= SOS(contamination=M, perplexity=4.5,)
sos_lable=sos.fit_predict(Y)
b=time.time()
print("time",b-a)
print("sosnum:", np.sum(sos_lable ==1))
print(sos_lable)


TP=0
FP=0
FN=0
TN=0
for i in range(len(sos_lable)):
    if(AC[i]==0 and sos_lable[i]==0):
        TP=TP+1
    if(AC[i]==1 and sos_lable[i]==0):
        FP=FP+1
    if (AC[i] == 0 and sos_lable[i] == 1):
        FN=FN+1
    if (AC[i] == 1 and sos_lable[i] == 1):
        TN=TN+1
print(TP,FP,FN,TN)
ZQL=(TP+TN)/(TP+TN+FP+FN)
JQL=(TP)/(TP+FP)
ZHL=TP/(TP+FN)
JZL=FP/(TN+FP)
print("Accuracy",ZQL,"Precision",JQL,"Recall",ZHL,"FPR",JZL)

#
# ##############################INNE####################

from others.inne import INNE
a=time.time()
inne= INNE(n_estimators=200,max_samples="auto",contamination=M,random_state=None)
inne_lable=inne.fit_predict(Y)
b=time.time()
print("time",b-a)
print("INNEnum:", np.sum(inne_lable ==1))
print(inne_lable)
TP=0
FP=0
FN=0
TN=0
for i in range(len(inne_lable)):
    if(AC[i]==0 and inne_lable[i]==0):
        TP=TP+1
    if(AC[i]==1 and inne_lable[i]==0):
        FP=FP+1
    if (AC[i] == 0 and inne_lable[i] == 1):
        FN=FN+1
    if (AC[i] == 1 and inne_lable[i] == 1):
        TN=TN+1
print(TP,FP,FN,TN)
ZQL=(TP+TN)/(TP+TN+FP+FN)
JQL=(TP)/(TP+FP)
ZHL=TP/(TP+FN)
JZL=FP/(TN+FP)
print("Accuracy",ZQL,"Precision",JQL,"Recall",ZHL,"FPR",JZL)



# # #################################GMM###############

from others.gmm import GMM
a=time.time()
gmm= GMM(n_components=1,covariance_type="full",tol=1e-3,reg_covar=1e-6,max_iter=100,n_init=1,init_params="kmeans",weights_init=None,means_init=None,precisions_init=None,random_state=None,warm_start=False,contamination=M,)
gmm_lable=gmm.fit_predict(Y)
b=time.time()
print("time",b-a)
print("GMMnum:", np.sum(gmm_lable ==1))
print(gmm_lable)

TP=0
FP=0
FN=0
TN=0
for i in range(len(gmm_lable)):
    if(AC[i]==0 and gmm_lable[i]==0):
        TP=TP+1
    if(AC[i]==1 and gmm_lable[i]==0):
        FP=FP+1
    if (AC[i] == 0 and gmm_lable[i] == 1):
        FN=FN+1
    if (AC[i] == 1 and gmm_lable[i] == 1):
        TN=TN+1
print(TP,FP,FN,TN)
ZQL=(TP+TN)/(TP+TN+FP+FN)
JQL=(TP)/(TP+FP)
ZHL=TP/(TP+FN)
JZL=FP/(TN+FP)
print("Accuracy",ZQL,"Precision",JQL,"Recall",ZHL,"FPR",JZL)

# #
# # # #######################LSCP(2019)################################
from others.lscp import LSCP
from others.knn import KNN
from others.lof import LOF
from others.cblof import CBLOF


a=time.time()
knn = KNN()
lof = LOF()
cblof = CBLOF()


lscp = LSCP([knn, lof, cblof],contamination=M)

lscp.fit(Y)

# 预测异常值
y_pred = lscp.predict(Y)
b=time.time()
print("time",b-a)
print("LSCPnum:", np.sum(y_pred ==1))
print(y_pred)

TP=0
FP=0
FN=0
TN=0
for i in range(len(y_pred)):
    if(AC[i]==0 and y_pred[i]==0):
        TP=TP+1
    if(AC[i]==1 and y_pred[i]==0):
        FP=FP+1
    if (AC[i] == 0 and y_pred[i] == 1):
        FN=FN+1
    if (AC[i] == 1 and y_pred[i] == 1):
        TN=TN+1
print(TP,FP,FN,TN)
ZQL=(TP+TN)/(TP+TN+FP+FN)
JQL=(TP)/(TP+FP)
ZHL=TP/(TP+FN)
JZL=FP/(TN+FP)
print("Accuracy",ZQL,"Precision",JQL,"Recall",ZHL,"FPR",JZL)

# # #####################LOCI###############################
from others.loci import LOCI
a=time.time()
loci= LOCI(contamination=0.1, alpha=0.5, k=3)#####K=0,1,2,3---70-80,60-70,20-30,5
loci_lable=loci.fit_predict(Y)
b=time.time()
print("time",b-a)
print("LOCInum:", np.sum(loci_lable ==1))
print(loci_lable)

TP=0
FP=0
FN=0
TN=0
for i in range(len(loci_lable)):
    if(AC[i]==0 and loci_lable[i]==0):
        TP=TP+1
    if(AC[i]==1 and loci_lable[i]==0):
        FP=FP+1
    if (AC[i] == 0 and loci_lable[i] == 1):
        FN=FN+1
    if (AC[i] == 1 and loci_lable[i] == 1):
        TN=TN+1
print(TP,FP,FN,TN)
ZQL=(TP+TN)/(TP+TN+FP+FN)
JQL=(TP)/(TP+FP)
ZHL=TP/(TP+FN)
JZL=FP/(TN+FP)
print("Accuracy",ZQL,"Precision",JQL,"Recall",ZHL,"FPR",JZL)

# # ########################COPOD####################
from others.copod import COPOD
a=time.time()
copod= COPOD(contamination=M, n_jobs=1)
copod_lable=copod.fit_predict(Y)
b=time.time()
print("time",b-a)
print("COPODnum:", np.sum(copod_lable ==1))
print(copod_lable)

TP=0
FP=0
FN=0
TN=0
for i in range(len(copod_lable)):
    if(AC[i]==0 and copod_lable[i]==0):
        TP=TP+1
    if(AC[i]==1 and copod_lable[i]==0):
        FP=FP+1
    if (AC[i] == 0 and copod_lable[i] == 1):
        FN=FN+1
    if (AC[i] == 1 and copod_lable[i] == 1):
        TN=TN+1
print(TP,FP,FN,TN)
ZQL=(TP+TN)/(TP+TN+FP+FN)
JQL=(TP)/(TP+FP)
ZHL=TP/(TP+FN)
JZL=FP/(TN+FP)
print("Accuracy",ZQL,"Precision",JQL,"Recall",ZHL,"FPR",JZL)

