# #!/usr/bin/env python
# # coding: utf-8

# # In[2]:


# from google.colab import drive
# drive.mount('/content/drive/')


# # In[3]:


# get_ipython().system('ls "/content/drive/My Drive/EE 569_HW 6"')


# # In[9]:


# cd Documents/DIP/HW6/EE569_2020Spring-master


# # In[7]:


# cd Users/sonalisreedhar/Documents/DIP/HW6/EE569_2020Spring-master


# # In[6]:


# cd /content/drive/My Drive/EE 569_HW 6/EE569_2020Spring


# # In[9]:


# get_ipython().system('git clone https://github.com/USC-MCL/EE569_2020Spring.git')


# # In[10]:


# ls


# In[11]:


import tensorflow
import saab
import pickle
import numpy as np
import sklearn
import cv2
import keras
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from keras.datasets import cifar10
from skimage.util import view_as_windows
from cwSaab import cwSaab
from keras.utils import to_categorical 
from pixelhop2 import Pixelhop2
from sklearn.model_selection import train_test_split


# In[12]:


from skimage.util import view_as_windows
import numpy as np
from skimage.measure import block_reduce
from sklearn.model_selection import train_test_split


# In[13]:


# Loading of CIFAR 10 dataset
(data_train, label_train),(data_test, label_test) = cifar10.load_data()

# Reshaping check of the loaded dataset
print("Shape of training data:")
print(data_train.shape)
print(label_train.shape)
print("Shape of test data:")
print(data_test.shape)
print(label_test.shape)

# Data preprocessing
data_train = (data_train.astype('float'))
data_test = (data_test.astype('float'))
data_train /= 255.0
data_test /= 255.0

# Data Normalisation
mean = np.mean(data_train, axis=0)
std = np.std(data_train, axis=0)

data_train -= mean
data_test -= mean
data_train /= std
data_test /= std

# Stratification of data
fit_data_train, data_val, fit_label_train, label_val = train_test_split(data_train, label_train, test_size = 0.8, random_state = 0, stratify = label_train)

# Defining Neighborhood Construction
def Shrink(X, shrinkArg):
        win = shrinkArg['win']
        stride = shrinkArg['stride']
        ch = X.shape[-1]
        X = view_as_windows(X, (1,win,win,ch), (1,stride,stride,ch))
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
        Xmax = block_reduce (X, block_size = (1,2,2,1), func=np.max)
        return Xmax

def Concat(X, concatArg):
        return X

SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':True, 'batch':None, 'cw': False},
            {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw': True}, 
            {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw': True}]
shrinkArgs = [{'func':Shrink, 'win':5, 'stride': 1},
                {'func': Shrink, 'win':5, 'stride': 1}, 
                {'func': Shrink, 'win':5, 'stride': 1}]
concatArg = {'func':Concat}

print(" -----> depth=3")
import time
start_time = time.clock()
p2 = Pixelhop2(depth=3, TH1=0.001, TH2=0.0001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
p2.fit(fit_data_train)
# output = p2.transform(data_train)
output1 = p2.transform(data_train[:10000,:,:])
output2 = p2.transform(data_train[10000:20000,:,:])
output3 = p2.transform(data_train[20000:30000,:,:])
output4 = p2.transform(data_train[30000:40000,:,:])
output5 = p2.transform(data_train[40000:50000,:,:])

print(output1[0].shape, output1[1].shape, output1[2].shape)
print(output2[0].shape, output2[1].shape, output2[2].shape)
print(output3[0].shape, output3[1].shape, output3[2].shape)
print(output4[0].shape, output4[1].shape, output4[2].shape)
print(output5[0].shape, output5[1].shape, output5[2].shape)

output_1 = np.concatenate((output1[0],output2[0],output3[0],output4[0],output5[0]))
output_2 = np.concatenate((output1[1],output2[1],output3[1],output4[1],output5[1]))
output_3 = np.concatenate((output1[2],output2[2],output3[2],output4[2],output5[2]))
print(output_1.shape, output_2.shape, output_3.shape)
# print(output[0].shape, output[1].shape, output[2].shape)
print("------- DONE -------\n")


# In[14]:


#Cross Entropy1
from cross_entropy import Cross_Entropy
temp1 = output_1.reshape((len(output_1), -1))
print("input feature shape is: %s" %str(temp1.shape))
fit_data_train, data_val, fit_label_train, label_val = train_test_split(data_train, label_train, test_size = 0.8, random_state = 0, stratify = label_train) 
CE = Cross_Entropy(num_class = 10, num_bin = 5)
output1Feature = np.zeros((temp1.shape[-1]))

for i in range((temp1.shape[-1])):
  output1Feature[i] = CE.KMeans_Cross_Entropy(temp1[:,i].reshape(-1,1), label_train)
  print("--> KMeans CE: %s"%str(output1Feature[i]))
print("------DONE-----\n")


# In[16]:


#Cross Entropy2
from cross_entropy import Cross_Entropy
temp2 = output_2.reshape((len(output_2), -1))
print("input feature shape is: %s" %str(temp2.shape))
fit_data_train, data_val, fit_label_train, label_val = train_test_split(data_train, label_train, test_size = 0.8, random_state = 0, stratify = label_train) 
CE = Cross_Entropy(num_class = 10, num_bin = 5)
output2Feature = np.zeros((temp2.shape[-1]))

for i in range((temp2.shape[-1])):
  output2Feature[i] = CE.KMeans_Cross_Entropy(temp2[:,i].reshape(-1,1), label_train)
  print("--> KMeans CE: %s"%str(output2Feature[i]))
print("------DONE-----\n")


# In[17]:


#Cross Entropy3
from cross_entropy import Cross_Entropy
temp3 = output_3.reshape((len(output_3), -1))
print("input feature shape is: %s" %str(temp3.shape))
fit_data_train, data_val, fit_label_train, label_val = train_test_split(data_train, label_train, test_size = 0.8, random_state = 0, stratify = label_train) 
CE = Cross_Entropy(num_class = 10, num_bin = 5)
output3Feature = np.zeros((temp3.shape[-1]))

for i in range((temp3.shape[-1])):
  output3Feature[i] = CE.KMeans_Cross_Entropy(temp3[:,i].reshape(-1,1), label_train)
  print("--> KMeans CE: %s"%str(output3Feature[i]))
print("------DONE-----\n")


# In[18]:


print(output1Feature.shape)
print(output2Feature.shape)
print(output3Feature.shape)


# In[19]:



#from sklearn import datasets
from sklearn.model_selection import train_test_split
from llsr import LLSR as myLLSR
from lag import LAG


# In[68]:


#LAG 1 layer
idx1 = np.argsort(output1Feature)
#idx2.shape
slice1 = idx1[:4018]
slice1.shape
output_1.shape
output_1 = output_1.reshape(len(output_1),-1)
t1 = (np.transpose(output_1))
t1.shape
in_lag1 = t1[slice1]
in_lag1 = in_lag1.reshape(len(in_lag1),-1)
in_lag_1=np.transpose(in_lag1)
in_lag_1.shape
######################################
lag1 = LAG(encode='distance', num_clusters = [5,5,5,5,5,5,5,5,5,5], alpha=10, learner=myLLSR(onehot=False))  
lag1.fit(in_lag_1, label_train)
X_train_trans1 = lag1.transform(in_lag_1)
X_train_predprob1 = lag1.predict_proba(in_lag_1)
# print(" --> train acc: %s"%str(lag.score(out1, y_train)))
# print(" --> test acc.: %s"%str(lag.score(data_test,label_test)))
print("------- DONE -------\n")


# In[69]:


#LAG 2 layer 
idx2 = np.argsort(output2Feature)
#idx2.shape
slice2 = idx2[:2687]
slice2.shape
output_2.shape
output_2 = output_2.reshape(len(output_2),-1)
t2 = (np.transpose(output_2))
t2.shape
in_lag2 = t2[slice2]
in_lag2 = in_lag2.reshape(len(in_lag2),-1)
in_lag_2=np.transpose(in_lag2)
in_lag_2.shape


# In[22]:


lag2 = LAG(encode='distance', num_clusters = [5,5,5,5,5,5,5,5,5,5], alpha=10, learner=myLLSR(onehot=False))  
lag2.fit(in_lag_2, label_train)
X_train_trans2 = lag2.transform(in_lag_2)
X_train_predprob2 = lag2.predict_proba(in_lag_2)
# print(" --> train acc: %s"%str(lag.score(out1, y_train)))
# print(" --> test acc.: %s"%str(lag.score(data_test,label_test)))
print("------- DONE -------\n")


# In[70]:


#LAG 3 layer
idx3 = np.argsort(output3Feature)
#idx2.shape
slice3 = idx3[:250]
# slice1.shape
# output_1.shape
output_3 = output_3.reshape(len(output_3),-1)
t3 = (np.transpose(output_3))
# t2.shape
in_lag3 = t3[slice3]
in_lag3 = in_lag3.reshape(len(in_lag3),-1)
in_lag_3=np.transpose(in_lag3)
in_lag_3.shape
#################################################
lag3 = LAG(encode='distance', num_clusters = [5,5,5,5,5,5,5,5,5,5], alpha=10, learner=myLLSR(onehot=False))  
lag3.fit(in_lag_3, label_train)
X_train_trans3 = lag3.transform(in_lag_3)
X_train_predprob3 = lag3.predict_proba(in_lag_3)
# print(" --> train acc: %s"%str(lag.score(out1, y_train)))
# print(" --> test acc.: %s"%str(lag.score(data_test,label_test)))
print("------- DONE -------\n")


# In[24]:


#print outputs from LAG units
print(X_train_trans1.shape)
print(X_train_trans2.shape)
print(X_train_trans3.shape)


# In[25]:


Lag_ = np.concatenate((X_train_trans1,X_train_trans2,X_train_trans3),axis=1)
print(Lag_.shape)


# # Preprocessing for Test dataset

# In[100]:


test_out = p2.transform(data_test)


# In[103]:


print(test_out[0].shape)
print(test_out[1].shape)
print(test_out[2].shape)


# In[112]:


test1=test_out[0].reshape(len(test_out[0]),-1)
test2=test_out[1].reshape(len(test_out[1]),-1)
test3=test_out[2].reshape(len(test_out[2]),-1)
print(test1.shape)
print(test2.shape)
print(test3.shape)


# In[115]:


#for input to LAG 1
test1=np.transpose(test1)
print(test1.shape)
#########

op = test1[slice1]
op = op.reshape(len(op),-1)
op=np.transpose(op)
print(op.shape)


# In[119]:


#for input to LAG 2
test2=np.transpose(test2)
print(test2.shape)
#########

op2 = test2[slice2]
op2 = op2.reshape(len(op2),-1)
op2=np.transpose(op2)
print(op2.shape)


# In[121]:


#for input to LAG 3
test3=np.transpose(test3)
print(test3.shape)
#########

op3 = test3[slice3]
op3 = op3.reshape(len(op3),-1)
op3=np.transpose(op3)
print(op3.shape)


# In[118]:


#Lag1 for test
print(" input feature shape: %s"%str(op.shape))

# lag_test1 = LAG(encode='distance', num_clusters=[5,5,5,5,5,5,5,5,5,5], alpha=10, learner=myLLSR(onehot=False))  
# lag_test1.fit(sort_1, label_test)
X_test_trans1 = lag1.transform(op)
X_test_predprob1  = lag1.predict_proba(op)
#print(" --> train acc: %s"%str(lag3.score(sort3, y_train)))
#print(" --> test acc.: %s"%str(lag.score(x_test1, y_test)))
print(X_test_trans1.shape)
print("------- DONE -------\n")
#**********


# In[120]:


#Lag2 for test
print(" input feature shape: %s"%str(op2.shape))

# lag_test1 = LAG(encode='distance', num_clusters=[5,5,5,5,5,5,5,5,5,5], alpha=10, learner=myLLSR(onehot=False))  
# lag_test1.fit(sort_1, label_test)
X_test_trans2 = lag2.transform(op2)
X_test_predprob2  = lag2.predict_proba(op2)
#print(" --> train acc: %s"%str(lag3.score(sort3, y_train)))
#print(" --> test acc.: %s"%str(lag.score(x_test1, y_test)))
print(X_test_trans2.shape)
print("------- DONE -------\n")
#**********


# In[122]:


#Lag3 for test
print(" input feature shape: %s"%str(op3.shape))

# lag_test1 = LAG(encode='distance', num_clusters=[5,5,5,5,5,5,5,5,5,5], alpha=10, learner=myLLSR(onehot=False))  
# lag_test1.fit(sort_1, label_test)
X_test_trans3 = lag3.transform(op3)
X_test_predprob3  = lag3.predict_proba(op3)
#print(" --> train acc: %s"%str(lag3.score(sort3, y_train)))
#print(" --> test acc.: %s"%str(lag.score(x_test1, y_test)))
print(X_test_trans3.shape)
print("------- DONE -------\n")
#**********


# In[123]:


Lag_test = np.concatenate((X_test_trans1,X_test_trans2,X_test_trans3),axis = 1)
Lag_test.shape


# In[124]:




#Random forest classifier
from sklearn.ensemble import RandomForestClassifier

 # create regressor object 
regressor = RandomForestClassifier(n_estimators = 100, random_state = 0) 
  
# fit the regressor with x and y data 
regressor.fit(Lag_, label_train)

# x_test1 = data_test.reshape(len(data_test),-1)
# x_test1.shape
from sklearn.metrics import accuracy_score

y_pred = regressor.predict(Lag_test)
acc  = accuracy_score(label_test,y_pred)
y_pred.shape
print(acc)

#Confusion matrix
from sklearn.metrics import confusion_matrix

cf = confusion_matrix(label_test, y_pred)
print(cf)

#Heat map
import seaborn as sns
sns.heatmap(cf, annot=True)

