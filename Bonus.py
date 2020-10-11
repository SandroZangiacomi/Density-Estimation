# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:39:52 2020

@author: Zangiacomi Sandro
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as m
import scipy.io

########PCA#############################################################


mat_label = scipy.io.loadmat('data/label.mat')
mat_data = scipy.io.loadmat('data/data.mat')
data_set_label=pd.DataFrame(mat_label['trueLabel'])
data_set_data=pd.DataFrame(mat_data['data'])


mean_data=data_set_data.mean(axis=1)
m_points=len(data_set_data.columns)


data_set_data_T=data_set_data.transpose()

Row=[]
for row in range(0,len(data_set_data_T.index)):
    Row.append(np.array(data_set_data_T.iloc[row,:].tolist())-np.asarray(mean_data).T)
    
X=np.asarray(Row)
Xt=X.T

Cov=(1/m_points)*Xt.dot(X)

k=5
def Eigen_Value_Vect_selected(data_set,k):
    Value, Vector=np.linalg.eig(data_set)
    Vector=np.real(Vector)
    Value=np.real(Value)  
    value_sorted=sorted(Value.tolist(),reverse=1) 
    Eigen_Vectors_Selected=[]
    for eigVal in value_sorted[0:k]:
        Eigen_Vectors_Selected.append(Vector[:,Value.tolist().index(eigVal)])
    return np.asarray(Eigen_Vectors_Selected), value_sorted[0:k]

Eigen_vector, Eigen_values=Eigen_Value_Vect_selected(Cov,k)

reduced_representation=[]

for data_point in Row:      
    reduced_representation.append([Eigen_vector[i].dot(data_point)/m.sqrt(Eigen_values[i]) for i in range(0,k)])

########PCA############################################################# 
    


C=2

pi_1= 0.5
pi_2= 0.5

mean_1=np.random.randn(784)
mean_2=np.random.randn(784)
 
Sigma1=np.random.randn(784,784)
Sigma2=np.random.randn(784,784)

Sigma1=Sigma1.T.dot(Sigma1)+np.identity(784)
Sigma2=Sigma2.T.dot(Sigma2)+np.identity(784)





def M_step(Projected_data,tau_0,tau_1,Sigma1,Sigma2,mean_1,mean_2,pi_1,pi_2):
    
    pi_1=np.sum(tau_0)/1990
    pi_2=np.sum(tau_1)/1990
    
    mean_1=tau_0.T.dot(np.asarray(Projected_data).T)/np.sum(tau_0)
    mean_2=tau_1.T.dot(np.asarray(Projected_data).T)/np.sum(tau_1)
    
    
    Centered_data_1=np.asarray(Projected_data.T-np.tile(mean_1,(1990,1)))
    Centered_data_2=np.asarray(Projected_data.T-np.tile(mean_2,(1990,1)))
    
    Sigma1=0
    Sigma2=0
    for i in range (0,1990):
        Sigma1+=tau_0.tolist()[i][0]*np.reshape(Centered_data_1[i],(784,1)).dot(np.reshape(Centered_data_1[i],(784,1)).T)
        Sigma2+=tau_1.tolist()[i][0]*np.reshape(Centered_data_2[i],(784,1)).dot(np.reshape(Centered_data_2[i],(784,1)).T)

    Sigma1=Sigma1/np.sum(tau_0)
    Sigma2=Sigma2/np.sum(tau_1)
    
  
    
    return pi_1,pi_2,mean_1,mean_2,Sigma1,Sigma2

def new_gaussian(Sigma,mean,data_set_data):
    Eigen_vector, Eigen_values=Eigen_Value_Vect_selected(Sigma,5)
    Centered_data=np.asarray(data_set_data.T-np.tile(mean,(1990,1)))
    Product=Eigen_vector.dot(Centered_data.T)
    Diag_values=np.diag([eig**(-1/2) for eig in Eigen_values])
    P=Diag_values.dot(Product)
    PP=[]
    for i in P.T :
        PP.append(np.linalg.norm(i))
    PP=np.asarray(PP)
        

    a=np.exp(PP)/np.prod( Eigen_values)**(1/2)
    return a
    
        
f=[]
X=[j for j in range(1,100)]
                    
for loop in range(0,100):    
    a=new_gaussian(Sigma1,mean_1,data_set_data)
    b=new_gaussian(Sigma2,mean_2,data_set_data)

    if loop>0:
        f.append(np.sum(np.log(pi_1*a+pi_2*b)))
    
        tau_0= pi_1 * a/(pi_1*a+pi_2*b)
        tau_0=np.reshape(tau_0, (1990,1))
        tau_1=pi_2 * b/(pi_1*a+pi_2*b)
        tau_1=np.reshape(tau_1, (1990,1))
        
        pi_1,pi_2,mean_1,mean_2,Sigma1,Sigma2= M_step(data_set_data,tau_0,tau_1,Sigma1,Sigma2,mean_1,mean_2,pi_1,pi_2)      
        print('loop {}'.format(loop))   
    
    
plt.figure()                  
plt.plot(X,f)
plt.xlabel('interations')
plt.ylabel('log-likelihood')
plt.show()
    
two_reconstruct =np.reshape(mean_1,(28,28))
six_reconstruct =np.reshape(mean_2,(28,28))

plt.figure()
plt.imshow(two_reconstruct.T,cmap='gray')
plt.figure()
plt.imshow(six_reconstruct.T,cmap='gray')




#d)


tau1=tau_0
tau2=tau_1


Augmented_data=data_set_data.T
Augmented_data['tau1']=tau1
Augmented_data['tau2']=tau2
Augmented_data['True_label']=data_set_label.T

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(data_set_data.T)


nbr_2=np.asarray(data_set_label).tolist()[0].count(2)
nbr_6=np.asarray(data_set_label).tolist()[0].count(6)

found_2=0
found_6=0
ite=0
for i in kmeans.labels_.tolist():
    if i==1 and np.asarray(data_set_label).tolist()[0][ite]==2:
        found_2+=1
    if i==0 and np.asarray(data_set_label).tolist()[0][ite]==6:
        found_6+=1
    ite+=1
        
k_means_miss=[ found_2/nbr_2,found_6/nbr_6]
   

found_2_G=0
found_6_G=0 
row=0
for i in Augmented_data['True_label']:
     if i==2 and Augmented_data.iloc[row,785]> Augmented_data.iloc[0,784]:
        found_2_G+=1
     if i==6 and Augmented_data.iloc[row,785]< Augmented_data.iloc[0,784]:
        found_6_G+=1
     row+=1
        
GMM_miss=[ found_2_G/nbr_2,found_6_G/nbr_6]    



