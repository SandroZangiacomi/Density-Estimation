# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 22:54:21 2020

@author: Zangiacomi Sandro
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m
import scipy
from sklearn.neighbors import KernelDensity
from mpl_toolkits import mplot3d
##########EXERCICE 1 ############

#a)
n90pol=pd.DataFrame(pd.read_csv('data/n90pol.csv'))

amygdala=n90pol['amygdala']
acc=n90pol['acc']


plt.figure()
hist_amy=amygdala.hist(bins=10)
plt.title(' 1-dimensional histogram amygdala')
#plt.show()

plt.figure()
hist_acc=acc.hist(bins=10)
plt.title(' 1-dimensional histogram acc')
#plt.show()

plt.figure()
KDE_amy =amygdala.plot.kde()
plt.title('1-dimensional KDE amygdala' )
#plt.show()
          
plt.figure()
plt.title('1-dimensional KDE acc' )
KDE_acc =acc.plot.kde()
#plt.show()

#b)

nb_bins_squareroot=int(m.sqrt(len(n90pol['acc'].values.tolist())))    #=9
nb_bins_ricerule=int(2*(len(n90pol['acc'].values.tolist()))**(1/3))   #=8

plt.figure()
plt.hist2d(amygdala,acc,cmap=plt.cm.Reds,bins=nb_bins_squareroot)
plt.xlabel('Amygdala')
plt.ylabel('acc')
plt.title('2-dimensions Histogram Amygdala and Acc with Sqare-root' )
plt.colorbar()
plt.figure()
plt.hist2d(amygdala,acc,cmap=plt.cm.Reds,bins=nb_bins_ricerule)
plt.xlabel('Amygdala')
plt.ylabel('acc')
plt.title('2-dimensions Histogram Amygdala and Acc with Rice Rule' )
plt.colorbar()

#plt.show()
def KDE2D(band_w,Data):
    x=np.asarray(Data.iloc[:,0]).T
    y=np.asarray(Data.iloc[:,1]).T
    xy = np.vstack([x,y])
    fig,ax=plt.subplots()
    kde = KernelDensity(bandwidth=band_w, metric='euclidean',kernel='gaussian', algorithm='ball_tree').fit(xy.T)
    X, Y = np.mgrid[min(x):max(x):1000j, min(y):max(y):1000j]
    pos = np.vstack([X.ravel(), Y.ravel()])
    A = np.reshape(np.exp(kde.score_samples(pos.T)), X.shape)
    ax=fig.gca(projection='3d')
    ax.view_init(elev=30,azim=30)
    ax.plot_surface(X,Y,A,cmap=plt.cm.viridis)
  
    plt.xlabel('Amygdala')
    plt.ylabel('acc')
    plt.title('Amyg_acc KDE2D')

    return A
    

amyg_acc2D=KDE2D(0.014,n90pol)
    

standard_dev=n90pol.iloc[:,:2].std(axis=0)
h_silver=1.06*standard_dev*(90)**(-1/5)                          #[	amygdala_hsilver = 0.01405227080724824,	acc_hsilver	0.008807245541800542]
h_scott=3.49*standard_dev/(90**(1/3))                        #[	amygdala_hscott = 0.0253922,	acc_hsilver	0.0159145]


#c)
def KDE1D(band_w,Data_col,name):
    x=np.asarray(Data_col).reshape(-1, 1)
    fig,ax=plt.subplots()
    kde = KernelDensity(bandwidth=band_w, metric='euclidean',kernel='gaussian',
                        algorithm='ball_tree').fit(x)
    X=np.linspace(min(x),max(x),1000)
    A = kde.score_samples(X)
    ax.plot(X,np.exp(A))
    plt.title('{}'.format(name))
    return A

amyg1D=KDE1D(0.014,n90pol.iloc[:,0],'Amyg KDE')
acc1D=KDE1D(0.0088,n90pol.iloc[:,1],'ACC KDE')


def independant(X,Y,XY):
    X=np.asarray(X)
    Y=np.asarray(Y)
    X_Y=np.zeros((len(X),len(X)))
    for i in range (0,len(X)):
        for j in range(0,len(X)):
            X_Y[i][j]=X[i]*X[j]
    Inde=abs(np.asarray(XY)-X_Y)

    plt.figure()
    plt.imshow(np.rot90(X_Y))
    plt.colorbar()
    plt.title('p(amygdala)p(acc)')
    
    plt.figure()
    plt.imshow(np.rot90(XY))
    plt.colorbar()
    plt.title('p(amygdala,acc)')
    
    plt.figure()
    fig1,ax_2=plt.subplots()
    plt.imshow(np.rot90(Inde))
    plt.colorbar()
    plt.title('p(amygdala,acc) = p(amygdala)p(acc)? ')
    return Inde    
        
Inde=independant(amyg1D,acc1D,amyg_acc2D)
#d)
def Orient_Condi_prob(Data_set,orientation,column):  
    prob=[]
    if column == 'acc':
        Col=np.asarray(Data_set.iloc[:,1])
        count=0
        for orient in np.asarray(Data_set.iloc[:,2]):
            if orient==orientation:
                prob.append(Col[count])
                count+=1
                
    elif column == 'amyg':
        Col=np.asarray(Data_set.iloc[:,1])
        count=0
        for orient in np.asarray(Data_set.iloc[:,2]):
            if orient==orientation:
                    prob.append(Col[count])
                    count+=1
    a=KDE1D(0.014,prob,'Probability of ' +column + 
            ' conditioned  on orientation = {}'.format(orientation))
    return a

Orient_Condi_prob(n90pol,2,'acc')
Orient_Condi_prob(n90pol,3,'acc')
Orient_Condi_prob(n90pol,4,'acc')
Orient_Condi_prob(n90pol,5,'acc')

Orient_Condi_prob(n90pol,2,'amyg')
Orient_Condi_prob(n90pol,3,'amyg')
Orient_Condi_prob(n90pol,4,'amyg')
Orient_Condi_prob(n90pol,5,'amyg')

#e)

def KDE2D_orientation(band_w,Data,orientation):
    count=0
    x=[]
    y=[]
    for orient in Data.iloc[:,2]:
        if orient==orientation:
            x.append(Data.iloc[count,0])
            y.append(Data.iloc[count,1])
        count+=1
    x=np.asarray(x)
    y=np.asarray(y)
    xy = np.vstack([x,y])
    fig,ax=plt.subplots()
    kde = KernelDensity(bandwidth=band_w, metric='euclidean',
                        kernel='gaussian', 
                        algorithm='ball_tree').fit(xy.T)
    X, Y = np.mgrid[min(x):max(x):1000j, min(y):max(y):1000j]
    pos = np.vstack([X.ravel(), Y.ravel()])
    A = np.reshape(np.exp(kde.score_samples(pos.T)), X.shape)
    ax=fig.gca(projection='3d')
    ax.view_init(elev=30,azim=30)
    ax.plot_surface(X,Y,A,cmap=plt.cm.viridis)
  
    plt.xlabel('Amygdala')
    plt.ylabel('acc')
    plt.title('Amyg_acc KDE2D'+ ' conditioned on orientation {}'
              .format(orientation))

    return A

KDE2D_orientation(0.014,n90pol,2)
KDE2D_orientation(0.014,n90pol,3)
KDE2D_orientation(0.014,n90pol,4)
KDE2D_orientation(0.014,n90pol,5)

#f)


def independant_conditionned(n90pol,XY,X,Y,orientation):
    X_Y=np.zeros((len(X),len(X)))
    for i in range (0,len(X)):
        for j in range(0,len(X)):
            X_Y[i][j]=X[i]*X[j]
    Inde=abs(np.asarray(XY)-X_Y)
    fig,ax_1=plt.subplots()
   
    im=ax_1.imshow(np.rot90(Inde),
                   extent=[min(X), max(X), min(Y), max(Y)])
    fig.colorbar(im)
    plt.title('p(amygdala,accc|orientation = {})= p(amygdala|orientation = {})p(acc|orientation = {})? '.format(orientation,orientation,orientation))
    return Inde 


O2D_2=KDE2D_orientation(0.014,n90pol,2)
O2D_3=KDE2D_orientation(0.014,n90pol,3)
O2D_4=KDE2D_orientation(0.014,n90pol,4)
O2D_5=KDE2D_orientation(0.014,n90pol,5)

amyg_2=Orient_Condi_prob(n90pol,2,'amyg')
amyg_3=Orient_Condi_prob(n90pol,3,'amyg')
amyg_4=Orient_Condi_prob(n90pol,4,'amyg')
amyg_5=Orient_Condi_prob(n90pol,5,'amyg')

acc_2=Orient_Condi_prob(n90pol,2,'acc')
acc_3=Orient_Condi_prob(n90pol,3,'acc')
acc_4=Orient_Condi_prob(n90pol,4,'acc')
acc_5=Orient_Condi_prob(n90pol,5,'acc')

independant_conditionned(n90pol,O2D_2,amyg_2,acc_2,2)
independant_conditionned(n90pol,O2D_3,amyg_3,acc_3,3)
independant_conditionned(n90pol,O2D_4,amyg_4,acc_4,4)
independant_conditionned(n90pol,O2D_5,amyg_5,acc_5,5)






