# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 21:17:31 2015

@author: james
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import copy


t1,t2,w = np.loadtxt('out.txt',unpack=True)
s = np.loadtxt('elo_score.txt')
med = np.loadtxt('elo_med.txt')

  
#  readcol,'2013_QB.csv', file,rank,name,team,pos,comp,att,pct,attG,$
#          yards,avg,yardsG,td,int,first,fpcnt,lng,l20,l40,sck,rate,$
#          f='(A,F,A,A,A,F)',delim=','

feat = pd.read_csv('2013_QB.csv', index_col=0, usecols=['comp','att','pct','attG','yards',
                                                        'avg','yardsG','td','int','first',
                                                        'fpcnt','lng','l20','l40','sck','rate'])


train_X = copy.copy(feat)
train_Y = copy.copy(med)
clf2 = RandomForestRegressor(n_estimators=5000, 
                            criterion='mse', max_depth=None, 
                            min_samples_split=2, min_samples_leaf=1, 
                            max_features='auto', max_leaf_nodes=None, 
                            bootstrap=True, oob_score=False, n_jobs=1, 
                            random_state=None, verbose=0, 
                            min_density=None, compute_importances=None)
                            
clf2.fit(train_X, train_Y)

Y_pred = clf2.predict(train_X)

plt.figure()

plt.scatter(train_Y, Y_pred - train_Y,c='cyan', marker='o')
plt.xlabel('Median Elo Score')
plt.ylabel('Predicted - Median Elo Score')
hlines(0,min(train_Y),max(train_Y),color="red")


plt.figure()
plt.scatter(train_Y, train_Y,c='cyan', marker='o')
plt.scatter(train_Y, Y_pred,c='red',marker='+')
plt.xlabel('Median Elo Score')
plt.ylabel('Predicted Elo Score')



plt.figure()
plt.bar(np.arange(len(clf2.feature_importances_)), clf2.feature_importances_)
plt.xlabel('Random Forest feature #')
plt.ylabel('Feature Importance')




