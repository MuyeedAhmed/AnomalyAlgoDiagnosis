#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 08:45:19 2023

@author: muyeedahmed
"""

import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn import metrics
from copy import copy, deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
# import pingouin as pg
import random
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
# import scikit_posthocs as sp
import scipy.stats as stats
from scipy.stats import gmean
import math
import time
import seaborn as sns
import matplotlib.pyplot as plt

datasetFolderDir = 'Dataset/'


def isolationforest(filename):
    # print(filename)
    # folderpath = datasetFolderDir
    
    # if os.path.exists(folderpath+filename+".mat") == 1:
    #     if os.path.getsize(folderpath+filename+".mat") > 200000: # 200KB
    #         # print("Didn\'t run -> Too large - ", filename)    
    #         return
    #     try:
    #         df = loadmat(folderpath+filename+".mat")
    #     except NotImplementedError:
    #         df = mat73.loadmat(folderpath+filename+".mat")

    #     gt=df["y"]
    #     gt = gt.reshape((len(gt)))
    #     X=df['X']
    #     if np.isnan(X).any():
    #         # print("Didn\'t run -> NaN - ", filename)
    #         return
        
    # elif os.path.exists(folderpath+filename+".csv") == 1:
    #     # if os.path.getsize(folderpath+filename+".csv") > 200000: # 200KB
    #     #     print("Didn\'t run -> Too large - ", filename)    
    #     #     return
    #     X = pd.read_csv(folderpath+filename+".csv")
    #     target=X["target"].to_numpy()
    #     print(X)
    #     X=X.drop("target", axis=1)
    #     gt = target
    #     if X.isna().any().any() == 1:
    #         print("Didn\'t run -> NaN value - ", filename)  
    #         return
    # else:
    #     print("File not found")
    #     return
    n_e = [50,100,150,200,250,300,350,400,450,500]
    
    # ari_all = []
    # t_all = []
    # for n in n_e:
    #     labels = []
    #     ari = []
    #     t = []
    #     for i in range(30):
    #         start = time.process_time()
    #         clustering = IsolationForest(n_estimators=n).fit(X)
    #         t.append(time.process_time() - start)
    #         l = clustering.predict(X)
    #         l = [0 if x == 1 else 1 for x in l]
    #         labels.append(l)


    
    #     for i in range(len(labels)):
    #         for j in range(i+1, len(labels)):
    #           ari.append(adjusted_rand_score(labels[i], labels[j]))
    #     ari_all.append(np.mean(ari))
    #     t_all.append(np.mean(t))
    # print(ari_all)
    # print(t_all)
    
    
    #matlab
    ari_all = [0.8338, 0.8814, 0.9009, 0.9164, 0.9278, 0.9297, 0.9371, 0.9413, 0.9433, 0.9471]
    t_all = [1.1644, 2.2677, 3.2687, 4.2743, 5.3597, 6.3944, 7.3752, 8.4535, 9.6213, 10.5779]
    
    fig,ax = plt.subplots()
    # ax.grid(False)

    # make a plot
    ax.plot(n_e, ari_all,
            color="red", 
            marker="o")
    # set x-axis label
    # ax.set_xlabel("n_estimators", fontsize = 12)
    ax.set_xlabel("NumLearners", fontsize = 12)
    # set y-axis label
    ax.set_ylabel("ARI",
                  color="red",
                  fontsize=12)
    ax2=ax.twinx()
    ax2.grid(False)

    # make a plot with different y-axis using second axis object
    ax2.plot(n_e, t_all,color="blue",marker="o")
    ax2.set_ylabel("Time (seconds)",color="blue",fontsize=12)
    plt.show()
    
    fig.savefig('Fig/n_e_graph_matlab.pdf', bbox_inches='tight')

    # plt.plot(n_e, ari_all)
    # plt.plot(n_e, t_all)
    
    
if __name__ == '__main__':
    isolationforest('yeast_ml8')