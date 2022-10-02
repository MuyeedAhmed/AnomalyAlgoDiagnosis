#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 20:44:46 2022

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

from sklearn.metrics.cluster import adjusted_rand_score

# datasetFolderDir = '/Users/muyeedahmed/Desktop/Research/Dataset/Dataset_Anomaly/'
# datasetFolderDir = '/home/neamtiu/Desktop/ma234/AnomalyDetection/Dataset/'
datasetFolderDir = '../Dataset_Combined/'


def isolationforest(filename):
    print(filename)
    folderpath = datasetFolderDir
    
    if os.path.exists(folderpath+filename+".mat") == 1:
        try:
            df = loadmat(folderpath+filename+".mat")
        except NotImplementedError:
            df = mat73.loadmat(folderpath+filename+".mat")

        gt=df["y"]
        gt = gt.reshape((len(gt)))
        X=df['X']
        if np.isnan(X).any():
            print("Didn\'t run -> NaN - ", filename)
            return
        
    elif os.path.exists(folderpath+filename+".csv") == 1:
        if os.path.getsize(folderpath+filename+".csv") > 1000000: # 10MB
            print("Didn\'t run -> Too large - ", filename)    
            return
        X = pd.read_csv(folderpath+filename+".csv")
        target=X["target"].to_numpy()
        X=X.drop("target", axis=1)
        gt = target
        if X.isna().any().any() == 1:
            print("Didn\'t run -> NaN value - ", filename)  
            return
    else:
        print("File doesn't exist")
        return

    n_estimators = [2, 4, 8, 16, 32, 64, 100, 128, 256, 512, 1024]
    max_samples = ['auto', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    contamination = ['auto'] 
    max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bootstrap = [True, False]
    n_jobs = [1, None] 
    warm_start = [True, False]
    
    print("n_estimators : ", end='')
    for ne in n_estimators:
        runIF(filename, X, gt, i_n_estimators=ne)
        print(ne, end = ', ')
    
    print("\nmax_samples : ", end='')
    for ms in max_samples:
        runIF(filename, X, gt, i_max_samples=ms)
        print(ms, end = ', ')
    
    print("\nmax_features : ", end='')
    for mf in max_features:
        runIF(filename, X, gt, i_max_features=mf)
        print(mf, end = ', ')
        
    print("\nbootstrap : ", end='')
    for bs in bootstrap:
        runIF(filename, X, gt, i_bootstrap=bs)
        print(bs, end = ', ')
        
    print("\nn_jobs : ", end='')
    for nj in n_jobs:
        runIF(filename, X, gt, i_n_jobs=nj)
        print(nj, end = ', ')
    
    print("\nwarm_start : ", end='')
    for ws in warm_start:
        runIF(filename, X, gt, i_warm_start=ws)
        print(ws, end = ', ')
    print()
    
def runIF(filename, X, gt, i_n_estimators=100, i_max_samples='auto', i_contamination='auto', 
          i_max_features=1.0, i_bootstrap=False, i_n_jobs=None, i_warm_start=False):
    labelFile = filename + "_" + str(i_n_estimators) + "_" + str(i_max_samples) + "_" + str(i_contamination) + "_" + str(i_max_features) + "_" + str(i_bootstrap) + "_" + str(i_n_jobs) + "_" + str(i_warm_start)
    if os.path.exists("IF/Labels_"+labelFile+".csv") == 1:
        # print("The Labels Already Exist")
        # print("IF/Labels_"+labelFile+".csv")
        return
    fstat_acc=open("Stats/SkIF_Accuracy.csv", "a")
    fstat_acc.write(filename+','+ str(i_n_estimators) + ','+ str(i_max_samples) + ',' + str(i_contamination) + ',' + str(i_max_features) + ',' + str(i_bootstrap) + ',' + str(i_n_jobs) + ',' + str(i_warm_start))
    
    fstat_f1=open("Stats/SkIF_F1.csv", "a")
    fstat_f1.write(filename+','+ str(i_n_estimators) + ','+ str(i_max_samples) + ',' + str(i_contamination) + ',' + str(i_max_features) + ',' + str(i_bootstrap) + ',' + str(i_n_jobs) + ',' + str(i_warm_start))
    
    fstat_ari=open("Stats/SkIF_ARI.csv", "a")
    fstat_ari.write(filename+','+ str(i_n_estimators) + ','+ str(i_max_samples) + ',' + str(i_contamination) + ',' + str(i_max_features) + ',' + str(i_bootstrap) + ',' + str(i_n_jobs) + ',' + str(i_warm_start))
    
    
    for i in range(30):
        clustering = IsolationForest(n_estimators=i_n_estimators, max_samples=i_max_samples, 
                                     max_features=i_max_features, bootstrap=i_bootstrap, 
                                     n_jobs=i_n_jobs, warm_start=i_warm_start).fit(X)
    
        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        
        flabel=open("IF/Labels_"+labelFile+".csv", 'a')
        flabel.write(','.join(str(s) for s in l) + '\n')
        flabel.close()
        
        accuracy = metrics.accuracy_score(gt, l)
        fstat_acc.write(','+str(accuracy))
        
        f1 = metrics.f1_score(gt, l)
        fstat_f1.write(','+str(f1))
        
        ari = adjusted_rand_score(gt, l)
        fstat_ari.write(','+str(ari))
        
     
    fstat_acc.write('\n')
    fstat_f1.write('\n')
    fstat_ari.write('\n')
    
    fstat_f1.close()
    fstat_acc.close()
    fstat_ari.close()



if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
    
    
    R = ""
    for i in range(29):
        R += "R"+str(i)+","
    R+="R29"
    if os.path.exists("Stats/SkIF_Accuracy.csv") == 0:
        fstat_acc=open("Stats/SkIF_Accuracy.csv", "w")
        fstat_acc.write('Filename,n_estimators,max_samples,contamination,max_features,bootstrap,n_jobs,warm_start,'+R+"\n")
        fstat_acc.close()
        
    if os.path.exists("Stats/SkIF_F1.csv") == 0: 
        fstat_f1=open("Stats/SkIF_F1.csv", "w")
        fstat_f1.write('Filename,n_estimators,max_samples,contamination,max_features,bootstrap,n_jobs,warm_start,'+R+"\n")
        fstat_f1.close()

    if os.path.exists("Stats/SkIF_ARI.csv") == 0:    
        fstat_ari=open("Stats/SkIF_ARI.csv", "w")
        fstat_ari.write('Filename,n_estimators,max_samples,contamination,max_features,bootstrap,n_jobs,warm_start,'+R+"\n")
        fstat_ari.close()

    
    for FileNumber in range(len(master_files)):
        print(FileNumber, end=' ')
        isolationforest(master_files[FileNumber])

    
    
    

