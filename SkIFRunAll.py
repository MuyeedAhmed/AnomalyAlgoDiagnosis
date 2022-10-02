#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 20:44:46 2022

@author: muyeedahmed
"""

from datetime import datetime
import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
import collections
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn import metrics
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics.cluster import adjusted_rand_score

datasetFolderDir = '/Users/muyeedahmed/Desktop/Research/Dataset/Dataset_Anomaly/'
# datasetFolderDir = '/home/neamtiu/Desktop/ma234/AnomalyDetection/Dataset/'


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
    n_jobs = [1, -1] 
    warm_start = [True, False]
    
    # for ne in n_estimators:
    #     for ms in max_samples:
    #         for cont in contamination:    
    #             for mf in max_features:
    #                 for bs in bootstrap:
    #                     for nj in n_jobs:
    #                         for ws in warm_start:
    #                             runIF(filename, X, gt, ne, ms, cont, mf, bs, nj, ws)
    
    
    for ne in n_estimators:
        runIF(filename, X, gt, i_n_estimators=ne)
        
    for ms in max_samples:
        runIF(filename, X, gt, i_max_samples=ms)
    
    # for mf in max_features:
    #     runIF(filename, X, gt, i_max_features=mf)
    
    # for bs in bootstrap:
    #     runIF(filename, X, gt, i_bootstrap=bs)
    
    for nj in n_jobs:
        runIF(filename, X, gt, i_n_jobs=nj)
    
    # for ws in warm_start:
    #     runIF(filename, X, gt, i_warm_start=ws)
    
    
def runIF(filename, X, gt, i_n_estimators=100, i_max_samples='auto', i_contamination='auto', 
          i_max_features=1.0, i_bootstrap=False, i_n_jobs=None, i_warm_start=False):
    labelFile = filename + "_" + str(i_n_estimators) + "_" + str(i_max_samples) + "_" + str(i_contamination) + "_" + str(i_max_features) + "_" + str(i_bootstrap) + "_" + str(i_n_jobs) + "_" + str(i_warm_start)
    if os.path.exists("IF/Labels_"+labelFile+".csv") == 1:
        print("The Labels Already Exist")
        print("IF/Labels_"+labelFile+".csv")
        return
    
    for i in range(30):
        clustering = IsolationForest(n_estimators=i_n_estimators, max_samples=i_max_samples, 
                                     max_features=i_max_features, bootstrap=i_bootstrap, 
                                     n_jobs=i_n_jobs, warm_start=i_warm_start).fit(X)
    
        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        
        flabel=open("IF/Labels_"+labelFile+".csv", 'a')
        flabel.write(','.join(str(s) for s in l) + '\n')
        flabel.close()

def calculate_draw_score(allFiles, parameter, parameter_values):
    i_n_estimators=100
    i_max_samples='auto'
    i_contamination='auto' 
    i_max_features=1.0
    i_bootstrap=False
    i_n_jobs=None
    i_warm_start = False
    
    
    
    accuracy_range_all = []
    accuracy_med_all = []
    f1_range_all = []
    f1_med_all = []

    for filename in allFiles:
        print(filename)
        folderpath = datasetFolderDir
    
        if os.path.exists(folderpath+filename+".mat") == 0:
            print("File doesn't exist")
            return
        try:
            df = loadmat(folderpath+filename+".mat")
        except NotImplementedError:
            df = mat73.loadmat(folderpath+filename+".mat")

        gt=df["y"]
        gt = gt.reshape((len(gt)))
        X=df['X']
        if np.isnan(X).any():
            print("Didn\'t run -> NaN")
            continue
        for p in parameter_values:
            if parameter == 'n_estimators':
                i_n_estimators = p
            elif parameter == 'max_samples':
                i_max_samples = p
            elif parameter == 'max_features':
                i_max_features = p
            elif parameter == 'bootstrap':
                i_bootstrap = p
            elif parameter == 'n_jobs':
                i_n_jobs = p
            elif parameter == 'warm_start':
                i_warm_start = p
            labelFile = "IF/Labels_"+filename + "_" + str(i_n_estimators) + "_" + str(i_max_samples) + "_" + str(i_contamination) + "_" + str(i_max_features) + "_" + str(i_bootstrap) + "_" + str(i_n_jobs) + "_" + str(i_warm_start)
            labels = pd.read_csv(labelFile+'.csv', header=None).to_numpy()
    
            accuracy = [0] * 30
            f1 = [0] * 30
            ari = [0] * 30
            for run in range(30):
                l = labels[run]
                accuracy[run] = metrics.accuracy_score(gt, l)
                f1[run] = metrics.f1_score(gt, l)
                # ari[run] = adjusted_rand_score(gt, l)
                
            accDiff = np.percentile(accuracy, 75) - np.percentile(accuracy, 25)
            accuracy_range_all.append([p, accDiff])
            accuracy_med_all.append([p, np.percentile(accuracy, 50)])
            
            f1Diff = np.percentile(f1, 75) - np.percentile(f1, 25)
            f1_range_all.append([p, f1Diff])
            f1_med_all.append([p, np.percentile(f1, 50)])

    df_acc_r = pd.DataFrame(accuracy_range_all, columns = [parameter, 'Accuracy_Range'])
    df_acc_m = pd.DataFrame(accuracy_med_all, columns = [parameter, 'Accuracy_Median'])
    df_f1_r = pd.DataFrame(f1_range_all, columns = [parameter, 'F1Score_Range'])
    df_f1_m = pd.DataFrame(f1_med_all, columns = [parameter, 'F1Score_Median'])
    
    
    fig = plt.Figure()
    axf = sns.boxplot(x=parameter, y="Accuracy_Range", data=df_acc_r)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_"+parameter+"_Accuracy_Range.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axf = sns.boxplot(x=parameter, y="Accuracy_Median", data=df_acc_m)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_"+parameter+"_Accuracy_Median.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axf = sns.boxplot(x=parameter, y="F1Score_Range", data=df_f1_r)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_"+parameter+"_F1Score_Range.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axf = sns.boxplot(x=parameter, y="F1Score_Median", data=df_f1_m)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_"+parameter+"_F1Score_Median.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    



if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
    
    
    for FileNumber in range(len(master_files)):
        isolationforest(master_files[FileNumber])

    
    
    

