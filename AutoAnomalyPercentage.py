#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 16:51:54 2022

@author: muyeedahmed
"""



import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import random

datasetFolderDir = 'Dataset/'

def algo(filename):
    print(filename)
    folderpath = datasetFolderDir
    
    if os.path.exists(folderpath+filename+".mat") == 1:
        if os.path.getsize(folderpath+filename+".mat") > 200000: # 10MB
            print("Didn\'t run -> Too large - ", filename)    
            return
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
        if os.path.getsize(folderpath+filename+".csv") > 200000: # 10MB
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
    
    runALGO(filename, X, gt)
            
       
    
    
def runALGO(filename, X, gt):
    labels = []
    # _, counts_act = np.unique(gt, return_counts=True)
    # act = min(counts_act)/len(gt)
    
    # for i in range(10):
    #     clustering = IsolationForest().fit(X)
    
    #     l = clustering.predict(X)
    #     l = [0 if x == 1 else 1 for x in l]
    #     labels.append(l)
    # _, counts_if = np.unique(labels, return_counts=True)
    # if_per = min(counts_if)/(len(gt)*10)

    
    # labels_lof = LocalOutlierFactor().fit_predict(X)
    # _, counts_lof = np.unique(labels_lof, return_counts=True)
    # lof_per = min(counts_lof)/(len(gt))
    
    
    nu_s = random.uniform(0, .5)
    clustering = OneClassSVM(nu = nu_s).fit(X)
    labels_ocsvm = clustering.predict(X)
    _, counts_ocsvm = np.unique(labels_ocsvm, return_counts=True)
    ocsvm_per = min(counts_ocsvm)/(len(gt))
    breakpoint()
    print(nu_s, ocsvm_per)
    # if if_per == 1:
    #     if_per = 0
    # if lof_per == 1:
    #     lof_per = 0
    # if ocsvm_per == 1:
    #     ocsvm_per = 0
        
    # fp=open("Stats/SkPercentage.csv", 'a')
    # fp.write(filename+','+str(act)+','+str(if_per)+','+str(lof_per)+','+str(ocsvm_per)+'\n')
    # fp.close()


if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
    

    # fp=open("Stats/SkPercentage.csv", "w")
    # fp.write('Filename,Actual,IF,LOF,OCSVM\n')
    # fp.close()
        
    
    for FileNumber in range(len(master_files)):
        breakpoint()
        algo(master_files[FileNumber])
        
        
        
        
        
        
        