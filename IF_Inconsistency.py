#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 03:11:38 2022

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

datasetFolderDir = 'Dataset/'


def get_ari(filename, param_sk, param_mat):
    labelFile_sk = filename + "_" + str(param_sk[0][1]) + "_" + str(param_sk[1][1]) + "_" + str(param_sk[2][1]) + "_" + str(param_sk[3][1]) + "_" + str(param_sk[4][1]) + "_" + str(param_sk[5][1]) + "_" + str(param_sk[6][1])
    labelFile_mat = filename + "_" + str(param_mat[0][1]) + "_" + str(param_mat[1][1]) + "_" + str(param_mat[2][1])

    if os.path.exists("../AnomalyAlgoDiagnosis_Labels/IF_Sk/Labels_Sk_IF_"+labelFile_sk+".csv") == 0:
        # print(labelFile_sk)
        return 0

    labels_sk =  pd.read_csv("../AnomalyAlgoDiagnosis_Labels/IF_Sk/Labels_Sk_IF_"+labelFile_sk+".csv", header=None).to_numpy()


    labels_mat =  pd.read_csv("IF_Matlab/Labels_Mat_IF_"+labelFile_mat+".csv", header=None).to_numpy()
    
    ari = []
    
    for i in range(len(labels_sk)):
        for j in range(len(labels_mat)):
            ari.append(adjusted_rand_score(labels_sk[i], labels_mat[j]))
            
    return ari


if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
    
    parameters_sk = []
    
    n_estimators = [2, 4, 8, 16, 32, 64, 100, 128, 256, 512]##
    max_samples = ['auto', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    contamination = ['auto'] 
    max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bootstrap = [True, False]
    n_jobs = [1, None] 
    warm_start = [True, False]
    
    parameters_sk.append(["n_estimators", 100, n_estimators])
    parameters_sk.append(["max_samples", 'auto', max_samples])
    parameters_sk.append(["contamination", 'auto', contamination])
    parameters_sk.append(["max_features", 1.0, max_features])
    parameters_sk.append(["bootstrap", False, bootstrap])
    parameters_sk.append(["n_jobs", None, n_jobs])
    parameters_sk.append(["warm_start", False, warm_start])
    
    parameters_mat = []
    
    ContaminationFraction = [0, 0.05, 0.1, 0.15, 0.2, 0.25, "LOF", "IF"];
    NumLearners = [1, 2, 4, 8, 16, 32, 64, 100, 128, 256, 512];
    NumObservationsPerLearner = ["auto", 0.05, 0.1, 0.2, 0.5, 1];
    
    parameters_mat.append(["ContaminationFraction", 0, ContaminationFraction])
    parameters_mat.append(["NumLearners", 100, NumLearners])
    parameters_mat.append(["NumObservationsPerLearner", 'auto', NumObservationsPerLearner])
    
    df = pd.DataFrame(columns = ['Filename', 'Configuration', 'Mean_ARI', 'Min_ARI'])
    
    # ari_all = []
    # ari_mean_all = []
    # ari_min_all = []
    for file in master_files:
        ari = get_ari(file, parameters_sk, parameters_mat)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        # ari_all.append(ari)
        # ari_mean_all.append(ari_mean)
        # ari_min_all.append(ari_min)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Default", 
                        'Mean_ARI' : ari_mean, 
                        'Min_ARI' : ari_min}, ignore_index=True)
        
    # ari_all_flat = [item for sublist in ari_all for item in sublist]
    
    
    parameters_mat[0][1] = 0.1
    for file in master_files:
        ari = get_ari(file, parameters_sk, parameters_mat)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 1", 
                        'Mean_ARI' : ari_mean, 
                        'Min_ARI' : ari_min}, ignore_index=True)
        
    parameters_mat[0][1] = "IF"
    for file in master_files:
        ari = get_ari(file, parameters_sk, parameters_mat)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 2", 
                        'Mean_ARI' : ari_mean,
                        'Min_ARI' : ari_min}, ignore_index=True)


    parameters_sk[0][1] = 512
    parameters_mat[1][1] = 512
    for file in master_files:
        ari = get_ari(file, parameters_sk, parameters_mat)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 3", 
                        'Mean_ARI' : ari_mean,
                        'Min_ARI' : ari_min}, ignore_index=True)
    
    fig = plt.Figure()
    axa = sns.boxplot(x="Configuration", y="Mean_ARI", data=df)
    plt.show()

    fig = plt.Figure()
    axa = sns.boxplot(x="Configuration", y="Min_ARI", data=df)
    plt.show()
