#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:06:34 2022

@author: muyeedahmed
"""

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
    
    
    labelFile_sk = filename + "_" + str(param_sk[0][1]) + "_" + str(param_sk[1][1]) + "_" + str(param_sk[2][1]) + "_" + str(param_sk[3][1])
    labelFile_mat = filename + "_" + str(param_mat[0][1]) + "_" + str(param_mat[1][1]) + "_" + str(param_mat[2][1]) + "_" + str(param_mat[3][1]) + "_" + str(param_mat[4][1]) + "_" + str(param_mat[5][1]) + "_" + str(param_mat[6][1]) + "_" + str(param_mat[7][1]) + "_" + str(param_mat[8][1])

    if os.path.exists("../AnomalyAlgoDiagnosis_Labels/EE_Sk/Labels_Sk_EE_"+labelFile_sk+".csv") == 0:
        # print(labelFile_sk)
        return 0

    labels_sk =  pd.read_csv("../AnomalyAlgoDiagnosis_Labels/EE_Sk/Labels_Sk_EE_"+labelFile_sk+".csv", header=None).to_numpy()


    labels_mat =  pd.read_csv("EE_Matlab/Labels_Mat_EE_"+labelFile_mat+".csv", header=None).to_numpy()
    
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
    
    store_precision = [True, False]
    assume_centered = [True, False]
    support_fraction = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    contamination = [0.1, 0.2, 0.3, 0.4, 0.5, 'LOF', 'IF']
    
    parameters_sk.append(["store_precision", True, store_precision])
    parameters_sk.append(["assume_centered", False, assume_centered])
    parameters_sk.append(["support_fraction", None, support_fraction])
    parameters_sk.append(["contamination", 0.1, contamination])
    
    parameters_mat = []
    
    Method = ["fmcd", "ogk", "olivehawkins"];
    OutlierFraction = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5];
    NumTrials = [2, 5, 10, 25, 50, 100, 250, 500, 750, 1000];
    BiasCorrection = [1, 0];
    NumOGKIterations = [1, 2, 3];
    UnivariateEstimator = ["tauscale", "qn"];
    ReweightingMethod = ["rfch", "rmvn"];
    NumConcentrationSteps = [2, 5, 10, 15, 20];
    StartMethod = ["classical", "medianball", "elemental"];    
    
    parameters_mat.append(["Method", "fmcd", Method])
    parameters_mat.append(["OutlierFraction", 0.5, OutlierFraction])
    parameters_mat.append(["NumTrials", 500, NumTrials])
    parameters_mat.append(["BiasCorrection", 1, BiasCorrection])
    parameters_mat.append(["NumOGKIterations", 2, NumOGKIterations])
    parameters_mat.append(["UnivariateEstimator", "tauscale", UnivariateEstimator])
    parameters_mat.append(["ReweightingMethod", "rfch", ReweightingMethod])
    parameters_mat.append(["NumConcentrationSteps", 10, NumConcentrationSteps])
    parameters_mat.append(["StartMethod", "classical", StartMethod])
    
    df = pd.DataFrame(columns = ['Filename', 'Configuration', 'Mean ARI', 'Min ARI'])
    
    
    for file in master_files:
        ari = get_ari(file, parameters_sk, parameters_mat)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Default", 
                        'Mean ARI' : ari_mean, 
                        'Min ARI' : ari_min}, ignore_index=True)
        
    
    
    parameters_mat[1][1] = 0.05
    for file in master_files:
        ari = get_ari(file, parameters_sk, parameters_mat)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 1", 
                        'Mean ARI' : ari_mean, 
                        'Min ARI' : ari_min}, ignore_index=True)
        
    
    parameters_sk[2][1] = 0.9
    for file in master_files:
        ari = get_ari(file, parameters_sk, parameters_mat)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 2", 
                        'Mean ARI' : ari_mean,
                        'Min ARI' : ari_min}, ignore_index=True)


    
    
    parameters_sk[2][1] = 0.6
    parameters_sk[3][1] = 0.2
    for file in master_files:
        ari = get_ari(file, parameters_sk, parameters_mat)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 3", 
                        'Mean ARI' : ari_mean,
                        'Min ARI' : ari_min}, ignore_index=True)

    ### Method: ogk, OutlierFraction = 0.5, UnivariateEstimator = qn
    parameters_mat[0][1] = 'ogk'
    parameters_mat[1][1] = 0.5
    parameters_mat[5][1] = 'qn'
    for file in master_files:
        ari = get_ari(file, parameters_sk, parameters_mat)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 4", 
                        'Mean ARI' : ari_mean,
                        'Min ARI' : ari_min}, ignore_index=True)

        
    fig = plt.Figure()
    axmean = sns.boxplot(x="Configuration", y="Mean ARI", data=df)
    axmean.set(xlabel=None)
    plt.title("Robust Covariance - Scikit-learn VS Matlab")
    plt.savefig("Fig/BoxPlot/EE_SkMat_MeanARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axmin = sns.boxplot(x="Configuration", y="Min ARI", data=df)
    axmin.set(xlabel=None)
    plt.title("Robust Covariance - Scikit-learn VS Matlab")
    plt.savefig("Fig/BoxPlot/EE_SkMat_MinARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    
    