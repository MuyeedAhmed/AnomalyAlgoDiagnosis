#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 18:01:52 2022

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


def get_ari_sk_mat(filename, param_sk, param_mat):
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


def get_ari_sk_r(filename, param_sk, param_r):
    labelFile_sk = filename + "_" + str(param_sk[0][1]) + "_" + str(param_sk[1][1]) + "_" + str(param_sk[2][1]) + "_" + str(param_sk[3][1]) + "_" + str(param_sk[4][1]) + "_" + str(param_sk[5][1]) + "_" + str(param_sk[6][1])
    labelFile_r = filename + "_" + str(param_r[0][1]) + "_" + str(param_r[1][1]) + "_" + str(param_r[2][1]) + "_" + str(param_r[3][1]) 
    if os.path.exists("../AnomalyAlgoDiagnosis_Labels/IF_Sk/Labels_Sk_IF_"+labelFile_sk+".csv") == 0:
        # print(labelFile_sk)
        return 0
    

    labels_sk =  pd.read_csv("../AnomalyAlgoDiagnosis_Labels/IF_Sk/Labels_Sk_IF_"+labelFile_sk+".csv", header=None).to_numpy()    
    
    
    labels_r =  pd.read_csv("Labels/IF_R/"+labelFile_r+".csv").to_numpy()

    
    ari = []
    
    for i in range(len(labels_sk)):
        for j in range(len(labels_r)):
            ari.append(adjusted_rand_score(labels_sk[i], np.int64((labels_r[j][1:])*1)))
            
    return ari


def get_ari_r_mat(filename, param_r, param_mat):
    labelFile_r = filename + "_" + str(param_r[0][1]) + "_" + str(param_r[1][1]) + "_" + str(param_r[2][1]) + "_" + str(param_r[3][1])
    labelFile_mat = filename + "_" + str(param_mat[0][1]) + "_" + str(param_mat[1][1]) + "_" + str(param_mat[2][1])

    if os.path.exists("IF_Matlab/Labels_Mat_IF_"+labelFile_mat+".csv") == 0:
        # print(labelFile_sk)
        return 0
    if os.path.exists("Labels/IF_R/"+labelFile_r+".csv") == 0:
        # print(labelFile_sk)
        return 0

    labels_r = pd.read_csv("Labels/IF_R/"+labelFile_r+".csv").to_numpy()
    # labels_r = np.int64((labels_r[0][1:])*1)
    # print(labels_r)
    labels_mat =  pd.read_csv("IF_Matlab/Labels_Mat_IF_"+labelFile_mat+".csv", header=None).to_numpy()
    # print(np.int64((labels_r[0][1:])*1))
    ari = []
    
    for i in range(len(labels_mat)):
        for j in range(len(labels_r)):
            ari.append(adjusted_rand_score(labels_mat[i], np.int64((labels_r[j][1:])*1)) )
            
    return ari


    
def run_all():
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
    
    ## Sk
    parameters_sk = []
    
    n_estimators = [2, 4, 8, 16, 32, 64, 100, 128, 256, 512]##
    max_samples = ['auto', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    contamination = ['auto']
    max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bootstrap = [True, False]
    n_jobs = [1, None] 
    warm_start = [True, False]
    
    parameters_sk.append(["n_estimators", 100, n_estimators, "Sk", 0])
    parameters_sk.append(["max_samples", 'auto', max_samples, "Sk", 1])
    parameters_sk.append(["contamination", 'auto', contamination, "Sk", 2])
    parameters_sk.append(["max_features", 1.0, max_features, "Sk", 3])
    parameters_sk.append(["bootstrap", False, bootstrap, "Sk", 4])
    parameters_sk.append(["n_jobs", None, n_jobs, "Sk", 5])
    parameters_sk.append(["warm_start", False, warm_start, "Sk", 6])
    
    ## R
    parameters_r = []
    
    ntrees = [2, 4, 8, 16, 32, 64, 100, 128, 256, 512]
    standardize_data = ["TRUE","FALSE"]
    sample_size = ['auto',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,"NULL"]
    ncols_per_tree = ['def',0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    parameters_r.append(["ntrees",512,ntrees])
    parameters_r.append(["standardize_data","TRUE",standardize_data, "R"])
    parameters_r.append(["sample_size",'auto',sample_size, "R"])
    parameters_r.append(["ncols_per_tree",'def',ncols_per_tree, "R"])
    
    ## Mat
    parameters_mat = []
    
    ContaminationFraction = [0, 0.05, 0.1, 0.15, 0.2, 0.25, "LOF", "IF"];
    NumLearners = [1, 2, 4, 8, 16, 32, 64, 100, 128, 256, 512];
    NumObservationsPerLearner = ["auto", 0.05, 0.1, 0.2, 0.5, 1];
    
    parameters_mat.append(["ContaminationFraction", 0, ContaminationFraction, "mat"])
    parameters_mat.append(["NumLearners", 100, NumLearners, "mat"])
    parameters_mat.append(["NumObservationsPerLearner", 'auto', NumObservationsPerLearner, "mat"])
    
    ## Merge All
    
    parameters = parameters_mat + parameters_sk + parameters_r
    print(parameters[8])
    ##
    # df = pd.DataFrame(columns = ['Filename', 'Configuration', 'Mean ARI', 'Min ARI'])
    
    
    # for file in master_files:
    #     ari = get_ari_r_mat(file, parameters_r, parameters_mat)
    #     if ari == 0:
    #         continue
    #     ari_mean = np.mean(ari)
    #     ari_min = np.min(ari)
        
    #     df = df.append({'Filename' : file,
    #                     'Configuration' : "Default", 
    #                     'Mean ARI' : ari_mean, 
    #                     'Min ARI' : ari_min}, ignore_index=True)
    

if __name__ == '__main__':
    run_all()
