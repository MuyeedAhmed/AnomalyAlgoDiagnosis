#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 16:18:20 2022

@author: muyeedahmed
"""

import warnings
warnings.filterwarnings("ignore")
import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
from sklearn.svm import OneClassSVM
import numpy as np
from sklearn import metrics
from copy import copy, deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
import random
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
import scipy.stats as stats
from scipy.stats import gmean


datasetFolderDir = 'Dataset/'


def ocsvm(filename, parameters, parameter_iteration):
    print(filename)
    folderpath = datasetFolderDir
    
    if os.path.exists(folderpath+filename+".mat") == 1:
        if os.path.getsize(folderpath+filename+".mat") > 200000: 
            # print("Didn\'t run -> Too large - ", filename)    
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
        if os.path.getsize(folderpath+filename+".csv") > 200000:
            # print("Didn\'t run -> Too large - ", filename)    
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
    
    for p in range(len(parameters)):
        passing_param = deepcopy(parameters)
        if p == 1 and passing_param[0][1] != 'poly':
            continue
        if p == 2 and (passing_param[0][1] != 'rbf' and passing_param[0][1] != 'poly' and passing_param[0][1] != 'sigmoid'):
            continue
        if p == 3 and (passing_param[0][1] != 'poly' and passing_param[0][1] != 'sigmoid'):
            continue
        print(parameters[p][0], end=': ')
        for pv in range(len(parameters[p][2])):
            passing_param[p][1] = parameters[p][2][pv]
            runOCSVM(filename, X, gt, passing_param, parameter_iteration)
            print(parameters[p][2][pv], end = ', ')
        print()
       
    
    
def runOCSVM(filename, X, gt, params, parameter_iteration):
    labelFile = filename + "_" + str(params[0][1]) + "_" + str(params[1][1]) + "_" + str(params[2][1]) + "_" + str(params[3][1]) + "_" + str(params[4][1]) + "_" + str(params[5][1]) + "_" + str(params[6][1]) + "_" + str(params[7][1]) + "_" + str(params[8][1])
    if os.path.exists("OCSVM/Labels_Sk_OCSVM_"+labelFile+".csv") == 1:
        return
    
    labels = []
    f1 = []
    
    try:
        clustering = OneClassSVM(kernel=params[0][1], degree=params[1][1], gamma=params[2][1], coef0=params[3][1], tol=params[4][1], nu=params[5][1], 
                                     shrinking=params[6][1], cache_size=params[7][1], max_iter=params[8][1]).fit(X)
    except:
        return
    
    l = clustering.predict(X)
    l = [0 if x == 1 else 1 for x in l]
    labels.append(l)

    f1.append(metrics.f1_score(gt, l))
        
    if os.path.exists("../AnomalyAlgoDiagnosis_Labels/Labels_Sk_OCSVM_"+labelFile+".csv") == 0:
        fileLabels=open("../AnomalyAlgoDiagnosis_Labels/Labels_Sk_OCSVM_"+labelFile+".csv", 'a')
        for l in labels:
            fileLabels.write(','.join(str(s) for s in l) + '\n')
        fileLabels.close()
    
    flabel_done=open("OCSVM/Labels_Sk_OCSVM_"+labelFile+".csv", 'a')
    flabel_done.write("Done")
    flabel_done.close()
   
    fstat_f1=open("Stats/SkOCSVM_F1.csv", "a")
    fstat_f1.write(filename+','+ str(params[0][1]) + ','+ str(params[1][1]) + ',' + str(params[2][1]) + ',' + str(params[3][1]) + ',' + str(params[4][1]) + ',' + str(params[5][1]) + ',' + str(params[6][1]) + ',' + str(params[7][1]) + ',' + str(params[8][1]) + ',' + str(parameter_iteration) + ',')
    fstat_f1.write(','.join(str(s) for s in f1) + '\n')
    fstat_f1.close()
    

def calculate_score(allFiles, parameter, parameter_values, all_parameters):
    i_kernel=all_parameters[0][1]
    i_degree=all_parameters[1][1]
    i_gamma=all_parameters[2][1]
    i_coef0=all_parameters[3][1]
    i_tol=all_parameters[4][1]
    i_nu=all_parameters[5][1]
    i_shrinking=all_parameters[6][1]
    i_cache_size=all_parameters[7][1]
    i_max_iter=all_parameters[8][1]
    
    # dfacc = pd.read_csv("Stats/SkOCSVM_Accuracy.csv")
    dff1 = pd.read_csv("Stats/SkOCSVM_F1.csv")
    
    f1_runs = []
    for i in range(10):
        f1_runs.append(('R'+str(i)))


    f1_range_all = []
    f1_med_all = []
    ari_all = []
    
    for filename in allFiles:
        for p in parameter_values:
            if parameter == 'kernel':
                i_kernel = p
            elif parameter == 'degree':
                i_degree = p
            elif parameter == 'gamma':
                i_gamma = p
            elif parameter == 'coef0':
                i_coef0 = p
            elif parameter == 'tol':
                i_tol = p
            elif parameter == 'nu':
                i_nu = p
            elif parameter == 'shrinking':
                i_shrinking = p
            elif parameter == 'cache_size':
                i_cache_size = p
            elif parameter == 'max_iter':
                i_max_iter = p
            
            
            f1 = dff1[(dff1['Filename']==filename)&
                            (dff1['kernel']==i_kernel)&
                            (dff1['degree']==i_degree)&
                            (dff1['gamma']==i_gamma)&
                            (dff1['coef0']==i_coef0)&
                            (dff1['tol']==i_tol)&
                            (dff1['nu']==i_nu)&
                            (dff1['shrinking']==i_shrinking)&
                            (dff1['cache_size']==i_cache_size)&
                            (dff1['max_iter']==i_max_iter)]
            if f1.empty:
                continue
                       
            f1_values = f1["R"].to_numpy()[0]
            
            
            f1_med_all.append([filename, p, np.percentile(f1_values, 50)])
            
            

    df_f1_m = pd.DataFrame(f1_med_all, columns = ['Filename', parameter, 'F1Score_Median'])

    
    ### Mann–Whitney U test

    mwu_f1_range = [[0 for i in range(len(parameter_values))] for j in range(len(parameter_values))]
    i = 0
    for p1 in parameter_values:
        p1_values = df_f1_m[df_f1_m[parameter] == p1]['F1Score_Median'].to_numpy()
        j = 0
        for p2 in parameter_values:
            p2_values = df_f1_m[df_f1_m[parameter] == p2]['F1Score_Median'].to_numpy()
            if len(p1_values) == 0 or len(p2_values) == 0:
                mwu_f1_range[i][j] = None
                continue
            _, mwu_f1_range[i][j] = stats.mannwhitneyu(x=p1_values, y=p2_values, alternative = 'greater')
            j += 1
        i += 1
    mwu_df_f1_range = pd.DataFrame(mwu_f1_range, columns = parameter_values)
    mwu_df_f1_range.index = parameter_values
    mwu_df_f1_range.to_csv("Mann–Whitney U test/MWU_SkOCSVM_F1_Median_"+parameter+".csv")
    
    try:
        mwu_geomean = gmean(gmean(mwu_f1_range))
        mwu_min = np.min(mwu_f1_range)
    except:
        mwu_geomean = 11
        mwu_min = 11
    
    f1_m_grouped = df_f1_m.groupby(parameter)[["F1Score_Median"]].median().reset_index()
    
    parameter_value_max_f1_median = f1_m_grouped[parameter].loc[f1_m_grouped["F1Score_Median"].idxmax()]
    
    return mwu_geomean, mwu_min, parameter_value_max_f1_median, 0, 0



if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
    
    parameters = []
    
    kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    degree = [3, 4, 5, 6] # Kernel poly only
    gamma = ['scale', 'auto'] # Kernel ‘rbf’, ‘poly’ and ‘sigmoid’
    coef0 = [0.0, 0.1, 0.2, 0.3, 0.4] # Kernel ‘poly’ and ‘sigmoid’
    tol = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    nu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    shrinking = [True, False]
    cache_size = [50, 100, 200, 400]
    max_iter = [50, 100, 150, 200, 250, 300, -1]
    
    parameters.append(["kernel", 'rbf', kernel])
    parameters.append(["degree", 3, degree])
    parameters.append(["gamma", 'scale', gamma])
    parameters.append(["coef0", 0.0, coef0])
    parameters.append(["tol", 1e-3, tol])
    parameters.append(["nu", 0.5, nu])
    parameters.append(["shrinking", True, shrinking])
    parameters.append(["cache_size", 200, cache_size])
    parameters.append(["max_iter", -1, max_iter])
    
    
    
        
    if os.path.exists("Stats/SkOCSVM_F1.csv") == 0: 
        fstat_f1=open("Stats/SkOCSVM_F1.csv", "w")
        fstat_f1.write("Filename,kernel,degree,gamma,coef0,tol,nu,shrinking,cache_size,max_iter,Parameter_Iteration,R\n")
        fstat_f1.close()

    
    if os.path.exists("Stats/SkOCSVM_Winners.csv") == 0:  
        fstat_winner=open("Stats/SkOCSVM_Winners.csv", "w")
        fstat_winner.write('Parameter,Friedman,Max_F1,Min_F1_Range,Max_ARI\n')
        fstat_winner.close()
    for param_iteration in range(len(parameters)):
        for FileNumber in range(len(master_files)):
            print(FileNumber, end=' ')
            ocsvm(master_files[FileNumber], parameters, param_iteration)
            
            

        MWU_geo = [10]*len(parameters)
        MWU_min = [10]*len(parameters)
        f1_range = [0]*len(parameters)
        f1_median =[0]*len(parameters)
        ari = [0]*len(parameters)
        for i in range(len(parameters)):
            if len(parameters[i][2]) > 1:
                mwu_geomean, mwu_min, f1_median[i], f1_range[i], ari[i] = calculate_score(master_files, parameters[i][0], parameters[i][2], parameters)
                
                MWU_geo[i] = mwu_geomean
                MWU_min[i] = mwu_min
        index_min = np.argmin(MWU_geo)

        if MWU_min[index_min] > 1:
            print("MWU_min: ", end='')
            print(MWU_min)
            break
        parameters[index_min][1] = f1_median[index_min]
        parameters[index_min][2] = [f1_median[index_min]]
        
        fstat_winner=open("Stats/SkOCSVM_Winners.csv", "a")
        fstat_winner.write(parameters[index_min][0]+','+str(MWU_geo[index_min])+','+str(f1_median[index_min])+','+str(f1_range[index_min])+','+str(ari[index_min])+'\n')
        fstat_winner.close()
        
        print(parameters)              



        
        
        
        
        