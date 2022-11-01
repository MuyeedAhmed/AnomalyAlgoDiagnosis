#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 20:11:05 2022

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
import scipy.stats as stats
from scipy.stats import gmean
import math

datasetFolderDir = 'Dataset/'

    

def calculate_score(allFiles, parameter, parameter_values, all_parameters):
    i_n_estimators=all_parameters[0][1]
    i_max_samples=all_parameters[1][1]
    i_contamination=all_parameters[2][1]
    i_max_features=all_parameters[3][1]
    i_bootstrap=all_parameters[4][1]
    i_n_jobs=all_parameters[5][1]
    i_warm_start = all_parameters[6][1]
    
    # dfacc = pd.read_csv("Stats/SkIF_Accuracy.csv")
    dff1 = pd.read_csv("Stats/SkIF_F1.csv")
    dfari =  pd.read_csv("Stats/SkIF_ARI.csv")
    
    f1_runs = []
    for i in range(10):
        f1_runs.append(('R'+str(i)))

    ari_runs = []
    for i in range(45):
        ari_runs.append(('R'+str(i)))

    # accuracy_range_all = []
    # accuracy_med_all = []
    f1_range_all = []
    f1_med_all = []
    ari_all = []
    
    for filename in allFiles:
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
                

            f1 = dff1[(dff1['Filename']==filename)&
                            (dff1['n_estimators']==i_n_estimators)&
                            (dff1['max_samples']==str(i_max_samples))&
                            (dff1['max_features']==i_max_features)&
                            (dff1['bootstrap']==i_bootstrap)&
                            (dff1['n_jobs']==str(i_n_jobs))&
                            (dff1['warm_start']==i_warm_start)]
            if f1.empty:
                continue
            ari = dfari[(dfari['Filename']==filename)&
                            (dfari['n_estimators']==i_n_estimators)&
                            (dfari['max_samples']==str(i_max_samples))&
                            (dfari['max_features']==i_max_features)&
                            (dfari['bootstrap']==i_bootstrap)&
                            (dfari['n_jobs']==str(i_n_jobs))&
                            (dfari['warm_start']==i_warm_start)]
            
            
            f1_values = f1[f1_runs].to_numpy()[0]
            ari_values = ari[ari_runs].to_numpy()[0]
            
            f1_med_all.append([filename, p, np.percentile(f1_values, 50)])            
            ari_all.append([filename, p, np.percentile(ari_values, 50)])
            
    df_f1_m = pd.DataFrame(f1_med_all, columns = ['Filename', parameter, 'F1Score_Median'])
    df_ari_m = pd.DataFrame(ari_all, columns = ['Filename', parameter, 'ARI'])

    if parameter == 'n_jobs':
        df_f1_m = df_f1_m.fillna(0)
        df_ari_m = df_ari_m.fillna(0)
        df_f1_m[parameter] = df_f1_m[parameter].astype(int)
        df_ari_m[parameter] = df_ari_m[parameter].astype(int)
        parameter_values = [1, 0]

   
    ### Mann–Whitney U test
   
    mwu_ari = [[0 for i in range(len(parameter_values))] for j in range(len(parameter_values))]
    i = 0
    for p1 in parameter_values:
        p1_values = df_ari_m[df_ari_m[parameter] == p1]['ARI'].to_numpy()
        j = 0
        for p2 in parameter_values:
            p2_values = df_ari_m[df_ari_m[parameter] == p2]['ARI'].to_numpy()
            if len(p1_values) == 0 or len(p2_values) == 0:
                mwu_ari[i][j] = None
                continue
            _, mwu_ari[i][j] = stats.mannwhitneyu(x=p1_values, y=p2_values)
            j += 1
        i += 1
    mwu_df_ari = pd.DataFrame(mwu_ari, columns = parameter_values)
    mwu_df_ari.index = parameter_values
    mwu_df_ari.to_csv("Mann–Whitney U test/MWU_SkIF_ARI_"+parameter+".csv")
    
    
    try:
        mwu_min = np.min(mwu_ari)
        mwu_ari = [[z+1 for z in y] for y in mwu_ari]
        mwu_geomean = gmean(gmean(mwu_ari)) - 1
        
    except:
        mwu_geomean = 11
        mwu_min = 11
    
    f1_m_grouped = df_f1_m.groupby(parameter)[["F1Score_Median"]].median().reset_index()
    ari_m_grouped = df_ari_m.groupby(parameter)[["ARI"]].median().reset_index()
    
    parameter_value_max_f1_median = f1_m_grouped[parameter].loc[f1_m_grouped["F1Score_Median"].idxmax()]
    parameter_value_max_ari = ari_m_grouped[parameter].loc[ari_m_grouped["ARI"].idxmax()]
    
    return mwu_geomean, mwu_min
  

# def lofrun(filename, parameters, parameter_iteration, parameter_rankings):
#     print(filename)
#     folderpath = datasetFolderDir
    
#     parameters_this_file = deepcopy(parameters)
    
#     if os.path.exists(folderpath+filename+".mat") == 1:
#         if os.path.getsize(folderpath+filename+".mat") > 200000: # 200KB
#             # print("Didn\'t run -> Too large - ", filename)    
#             return
#         try:
#             df = loadmat(folderpath+filename+".mat")
#         except NotImplementedError:
#             df = mat73.loadmat(folderpath+filename+".mat")

#         gt=df["y"]
#         gt = gt.reshape((len(gt)))
#         X=df['X']
#         if np.isnan(X).any():
#             # print("Didn\'t run -> NaN - ", filename)
#             return
        
#     elif os.path.exists(folderpath+filename+".csv") == 1:
#         if os.path.getsize(folderpath+filename+".csv") > 200000: # 200KB
#             print("Didn\'t run -> Too large - ", filename)    
#             return
#         X = pd.read_csv(folderpath+filename+".csv")
#         target=X["target"].to_numpy()
#         X=X.drop("target", axis=1)
#         gt = target
#         if X.isna().any().any() == 1:
#             print("Didn\'t run -> NaN value - ", filename)  
#             return
#     else:
#         print("File doesn't exist")
#         return
#     a , b = LOF(X, 20)
#     print(a, b)

if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
    
    
    # parameters = []
    
    # n_estimators = [2, 4, 8, 16, 32, 64, 100, 128, 256, 512]##
    # max_samples = ['auto', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # contamination = ['auto'] 
    # max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # bootstrap = [True, False]
    # n_jobs = [1, None] 
    # warm_start = [True, False]
    
    # parameters.append(["n_estimators", 100, n_estimators])
    # parameters.append(["max_samples", 'auto', max_samples])
    # parameters.append(["contamination", 'auto', contamination])
    # parameters.append(["max_features", 1.0, max_features])
    # parameters.append(["bootstrap", False, bootstrap])
    # parameters.append(["n_jobs", None, n_jobs])
    # parameters.append(["warm_start", False, warm_start])
    

    
    # MWU_geo = [10]*len(parameters)
    # MWU_min = [10]*len(parameters)
    # MWU_Store = []
    # f1_median =[0]*len(parameters) 
    # ari = [0]*len(parameters)
    # for i in range(len(parameters)):
    #     if len(parameters[i][2]) > 1:
    #         mwu_geomean, mwu_min = calculate_score(master_files, parameters[i][0], parameters[i][2], parameters)
            
    #         MWU_geo[i] = mwu_geomean
    #         MWU_min[i] = mwu_min
    #         MWU_Store.append([parameters[i][0], mwu_geomean, mwu_min])
            
    # MWU_Store = pd.DataFrame(MWU_Store, columns=["Parameter", "GeoMean", "Min"])
    # MWU_Store["Rank"] = MWU_Store["GeoMean"].rank()
    # print(MWU_Store)
    # MWU_Store.to_csv("Stats/SkIF_MWU_Rank.csv")

    data = pd.read_csv("Stats/SkIF_Grouped_Median.csv")
    data["bootstrap"] = data["bootstrap"].astype(int)
    data["warm_start"] = data["warm_start"].astype(int)
    
    y = data['ARI_Median']
    
    x = data[['n_estimators']]
    model = sm.OLS(y, x).fit()
    print('n_estimators', model.aic)
    
    x = data[['max_samples']]
    model = sm.OLS(y, x).fit()
    print('max_samples', model.aic)
        
    x = data[['max_features']]
    model = sm.OLS(y, x).fit()
    print('max_features', model.aic)
    
    x = data[['bootstrap']]
    model = sm.OLS(y, x).fit()
    print('bootstrap', model.aic)
        
    x = data[['n_jobs']]
    model = sm.OLS(y, x).fit()
    print('n_jobs', model.aic)
        
    x = data[['warm_start']]
    model = sm.OLS(y, x).fit()
    print('warm_start', model.aic)
        
    x = data[['ARI_Median']]
    model = sm.OLS(y, x).fit()
    print('ARI_Median', model.aic)
        
        
        