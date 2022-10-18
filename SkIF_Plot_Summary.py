#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 02:42:43 2022

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



def plot_ari_f1():
    df_f1 = pd.read_csv("Stats/SkIF_F1.csv")
    df_ari = pd.read_csv("Stats/SkIF_ARI.csv")
    
    runs_r = []
    runs_ari = []
    
    for i in range(10):
        runs_r.append(('R'+str(i)))
    for i in range(45):
        runs_ari.append(('R'+str(i)))

    df_f1["F1_Median"] = 0
    df_f1["F1_Range"] = 0
    df_ari["ARI_Median"] = 0
    
    for i in range(df_f1.shape[0]):
        run_values = df_f1.loc[i, runs_r].tolist()
        
        range_ = (np.percentile(run_values, 75) - np.percentile(run_values, 25))/(np.percentile(run_values, 75) + np.percentile(run_values, 25))
        if math.isnan(range_):
            range_ = 0
            
        df_f1.iloc[i, df_f1.columns.get_loc('F1_Median')] =  np.mean(run_values)
        df_f1.iloc[i, df_f1.columns.get_loc('F1_Range')] = range_
    df_f1 = df_f1.drop(columns=runs_r, axis=1)
        
    for i in range(df_ari.shape[0]):
        run_values = df_ari.loc[i, runs_ari].tolist()
        df_ari.iloc[i, df_ari.columns.get_loc('ARI_Median')] =  np.mean(run_values)
    df_ari= df_ari.drop(columns=runs_ari, axis=1)
    
    
    parameter_names = ["n_estimators", "max_samples", "contamination", "max_features", "bootstrap", "n_jobs", "warm_start", "Parameter_Iteration"]
    join_on =["Filename", "n_estimators", "max_samples", "contamination", "max_features", "bootstrap", "n_jobs", "warm_start", "Parameter_Iteration"]

    df_all = pd.merge(df_f1, df_ari,  how='left', left_on=join_on, right_on =join_on)
    
    df_all.to_csv("Stats/SkIF_Merged.csv")
    
    median_df = df_all.groupby(parameter_names)[["F1_Median", "F1_Range", "ARI_Median"]].mean()
    median_df = median_df.reset_index()
    
    median_df.to_csv("Stats/SkIF_Grouped_Median.csv")
    
    
    print(median_df.iloc[median_df["Performance"].idxmax()])
    print(median_df.iloc[median_df["Nondeterminism"].idxmin()])
    
    '''
    Plot Group Summary
    '''
    i_n_estimators=100
    i_max_samples='auto'
    i_contamination='auto' 
    i_max_features=1.0
    i_bootstrap=False
    i_n_jobs="None"
    i_warm_start = False
    default_run = median_df[(median_df['n_estimators']==i_n_estimators)&
                            (median_df['max_samples']==str(i_max_samples))&
                            (median_df['max_features']==i_max_features)&
                            (median_df['bootstrap']==i_bootstrap)&
                            (median_df['n_jobs']==str(i_n_jobs))&
                            (median_df['warm_start']==i_warm_start)]

       
    default_performance = default_run['F1_Median'].values[0]
    default_nondeter_range = default_run['F1_Range'].values[0]
    default_nondeter_ari = default_run['ARI_Median'].values[0]
    
    performance = median_df["F1_Median"].values    
    nondeterminism_range = median_df["F1_Range"].values
    nondeterminism_ari = median_df["ARI_Median"].values

    fig = plt.Figure()
    plt.plot(nondeterminism_range, performance, ".")
    plt.plot(default_nondeter_range, default_performance, "o")
    plt.title("Scikit-learn - Isolation Forest")
    plt.xlabel("Nondeterminism (F1 Score Range)")
    plt.ylabel("Performance (F1 Score)")
    plt.savefig("Fig/SkIF_F1_Range.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

    fig = plt.Figure()
    plt.plot(nondeterminism_ari, performance, ".")
    plt.plot(default_nondeter_ari, default_performance, "o")
    plt.title("Scikit-learn - Isolation Forest")
    plt.xlabel("Determinism (ARI)")
    plt.ylabel("Performance (F1 Score)")
    plt.savefig("Fig/SkIF_F1_ARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    
    
    '''
    Plot For All Dataset
    '''
    ## Default
    default_run = df_all[(df_all['n_estimators']==i_n_estimators)&
                        (df_all['max_samples']==str(i_max_samples))&
                        (df_all['max_features']==i_max_features)&
                        (df_all['bootstrap']==i_bootstrap)&
                        (df_all['n_jobs']==str(i_n_jobs))&
                        (df_all['warm_start']==i_warm_start)]
    default_performance = default_run['F1_Median'].values
    default_nondeter = default_run['ARI_Median'].values
    
    ## Settings 1
    settings1_run = df_all[(df_all['n_estimators']==512)&
                        (df_all['max_samples']==str(1.0))&
                        (df_all['max_features']==0.7)&
                        (df_all['bootstrap']==False)&
                        (df_all['n_jobs']==str(None))&
                        (df_all['warm_start']==False)]
    settings1_performance = settings1_run['F1_Median'].values
    settings1_nondeter = settings1_run['ARI_Median'].values
    
    ## Settings 2
    settings2_run = df_all[(df_all['n_estimators']==512)&
                        (df_all['max_samples']==str(0.4))&
                        (df_all['max_features']==0.7)&
                        (df_all['bootstrap']==False)&
                        (df_all['n_jobs']==str(None))&
                        (df_all['warm_start']==False)]
    
    settings2_performance = settings2_run['F1_Median'].values
    settings2_nondeter = settings2_run['ARI_Median'].values
    
    fig = plt.Figure()
    plt.plot(settings2_nondeter, settings2_performance, ".", color = 'blue')
    plt.plot(settings1_nondeter, settings1_performance, ".", color = 'green')
    plt.plot(default_nondeter, default_performance, ".", color='red')
    plt.title("Scikit-learn - Isolation Forest")
    plt.xlabel("Determinism")
    plt.ylabel("Performance")
    plt.savefig("Fig/SkIF_D_F1_ARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    
    ## Calculate Percentage
    
    s1_win_performance = 0
    s1_lose_performance = 0
    s1_win_nd = 0
    s1_lose_nd = 0
    s2_win_performance = 0
    s2_lose_performance = 0
    s2_win_nd = 0
    s2_lose_nd = 0
    for i in range(default_run.shape[0]):
        fname = default_run.iloc[i]["Filename"]
        F1_Median = default_run.iloc[i]["F1_Median"]
        ARI_Median = default_run.iloc[i]["ARI_Median"]
        s1_run = settings1_run[settings1_run["Filename"] == fname]
        s1_F1_Median = s1_run["F1_Median"].values
        s1_ARI_Median = s1_run["ARI_Median"].values
        if s1_F1_Median > F1_Median:
            s1_win_performance += 1
        elif s1_F1_Median < F1_Median:
            s1_lose_performance += 1
        if s1_ARI_Median > ARI_Median:
            s1_win_nd += 1
        elif s1_ARI_Median > ARI_Median:
            s1_lose_nd += 1

        s2_run = settings2_run[settings2_run["Filename"] == fname]
        s2_F1_Median = s2_run["F1_Median"].values
        s2_ARI_Median = s2_run["ARI_Median"].values
        if s2_F1_Median > F1_Median:
            s2_win_performance += 1
        elif s2_F1_Median < F1_Median:
            s2_lose_performance += 1
        if s2_ARI_Median > ARI_Median:
            s2_win_nd += 1
        elif s2_ARI_Median > ARI_Median:
            s2_lose_nd += 1
            
    print(s1_win_performance/(s1_win_performance+s1_lose_performance))
    print(s1_win_nd/(s1_win_nd+s1_lose_nd))
    
    print(s2_win_performance/(s2_win_performance+s2_lose_performance))
    print(s2_win_nd/(s2_win_nd+s2_lose_nd))
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    
    plot_ari_f1()
        
        
        