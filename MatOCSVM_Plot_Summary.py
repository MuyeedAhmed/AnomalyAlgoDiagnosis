#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 12:11:36 2022

@author: muyeedahmed
"""


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
import math


def plot_ari_f1():
    df_f1 = pd.read_csv("Stats/MatOCSVM_F1.csv")
    df_ari = pd.read_csv("Stats/MatOCSVM_ARI.csv")
    
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
    
    parameter_names = ["ContaminationFraction","KernelScale","Lambda","NumExpansionDimensions","StandardizeData","BetaTolerance","GradientTolerance","IterationLimit"]
    join_on =["Filename", "ContaminationFraction","KernelScale","Lambda","NumExpansionDimensions","StandardizeData","BetaTolerance","GradientTolerance","IterationLimit", "Parameter_Iteration"]
    df_all = pd.merge(df_f1, df_ari,  how='left', left_on = join_on, right_on = join_on)
    
    df_all.to_csv("Stats/MatOCSVM_Merged.csv")
    
    median_df = df_all.groupby(parameter_names)[["F1_Median", "F1_Range", "ARI_Median"]].mean()
    median_df = median_df.reset_index()
    
    median_df.to_csv("Stats/MatOCSVM_Grouped_Median.csv")

    '''
    Plot Group Summary
    '''
    i_ContaminationFraction=0
    i_NumLearners=100
    i_NumObservationsPerLearner='auto'
   
    default_run = median_df[(median_df['ContaminationFraction']==str(0))&
                            (median_df['KernelScale']==str(1))&
                            (median_df['Lambda']=='auto')&
                            (median_df['NumExpansionDimensions']=='auto')&
                            (median_df['StandardizeData']==0)&
                            (median_df['BetaTolerance']==1e-4)&
                            (median_df['GradientTolerance']==1e-4)&
                            (median_df['IterationLimit']==1000)]
    
    default_performance = default_run['F1_Median'].values[0]
    default_nondeter_range = default_run['F1_Range'].values[0]
    default_nondeter_ari = default_run['ARI_Median'].values[0]
    
    performance = median_df["F1_Median"].values    
    nondeterminism_range = median_df["F1_Range"].values
    nondeterminism_ari = median_df["ARI_Median"].values

    fig = plt.Figure()
    plt.plot(nondeterminism_range, performance, ".")
    plt.plot(default_nondeter_range, default_performance, "o")
    plt.title("Matlab - One Class SVM")
    plt.xlabel("Nondeterminism (F1 Score Range)")
    plt.ylabel("Performance (F1 Score)")
    plt.savefig("Fig/MatOCSVM_F1_Range.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

    fig = plt.Figure()
    plt.plot(nondeterminism_ari, performance, ".")
    plt.plot(default_nondeter_ari, default_performance, "o")
    plt.title("Matlab - One Class SVM")
    plt.xlabel("Determinism (ARI)")
    plt.ylabel("Performance (F1 Score)")
    plt.savefig("Fig/MatOCSVM_F1_ARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    
    '''
    Plot For All Dataset
    '''
    ## Default

    default_run = df_all[(df_all['ContaminationFraction']==str(0))&
                            (df_all['KernelScale']==str(1))&
                            (df_all['Lambda']=='auto')&
                            (df_all['NumExpansionDimensions']=='auto')&
                            (df_all['StandardizeData']==0)&
                            (df_all['BetaTolerance']==1e-4)&
                            (df_all['GradientTolerance']==1e-4)&
                            (df_all['IterationLimit']==1000)]
    
    default_performance = default_run['F1_Median'].values
    default_nondeter = default_run['ARI_Median'].values
    ## Settings 1
    settings1_run = df_all[(df_all['ContaminationFraction']=='IF')&
                           (df_all['KernelScale']==str(1))&
                           (df_all['Lambda']=='auto')&
                           (df_all['NumExpansionDimensions']=='auto')&
                           (df_all['StandardizeData']==0)&
                           (df_all['BetaTolerance']==1e-4)&
                           (df_all['GradientTolerance']==1e-4)&
                           (df_all['IterationLimit']==1000)]
    settings1_performance = settings1_run['F1_Median'].values
    settings1_nondeter = settings1_run['ARI_Median'].values
    
    ## Settings 2
    settings2_run = df_all[(df_all['ContaminationFraction']=='LOF')&
                           (df_all['KernelScale']==str(1))&
                           (df_all['Lambda']=='auto')&
                           (df_all['NumExpansionDimensions']=='auto')&
                           (df_all['StandardizeData']==0)&
                           (df_all['BetaTolerance']==1e-4)&
                           (df_all['GradientTolerance']==1e-4)&
                           (df_all['IterationLimit']==1000)]
    
    settings2_performance = settings2_run['F1_Median'].values
    settings2_nondeter = settings2_run['ARI_Median'].values
    
    fig = plt.Figure()
    plt.plot(settings1_nondeter, settings1_performance, ".", color = 'green')
    plt.plot(settings2_nondeter, settings2_performance, ".", color = 'blue')
    plt.plot(default_nondeter, default_performance, "o", color='red')
    plt.title("Matlab - One Class SVM")
    plt.xlabel("Determinism")
    plt.ylabel("Performance")
    plt.savefig("Fig/MatOCSVM_D_F1_ARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    

if __name__ == '__main__':
    plot_ari_f1()
        
        
        