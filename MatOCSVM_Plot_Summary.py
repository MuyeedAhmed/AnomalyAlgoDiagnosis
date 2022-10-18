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


def calculate():
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
    
    df_all.to_csv("Stats/MatOCSVM_Merged.csv", index=False)
    
    median_df = df_all.groupby(parameter_names)[["F1_Median", "F1_Range", "ARI_Median"]].mean()
    median_df = median_df.reset_index()
    
    median_df.to_csv("Stats/MatOCSVM_Grouped_Median.csv", index=False)


def plot_ari_f1():
    median_df = pd.read_csv("Stats/MatOCSVM_Grouped_Median.csv")    
    df_all = pd.read_csv("Stats/MatOCSVM_Merged.csv")
    '''
    Plot Group Summary
    '''
   
    default_run_mean = median_df[(median_df['ContaminationFraction']==str(0))&
                            (median_df['KernelScale']==str(1))&
                            (median_df['Lambda']=='auto')&
                            (median_df['NumExpansionDimensions']=='auto')&
                            (median_df['StandardizeData']==0)&
                            (median_df['BetaTolerance']==1e-4)&
                            (median_df['GradientTolerance']==1e-4)&
                            (median_df['IterationLimit']==1000)]
    
    mean_default_performance = default_run_mean['F1_Median'].values[0]
    mean_default_nondeter_range = default_run_mean['F1_Range'].values[0]
    mean_default_nondeter_ari = default_run_mean['ARI_Median'].values[0]
    
    performance = median_df["F1_Median"].values    
    nondeterminism_range = median_df["F1_Range"].values
    nondeterminism_ari = median_df["ARI_Median"].values

    fig = plt.Figure()
    plt.plot(nondeterminism_range, performance, ".")
    plt.plot(mean_default_nondeter_range, mean_default_performance, "d")
    plt.title("Matlab - One Class SVM")
    plt.xlabel("Nondeterminism (F1 Score Range)")
    plt.ylabel("Performance (F1 Score)")
    plt.savefig("Fig/MatOCSVM_F1_Range.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

    fig = plt.Figure()
    plt.plot(nondeterminism_ari, performance, ".")
    plt.plot(mean_default_nondeter_ari, mean_default_performance, marker ='d')
    plt.title("Matlab - One Class SVM")
    plt.xlabel("Determinism (ARI)")
    plt.ylabel("Performance (F1 Score)")
    plt.savefig("Fig/MatOCSVM_F1_ARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    
    
    '''
    Plot For All Dataset
    '''
    ## Default
    default_run_all = df_all[(df_all['ContaminationFraction']==str(0))&
                            (df_all['KernelScale']==str(1))&
                            (df_all['Lambda']=='auto')&
                            (df_all['NumExpansionDimensions']=='auto')&
                            (df_all['StandardizeData']==0)&
                            (df_all['BetaTolerance']==1e-4)&
                            (df_all['GradientTolerance']==1e-4)&
                            (df_all['IterationLimit']==1000)]
    
    default_all_performance = default_run_all['F1_Median'].values
    default_all_nondeter = default_run_all['ARI_Median'].values
    
    ## Settings 1
    settings1_run = df_all[(df_all['ContaminationFraction']=='IF')&
                           (df_all['KernelScale']=='auto')&
                           (df_all['Lambda']==str(1))&
                           (df_all['NumExpansionDimensions']=='auto')&
                           (df_all['StandardizeData']==0)&
                           (df_all['BetaTolerance']==1e-4)&
                           (df_all['GradientTolerance']==1e-4)&
                           (df_all['IterationLimit']==1000)]
    settings1_performance = settings1_run['F1_Median'].values
    settings1_nondeter = settings1_run['ARI_Median'].values
    
    mean_settings1 = median_df[(median_df['ContaminationFraction']=='IF')&
                           (median_df['KernelScale']=='auto')&
                           (median_df['Lambda']==str(1))&
                           (median_df['NumExpansionDimensions']=='auto')&
                           (median_df['StandardizeData']==0)&
                           (median_df['BetaTolerance']==1e-4)&
                           (median_df['GradientTolerance']==1e-4)&
                           (median_df['IterationLimit']==1000)]
    mean_settings1_performance = mean_settings1['F1_Median'].values
    mean_settings1_nondeter = mean_settings1['ARI_Median'].values
    
    ## Settings 2
    settings2_run = df_all[(df_all['ContaminationFraction']=='IF')&
                           (df_all['KernelScale']=="auto")&
                           (df_all['Lambda']=='auto')&
                           (df_all['NumExpansionDimensions']=='auto')&
                           (df_all['StandardizeData']==0)&
                           (df_all['BetaTolerance']==1e-4)&
                           (df_all['GradientTolerance']==1e-4)&
                           (df_all['IterationLimit']==1000)]
    
    settings2_performance = settings2_run['F1_Median'].values
    settings2_nondeter = settings2_run['ARI_Median'].values
    
    mean_settings2 = median_df[(median_df['ContaminationFraction']=='IF')&
                           (median_df['KernelScale']=="auto")&
                           (median_df['Lambda']=='auto')&
                           (median_df['NumExpansionDimensions']=='auto')&
                           (median_df['StandardizeData']==0)&
                           (median_df['BetaTolerance']==1e-4)&
                           (median_df['GradientTolerance']==1e-4)&
                           (median_df['IterationLimit']==1000)]
    mean_settings2_performance = mean_settings2['F1_Median'].values
    mean_settings2_nondeter = mean_settings2['ARI_Median'].values
    
    fig = plt.Figure()
    
    plt.plot(default_all_nondeter, default_all_performance, '.', color='red', marker = 'd', markersize = 4, alpha=.5)
    plt.plot(settings1_nondeter, settings1_performance, '.', color = 'green', marker = 'v', markersize = 4, alpha=.5)
    plt.plot(settings2_nondeter, settings2_performance, '.', color = 'blue', marker = '^', markersize = 4, alpha=.5)
     
    plt.plot(mean_default_nondeter_ari, mean_default_performance, '.', color='red', marker = 'd', markersize = 8, markeredgecolor='black', markeredgewidth=1.5)
    plt.plot(mean_settings1_nondeter, mean_settings1_performance, '.', color = 'green', marker = 'v', markersize = 8, markeredgecolor='black', markeredgewidth=1.5)
    plt.plot(mean_settings2_nondeter, mean_settings2_performance, '.', color = 'blue', marker = '^', markersize = 8, markeredgecolor='black', markeredgewidth=1.5)
    
    plt.legend(['Default Setting', 'Custom Setting 1', 'Custom Setting 2'])
    plt.title("Matlab - One Class SVM")
    plt.xlabel("Determinism (ARI)")
    plt.ylabel("Performance (F1 Score)")
    plt.savefig("Fig/MatOCSVM_D_F1_ARI.pdf", bbox_inches="tight", pad_inches=0)
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

    for i in range(default_run_all.shape[0]):
        fname = default_run_all.iloc[i]["Filename"]
        F1_Median = default_run_all.iloc[i]["F1_Median"]
        ARI_Median = default_run_all.iloc[i]["ARI_Median"]
        
        s1_run = settings1_run[settings1_run["Filename"] == fname]
        try:
            s1_F1_Median = s1_run["F1_Median"].values[0]
            s1_ARI_Median = s1_run["ARI_Median"].values[0]
        except:
            continue
        if s1_F1_Median >= F1_Median:
            s1_win_performance += 1
        elif s1_F1_Median < F1_Median:
            s1_lose_performance += 1
        if s1_ARI_Median >= ARI_Median:
            s1_win_nd += 1
        elif s1_ARI_Median < ARI_Median:
            s1_lose_nd += 1

        s2_run = settings2_run[settings2_run["Filename"] == fname]
        s2_F1_Median = s2_run["F1_Median"].values
        s2_ARI_Median = s2_run["ARI_Median"].values
        if s2_F1_Median >= F1_Median:
            s2_win_performance += 1
        elif s2_F1_Median < F1_Median:
            s2_lose_performance += 1
        if s2_ARI_Median >= ARI_Median:
            s2_win_nd += 1
        elif s2_ARI_Median < ARI_Median:
            s2_lose_nd += 1
            
    print("Default: ", mean_default_nondeter_ari, mean_default_performance)
    print("Settings1 : ", mean_settings1_nondeter, mean_settings1_performance)
    print("Settings1 : ", mean_settings2_nondeter, mean_settings2_performance)
    
    print("Setting 1", end=': ')
    print("Performance: ", s1_win_performance, s1_lose_performance)
    print("Deter: ", s1_win_nd, s1_lose_nd)
    
    # print(s1_win_performance/(s1_win_performance+s1_lose_performance), end=' / ')
    # print(s1_win_nd/(s1_win_nd+s1_lose_nd))
    print("Setting 2", end=': ')
    print("Performance: ", s2_win_performance, s2_lose_performance)
    print("Deter: ", s2_win_nd, s2_lose_nd)
    # print(s2_win_performance/(s2_win_performance+s2_lose_performance), end=' / ')
    # print(s2_win_nd/(s2_win_nd+s2_lose_nd))

if __name__ == '__main__':
    # calculate()
    plot_ari_f1()
        
        
        