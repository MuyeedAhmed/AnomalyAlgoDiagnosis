#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 03:27:09 2022

@author: muyeedahmed
"""

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
import math

def plot_ari_f1():
    df_f1 = pd.read_csv("Stats/SkEE_F1.csv")
    df_ari = pd.read_csv("Stats/SkEE_ARI.csv")
    
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
    
    
    parameter_names = ["store_precision", "assume_centered", "support_fraction", "contamination"]
    join_on =["Filename", "store_precision", "assume_centered", "support_fraction", "contamination", "Parameter_Iteration"]

    df_all = pd.merge(df_f1, df_ari,  how='left', left_on=join_on, right_on =join_on)
    
    df_all.to_csv("Stats/SkEE_Merged.csv")
    
    median_df = df_all.groupby(parameter_names)[["F1_Median", "F1_Range", "ARI_Median"]].mean()
    median_df = median_df.reset_index()
    
    median_df.to_csv("Stats/SkEE_Grouped_Median.csv")
    
    
    # print(median_df.iloc[median_df["Performance"].idxmax()])
    # print(median_df.iloc[median_df["Nondeterminism"].idxmin()])
    
    '''
    Plot Group Summary
    '''
    default_run = median_df[(median_df['store_precision']==True)&
                            (median_df['assume_centered']==False)&
                            (median_df['support_fraction']==str(None))&
                            (median_df['contamination']==str(0.1))]

       
    default_performance = default_run['F1_Median'].values[0]
    default_nondeter_range = default_run['F1_Range'].values[0]
    default_nondeter_ari = default_run['ARI_Median'].values[0]
    
    performance = median_df["F1_Median"].values    
    nondeterminism_range = median_df["F1_Range"].values
    nondeterminism_ari = median_df["ARI_Median"].values

    fig = plt.Figure()
    plt.plot(nondeterminism_range, performance, ".")
    plt.plot(default_nondeter_range, default_performance, "o")
    plt.title("Scikit-learn - Robust Covariance")
    plt.xlabel("Nondeterminism (F1 Score Range)")
    plt.ylabel("Performance (F1 Score)")
    plt.savefig("Fig/SkEE_F1_Range.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

    fig = plt.Figure()
    plt.plot(nondeterminism_ari, performance, ".")
    plt.plot(default_nondeter_ari, default_performance, "o")
    plt.title("Scikit-learn - Robust Covariance")
    plt.xlabel("Determinism (ARI)")
    plt.ylabel("Performance (F1 Score)")
    plt.savefig("Fig/SkEE_F1_ARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    
    
    '''
    Plot For All Dataset
    '''
    ## Default
    default_run = df_all[(df_all['store_precision']==True)&
                        (df_all['assume_centered']==False)&
                        (df_all['support_fraction']==str(None))&
                        (df_all['contamination']==str(0.1))]
    default_performance = default_run['F1_Median'].values
    default_nondeter = default_run['ARI_Median'].values
    ## Settings 1
    settings1_run = df_all[(df_all['store_precision']==True)&
                        (df_all['assume_centered']==False)&
                        (df_all['support_fraction']==str(0.9))&
                        (df_all['contamination']==str(0.1))]
    settings1_performance = settings1_run['F1_Median'].values
    settings1_nondeter = settings1_run['ARI_Median'].values
    
    ## Settings 2
    settings2_run = df_all[(df_all['store_precision']==True)&
                        (df_all['assume_centered']==False)&
                        (df_all['support_fraction']==str(0.6))&
                        (df_all['contamination']==str(0.2))]
    settings2_performance = settings2_run['F1_Median'].values
    settings2_nondeter = settings2_run['ARI_Median'].values
    
    fig = plt.Figure()
    plt.plot(settings1_nondeter, settings1_performance, ".", color = 'green')
    plt.plot(settings2_nondeter, settings2_performance, ".", color = 'blue')
    plt.plot(default_nondeter, default_performance, ".", color='red')
    plt.title("Scikit-learn - Robust Covariance")
    plt.xlabel("Determinism")
    plt.ylabel("Performance")
    plt.savefig("Fig/SkEE_D_F1_ARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    
    plot_ari_f1()
        
        
        