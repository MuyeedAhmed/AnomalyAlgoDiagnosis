#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 02:30:04 2022

@author: muyeedahmed
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 04:20:08 2022

@author: muyeedahmed
"""

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
import math


def calculate():
    df_f1 = pd.read_csv("Stats/MatIF_F1.csv")
    df_ari = pd.read_csv("Stats/MatIF_ARI.csv")
    
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
    
    df_all = pd.merge(df_f1, df_ari,  how='left', left_on=["Filename","ContaminationFraction", "NumLearners", "NumObservationsPerLearner", "Parameter_Iteration"], right_on = ["Filename","ContaminationFraction", "NumLearners", "NumObservationsPerLearner", "Parameter_Iteration"])
    
    df_all.to_csv("Stats/MatIF_Merged.csv", index=False)
    
    median_df = df_all.groupby(["ContaminationFraction", "NumLearners", "NumObservationsPerLearner"])[["F1_Median", "F1_Range", "ARI_Median"]].mean()
    median_df = median_df.reset_index()
    
    median_df.to_csv("Stats/MatIF_Grouped_Median.csv", index=False)

def plot_ari_f1():
    median_df = pd.read_csv("Stats/MatIF_Grouped_Median.csv")    
    df_all = pd.read_csv("Stats/MatIF_Merged.csv")
    '''
    Plot Group Summary
    '''
    i_ContaminationFraction=0
    i_NumLearners=100
    i_NumObservationsPerLearner='auto'
   
    default_run_mean = median_df[(median_df['ContaminationFraction']==str(i_ContaminationFraction))&
                    (median_df['NumLearners']==i_NumLearners)&
                    (median_df['NumObservationsPerLearner']==str(i_NumObservationsPerLearner))]
    
    
    mean_default_performance = default_run_mean['F1_Median'].values[0]
    mean_default_nondeter_range = default_run_mean['F1_Range'].values[0]
    mean_default_nondeter_ari = default_run_mean['ARI_Median'].values[0]
    
    performance = median_df["F1_Median"].values    
    nondeterminism_range = median_df["F1_Range"].values
    nondeterminism_ari = median_df["ARI_Median"].values

    fig = plt.Figure()
    plt.plot(nondeterminism_range, performance, ".")
    plt.plot(mean_default_nondeter_range, mean_default_performance, "d")
    plt.title("Matlab - Isolation Forest")
    plt.xlabel("Nondeterminism (F1 Score Range)")
    plt.ylabel("Performance (F1 Score)")
    plt.savefig("Fig/MatIF_F1_Range.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

    fig = plt.Figure()
    plt.plot(nondeterminism_ari, performance, ".")
    plt.plot(mean_default_nondeter_ari, mean_default_performance, marker ='d')
    plt.title("Matlab - Isolation Forest")
    plt.xlabel("Determinism (ARI)")
    plt.ylabel("Performance (F1 Score)")
    plt.savefig("Fig/MatIF_F1_ARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    
    
    '''
    Plot For All Dataset
    '''
    ## Default
    default_run_all = df_all[(df_all['ContaminationFraction']==str(0))&
                    (df_all['NumLearners']==100)&
                    (df_all['NumObservationsPerLearner']=='auto')]
    default_all_performance = default_run_all['F1_Median'].values
    default_all_nondeter = default_run_all['ARI_Median'].values
    
    ## Settings 1
    settings1_run = df_all[(df_all['ContaminationFraction']=='IF')&
                            (df_all['NumLearners']==100)&
                            (df_all['NumObservationsPerLearner']=='auto')]
    settings1_performance = settings1_run['F1_Median'].values
    settings1_nondeter = settings1_run['ARI_Median'].values
    
    mean_settings1 = median_df[(median_df['ContaminationFraction']=='IF')&
                                (median_df['NumLearners']==100)&
                                (median_df['NumObservationsPerLearner']=='auto')]
    mean_settings1_performance = mean_settings1['F1_Median'].values
    mean_settings1_nondeter = mean_settings1['ARI_Median'].values
    
    ## Settings 2
    settings2_run = df_all[(df_all['ContaminationFraction']=='LOF')&
                            (df_all['NumLearners']==100)&
                            (df_all['NumObservationsPerLearner']=='auto')]
    
    settings2_performance = settings2_run['F1_Median'].values
    settings2_nondeter = settings2_run['ARI_Median'].values
    
    mean_settings2 = median_df[(median_df['ContaminationFraction']=='LOF')&
                                (median_df['NumLearners']==100)&
                                (median_df['NumObservationsPerLearner']=='auto')]
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
    plt.title("Matlab - Isolation Forest")
    plt.xlabel("Determinism (ARI)")
    plt.ylabel("Performance (F1 Score)")
    plt.savefig("Fig/MatIF_D_F1_ARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()




if __name__ == '__main__':
    calculate()
    plot_ari_f1()
        
        
        