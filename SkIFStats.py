#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 21:00:58 2022

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
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
from sklearn.metrics.cluster import adjusted_rand_score
import pingouin as pg
import scikit_posthocs as sp

datasetFolderDir = '/Users/muyeedahmed/Desktop/Research/Dataset/Dataset_Combined/'
# datasetFolderDir = '/home/neamtiu/Desktop/ma234/AnomalyDetection/Dataset/'
# datasetFolderDir = '../Dataset_Combined/'


def calculate_draw_score(allFiles, Tool, Algo, parameter, parameter_values):
    i_n_estimators=100
    i_max_samples='auto'
    i_contamination='auto' 
    i_max_features=1.0
    i_bootstrap=False
    i_n_jobs="None"
    i_warm_start = False
    
    dfacc = pd.read_csv("Stats/SkIF_Accuracy.csv")
    dff1 = pd.read_csv("Stats/SkIF_F1.csv")
    
    runs = []
    for i in range(30):
        runs.append(('R'+str(i)))


    accuracy_range_all = []
    accuracy_med_all = []
    f1_range_all = []
    f1_med_all = []
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
            
            accuracy = dfacc[(dfacc['Filename']==filename)&
                            (dfacc['n_estimators']==i_n_estimators)&
                            (dfacc['max_samples']==i_max_samples)&
                            (dfacc['max_features']==i_max_features)&
                            (dfacc['bootstrap']==i_bootstrap)&
                            (dfacc['n_jobs']==i_n_jobs)&
                            (dfacc['warm_start']==i_warm_start)]
                

            f1 = dff1[(dff1['Filename']==filename)&
                            (dff1['n_estimators']==i_n_estimators)&
                            (dff1['max_samples']==i_max_samples)&
                            (dff1['max_features']==i_max_features)&
                            (dff1['bootstrap']==i_bootstrap)&
                            (dff1['n_jobs']==i_n_jobs)&
                            (dff1['warm_start']==i_warm_start)]
            if f1.empty:
                continue
            
            accuracy_values = accuracy[runs].to_numpy()[0]
            
            f1_values = f1[runs].to_numpy()[0]

            # ari[run] = adjusted_rand_score(gt, l)
                
            accDiff = (np.percentile(accuracy_values, 75) - np.percentile(accuracy_values, 25))/(np.percentile(accuracy_values, 75) + np.percentile(accuracy_values, 25))
            accuracy_range_all.append([filename, p, accDiff])
            accuracy_med_all.append([filename, p, np.percentile(accuracy_values, 50)])
            
            f1Diff = (np.percentile(f1_values, 75) - np.percentile(f1_values, 25))/(np.percentile(f1_values, 75) + np.percentile(f1_values, 25))
            f1_range_all.append([filename, p, f1Diff])
            f1_med_all.append([filename, p, np.percentile(f1_values, 50)])

    df_acc_r = pd.DataFrame(accuracy_range_all, columns = ['Filename', parameter, 'Accuracy_Range'])
    df_acc_m = pd.DataFrame(accuracy_med_all, columns = ['Filename', parameter, 'Accuracy_Median'])
    df_f1_r = pd.DataFrame(f1_range_all, columns = ['Filename', parameter, 'F1Score_Range'])
    df_f1_m = pd.DataFrame(f1_med_all, columns = ['Filename', parameter, 'F1Score_Median'])
    
    
    print(df_f1_r)
    friedman_test = pg.friedman(data=df_f1_r, dv="F1Score_Range", within=parameter, subject="Filename")
    pvalue_friedman=friedman_test['p-unc']
    
    
    # conv = sp.posthoc_conover_friedman(a=df_f1_r, y_col="F1Score_Range", group_col=parameter, block_col="Filename", 
    #                              p_adjust="fdr_bh", melted=True)
    # print(conv)
    
    # fig = plt.Figure()
    # axf = sns.boxplot(x=parameter, y="Accuracy_Range", data=df_acc_r)
    # # axf.set(xlabel=None)
    # plt.savefig("Fig/"+Algo+'_'+Tool+'_'+parameter+"_Accuracy_Range.pdf", bbox_inches="tight", pad_inches=0)
    # plt.clf()
    
    # fig = plt.Figure()
    # axf = sns.boxplot(x=parameter, y="Accuracy_Median", data=df_acc_m)
    # plt.savefig("Fig/"+Algo+'_'+Tool+'_'+parameter+"_Accuracy_Median.pdf", bbox_inches="tight", pad_inches=0)
    # plt.clf()
    
    # fig = plt.Figure()
    # axf = sns.boxplot(x=parameter, y="F1Score_Range", data=df_f1_r)
    # plt.savefig("Fig/"+Algo+'_'+Tool+'_'+parameter+"_F1Score_Range.pdf", bbox_inches="tight", pad_inches=0)
    # plt.clf()
    
    # fig = plt.Figure()
    # axf = sns.boxplot(x=parameter, y="F1Score_Median", data=df_f1_m)
    # plt.savefig("Fig/"+Algo+'_'+Tool+'_'+parameter+"_F1Score_Median.pdf", bbox_inches="tight", pad_inches=0)
    # plt.clf()

if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()

    n_estimators = [2, 4, 8, 16, 32, 64, 100, 128, 256, 512, 1024]
    max_samples = ['auto', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    contamination = ['auto'] 
    max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bootstrap = [True, False]
    n_jobs = [1, -1] 
    warm_start = [True, False]
    
    
    calculate_draw_score(master_files, 'Sk', 'IF', 'n_estimators', n_estimators)
    # calculate_draw_score(master_files, 'Sk', 'IF', 'max_samples', max_samples)
    # calculate_draw_score(master_files, 'Sk', 'IF', 'max_features', max_features)
    # calculate_draw_score(master_files, 'Sk', 'IF', 'bootstrap', bootstrap)
    # calculate_draw_score(master_files, 'Sk', 'IF', 'n_jobs', n_jobs)
    # calculate_draw_score(master_files, 'Sk', 'IF', 'warm_start', warm_start)
