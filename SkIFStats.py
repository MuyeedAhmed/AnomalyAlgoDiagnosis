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
import scipy.stats as stats

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
                            (dfacc['max_samples']==str(i_max_samples))&
                            (dfacc['max_features']==i_max_features)&
                            (dfacc['bootstrap']==i_bootstrap)&
                            (dfacc['n_jobs']==str(i_n_jobs))&
                            (dfacc['warm_start']==i_warm_start)]
                

            f1 = dff1[(dff1['Filename']==filename)&
                            (dff1['n_estimators']==i_n_estimators)&
                            (dff1['max_samples']==str(i_max_samples))&
                            (dff1['max_features']==i_max_features)&
                            (dff1['bootstrap']==i_bootstrap)&
                            (dff1['n_jobs']==str(i_n_jobs))&
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
    
    
    ### Mann–Whitney U test
    mwu = [[0 for i in range(len(parameter_values))] for j in range(len(parameter_values))]
    i = 0
    for p1 in parameter_values:
        p1_values = df_f1_m[df_f1_m[parameter] == p1]['F1Score_Median'].to_numpy()
        j = 0
        for p2 in parameter_values:
            p2_values = df_f1_m[df_f1_m[parameter] == p2]['F1Score_Median'].to_numpy()
            _, mwu[i][j] = stats.mannwhitneyu(x=p1_values, y=p2_values, alternative = 'greater')
            j += 1
        i += 1
    mwu_df_f1_range = pd.DataFrame(mwu, columns = parameter_values)
    mwu_df_f1_range.index = parameter_values
    mwu_df_f1_range.to_csv("Mann–Whitney U test/MWU_SkIF_F1_Median_"+parameter+".csv")
    
    
    mwu = [[0 for i in range(len(parameter_values))] for j in range(len(parameter_values))]
    i = 0
    for p1 in parameter_values:
        p1_values = df_f1_r[df_f1_r[parameter] == p1]['F1Score_Range'].to_numpy()
        j = 0
        for p2 in parameter_values:
            p2_values = df_f1_r[df_f1_r[parameter] == p2]['F1Score_Range'].to_numpy()
            _, mwu[i][j] = stats.mannwhitneyu(x=p1_values, y=p2_values, alternative = 'greater')
            j += 1
        i += 1
    mwu_df_f1_range = pd.DataFrame(mwu, columns = parameter_values)
    mwu_df_f1_range.index = parameter_values
    mwu_df_f1_range.to_csv("Mann–Whitney U test/MWU_SkIF_F1_Range_"+parameter+".csv")

    
    
    ### Friedman Test
    friedman_test_f1_r = pg.friedman(data=df_f1_r, dv="F1Score_Range", within=parameter, subject="Filename")
    pvalue_friedman_f1_r = friedman_test_f1_r['p-unc']
    
    
    
    return pvalue_friedman_f1_r
    
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


def plot_acc_range(measurement):
    df = pd.read_csv("Stats/SkIF_"+measurement+".csv")
    runs = []
    for i in range(30):
        runs.append(('R'+str(i)))
    
    df["Performance"] = 0
    df["Nondeterminism"] = 0
    for i in range(df.shape[0]):
        run_values = df.loc[i, runs].tolist()
        
        range_ = (np.percentile(run_values, 75) - np.percentile(run_values, 25))/(np.percentile(run_values, 75) + np.percentile(run_values, 25))
        
        df.iloc[i, df.columns.get_loc('Performance')] =  np.mean(run_values)
        df.iloc[i, df.columns.get_loc('Nondeterminism')] = range_
    
    median_df = df.groupby(["n_estimators", "max_samples", "max_features", "bootstrap", "n_jobs", "warm_start"])[["Performance", "Nondeterminism"]].median()
    median_df = median_df.reset_index()
    # print(median_df)
    # print(median_df["Performance"].idxmax())
    print(median_df.iloc[median_df["Performance"].idxmax()])
    # print(median_df["Nondeterminism"].idxmin())
    print(median_df.iloc[median_df["Nondeterminism"].idxmin()])
    
    
    
    # i_n_estimators=100
    # i_max_samples='auto'
    # i_contamination='auto' 
    # i_max_features=1.0
    # i_bootstrap=False
    # i_n_jobs="None"
    # i_warm_start = False
    # default_run = median_df[(median_df['n_estimators']==i_n_estimators)&
    #                                 (median_df['max_samples']==str(i_max_samples))&
    #                                 (median_df['max_features']==i_max_features)&
    #                                 (median_df['bootstrap']==i_bootstrap)&
    #                                 (median_df['n_jobs']==str(i_n_jobs))&
    #                                 (median_df['warm_start']==i_warm_start)]
    # default_performance = default_run['Performance'].values[0]
    # default_nondeter = default_run['Nondeterminism'].values[0]
        
    
    # performance = median_df["Performance"].values
    # nondeterminism = median_df["Nondeterminism"].values

    # fig = plt.Figure()
    # plt.plot(nondeterminism, performance, ".")
    # plt.plot(default_nondeter, default_performance, "o")
    # plt.title(measurement)
    # plt.xlabel("Nondeterminism")
    # plt.ylabel("Performance")
    # plt.savefig("Fig/IF_Sk_"+measurement+"_Iter1.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()
    
    
    
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
    n_jobs = [1, 'None'] 
    warm_start = [True, False]
    
    plot_acc_range("Accuracy")
    plot_acc_range("F1")
    plot_acc_range("ARI")
    
    # calculate_draw_score(master_files, 'Sk', 'IF', 'n_estimators', n_estimators)
    # calculate_draw_score(master_files, 'Sk', 'IF', 'max_samples', max_samples)
    # calculate_draw_score(master_files, 'Sk', 'IF', 'max_features', max_features)
    # calculate_draw_score(master_files, 'Sk', 'IF', 'bootstrap', bootstrap)
    # calculate_draw_score(master_files, 'Sk', 'IF', 'n_jobs', n_jobs)
    # calculate_draw_score(master_files, 'Sk', 'IF', 'warm_start', warm_start)
