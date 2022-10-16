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
# datasetFolderDir = '/Users/muyeedahmed/Desktop/Research/Dataset/Dataset_Anomaly/'
# datasetFolderDir = '/home/neamtiu/Desktop/ma234/AnomalyDetection/Dataset/'
# datasetFolderDir = '../Dataset_Combined/'
datasetFolderDir = 'Dataset/'

def ocsvm(filename, parameters, parameter_iteration):
    print(filename)
    folderpath = datasetFolderDir
    
    if os.path.exists(folderpath+filename+".mat") == 1:
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
        if os.path.getsize(folderpath+filename+".csv") > 1000000: # 10MB
            print("Didn\'t run -> Too large - ", filename)    
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
    # accuracy = []
    f1 = []
    ari = []
    for i in range(10):
        try:
            clustering = OneClassSVM(kernel=params[0][1], degree=params[1][1], gamma=params[2][1], coef0=params[3][1], tol=params[4][1], nu=params[5][1], 
                                         shrinking=params[6][1], cache_size=params[7][1], max_iter=params[8][1]).fit(X)
        except:
            return
    
        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        labels.append(l)

        # accuracy.append(metrics.accuracy_score(gt, l))
        f1.append(metrics.f1_score(gt, l))
        
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
          ari.append(adjusted_rand_score(labels[i], labels[j]))      
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
    
    fstat_ari=open("Stats/SkOCSVM_ARI.csv", "a")
    fstat_ari.write(filename+','+ str(params[0][1]) + ','+ str(params[1][1]) + ',' + str(params[2][1]) + ',' + str(params[3][1]) + ',' + str(params[4][1]) + ',' + str(params[5][1]) + ',' + str(params[6][1]) + ',' + str(params[7][1]) + ',' + str(params[8][1]) + ',' + str(parameter_iteration) + ',')
    fstat_ari.write(','.join(str(s) for s in ari) + '\n')
    fstat_ari.close()

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
    dfari =  pd.read_csv("Stats/SkOCSVM_ARI.csv")
    
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
            ari = dfari[(dfari['Filename']==filename)&
                            (dfari['kernel']==i_kernel)&
                            (dfari['degree']==i_degree)&
                            (dfari['gamma']==i_gamma)&
                            (dfari['coef0']==i_coef0)&
                            (dfari['tol']==i_tol)&
                            (dfari['nu']==i_nu)&
                            (dfari['shrinking']==i_shrinking)&
                            (dfari['cache_size']==i_cache_size)&
                            (dfari['max_iter']==i_max_iter)]
                        
            f1_values = f1[f1_runs].to_numpy()[0]
            ari_values = ari[ari_runs].to_numpy()[0]
            
            # accDiff = (np.percentile(accuracy_values, 75) - np.percentile(accuracy_values, 25))/(np.percentile(accuracy_values, 75) + np.percentile(accuracy_values, 25))
            # accuracy_range_all.append([filename, p, accDiff])
            # accuracy_med_all.append([filename, p, np.percentile(accuracy_values, 50)])
            
            f1Diff = (np.percentile(f1_values, 75) - np.percentile(f1_values, 25))/(np.percentile(f1_values, 75) + np.percentile(f1_values, 25))
            f1_range_all.append([filename, p, f1Diff])
            f1_med_all.append([filename, p, np.percentile(f1_values, 50)])
            
            ari_all.append([filename, p, np.percentile(ari_values, 50)])
            
    # df_acc_r = pd.DataFrame(accuracy_range_all, columns = ['Filename', parameter, 'Accuracy_Range'])
    # df_acc_m = pd.DataFrame(accuracy_med_all, columns = ['Filename', parameter, 'Accuracy_Median'])
    df_f1_r = pd.DataFrame(f1_range_all, columns = ['Filename', parameter, 'F1Score_Range'])
    df_f1_m = pd.DataFrame(f1_med_all, columns = ['Filename', parameter, 'F1Score_Median'])
    df_ari_m = pd.DataFrame(ari_all, columns = ['Filename', parameter, 'ARI'])


    ### Friedman Test
    
    # friedman_test_f1_m = pg.friedman(data=df_f1_m, dv="F1Score_Median", within=parameter, subject="Filename")
    # p_f_f1_m = friedman_test_f1_m['p-unc']['Friedman']
    
    # friedman_test_f1_r = pg.friedman(data=df_f1_r, dv="F1Score_Range", within=parameter, subject="Filename")
    # p_f_f1_r = friedman_test_f1_r['p-unc']['Friedman']
    
    # friedman_test_f1_m = pg.friedman(data=df_f1_m, dv="F1Score_Median", within=parameter, subject="Filename")
    # p_f_ari = friedman_test_f1_m['p-unc']['Friedman']
    
    
    ### Mann–Whitney U test

    mwu_f1_range = [[0 for i in range(len(parameter_values))] for j in range(len(parameter_values))]
    i = 0
    for p1 in parameter_values:
        p1_values = df_f1_r[df_f1_r[parameter] == p1]['F1Score_Range'].to_numpy()
        j = 0
        for p2 in parameter_values:
            p2_values = df_f1_r[df_f1_r[parameter] == p2]['F1Score_Range'].to_numpy()
            if len(p1_values) == 0 or len(p2_values) == 0:
                mwu_f1_range[i][j] = None
                continue
            _, mwu_f1_range[i][j] = stats.mannwhitneyu(x=p1_values, y=p2_values, alternative = 'greater')
            j += 1
        i += 1
    mwu_df_f1_range = pd.DataFrame(mwu_f1_range, columns = parameter_values)
    mwu_df_f1_range.index = parameter_values
    mwu_df_f1_range.to_csv("Mann–Whitney U test/MWU_SkOCSVM_F1_Range_"+parameter+".csv")
    
    try:
        mwu_geomean = gmean(gmean(mwu_f1_range))
        mwu_min = np.min(mwu_f1_range)
    except:
        mwu_geomean = 11
        mwu_min = 11
    
    f1_m_grouped = df_f1_m.groupby(parameter)[["F1Score_Median"]].median().reset_index()
    f1_r_grouped = df_f1_r.groupby(parameter)[["F1Score_Range"]].median().reset_index()
    ari_m_grouped = df_ari_m.groupby(parameter)[["ARI"]].median().reset_index()
    
    parameter_value_max_f1_median = f1_m_grouped[parameter].loc[f1_m_grouped["F1Score_Median"].idxmax()]
    parameter_value_min_f1_range = f1_r_grouped[parameter].loc[f1_r_grouped["F1Score_Range"].idxmin()]  
    parameter_value_max_ari = ari_m_grouped[parameter].loc[ari_m_grouped["ARI"].idxmax()]
    
    return mwu_geomean, mwu_min, parameter_value_max_f1_median, parameter_value_min_f1_range, parameter_value_max_ari


def plot_acc_range():
    df = pd.read_csv("Stats/SkOCSVM_F1.csv")
    runs = []
    for i in range(10):
        runs.append(('R'+str(i)))
    
    df["Performance"] = 0
    df["Nondeterminism"] = 0
    for i in range(df.shape[0]):
        run_values = df.loc[i, runs].tolist()
        
        range_ = (np.percentile(run_values, 75) - np.percentile(run_values, 25))/(np.percentile(run_values, 75) + np.percentile(run_values, 25))
        
        df.iloc[i, df.columns.get_loc('Performance')] =  np.mean(run_values)
        df.iloc[i, df.columns.get_loc('Nondeterminism')] = range_
    
    median_df = df.groupby(["kernel", "degree", "gamma", "coef0", "tol", "nu", "shrinking", "cache_size", "max_iter"])[["Performance", "Nondeterminism"]].median()
    median_df = median_df.reset_index()
    
    print(median_df.iloc[median_df["Performance"].idxmax()])
    print(median_df.iloc[median_df["Nondeterminism"].idxmin()])
    
    
    default_run = median_df[(median_df['kernel']=='rbf')&
                                    (median_df['degree']==3)&
                                    (median_df['gamma']=='scale')&
                                    (median_df['coef0']==0.0)&
                                    (median_df['tol']==1e-3)&
                                    (median_df['nu']==0.5)&
                                    (median_df['shrinking']==True)&
                                    (median_df['cache_size']==200)&
                                    (median_df['max_iter']==-1)]
    default_performance = default_run['Performance'].values[0]
    default_nondeter = default_run['Nondeterminism'].values[0]
        
    
    performance = median_df["Performance"].values
    nondeterminism = median_df["Nondeterminism"].values

    fig = plt.Figure()
    plt.plot(nondeterminism, performance, ".")
    plt.plot(default_nondeter, default_performance, "o")
    plt.title("F1 Score")
    plt.xlabel("Nondeterminism")
    plt.ylabel("Performance")
    plt.savefig("Fig/OCSVM_Sk_F1_Iter2.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


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
    
    R = ""
    for i in range(9):
        R += "R"+str(i)+","
    R+="R9"
    ARI_R = ""
    for i in range(44):
        ARI_R += "R"+str(i)+","
    ARI_R+="R44"
    
    if os.path.exists("Stats/SkOCSVM_Accuracy.csv") == 0:
        fstat_acc=open("Stats/SkOCSVM_Accuracy.csv", "w")
        fstat_acc.write('Filename,kernel,degree,gamma,coef0,tol,nu,shrinking,cache_size,max_iter,Parameter_Iteration,'+R+"\n")
        fstat_acc.close()
        
    if os.path.exists("Stats/SkOCSVM_F1.csv") == 0: 
        fstat_f1=open("Stats/SkOCSVM_F1.csv", "w")
        fstat_f1.write('Filename,kernel,degree,gamma,coef0,tol,nu,shrinking,cache_size,max_iter,Parameter_Iteration,'+R+"\n")
        fstat_f1.close()

    if os.path.exists("Stats/SkOCSVM_ARI.csv") == 0:    
        fstat_ari=open("Stats/SkOCSVM_ARI.csv", "w")
        fstat_ari.write('Filename,kernel,degree,gamma,coef0,tol,nu,shrinking,cache_size,max_iter,Parameter_Iteration,'+ARI_R+"\n")
        fstat_ari.close()
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
        parameters[index_min][1] = f1_range[index_min]
        parameters[index_min][2] = [f1_range[index_min]]
        
        fstat_winner=open("Stats/SkOCSVM_Winners.csv", "a")
        fstat_winner.write(parameters[index_min][0]+','+str(MWU_geo[index_min])+','+str(f1_median[index_min])+','+str(f1_range[index_min])+','+str(ari[index_min])+'\n')
        fstat_winner.close()
        
        print(parameters)              

    plot_acc_range()


        
        
        
        
        