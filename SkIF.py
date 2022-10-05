#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 16:18:20 2022

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
import pingouin as pg
import random

# datasetFolderDir = '/Users/muyeedahmed/Desktop/Research/Dataset/Dataset_Anomaly/'
# datasetFolderDir = '/home/neamtiu/Desktop/ma234/AnomalyDetection/Dataset/'
datasetFolderDir = '../Dataset_Combined/'


# def isolationforest(filename, n_estimators, max_samples, max_features, bootstrap, n_jobs, warm_start):
def isolationforest(filename, parameters, parameter_iteration):
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
        print(parameters[p][0], end=': ')
        for pv in range(len(parameters[p][2])):
            passing_param[p][1] = parameters[p][2][pv]
            runIF(filename, X, gt, passing_param, parameter_iteration)
            print(parameters[p][2][pv], end = ', ')
        print()
       
    
    
def runIF(filename, X, gt, params, parameter_iteration):

    labelFile = filename + "_" + str(params[0][1]) + "_" + str(params[1][1]) + "_" + str(params[2][1]) + "_" + str(params[3][1]) + "_" + str(params[4][1]) + "_" + str(params[5][1]) + "_" + str(params[6][1])

    if os.path.exists("IF/Labels_"+labelFile+".csv") == 1:
        # print("The Labels Already Exist")
        # print("IF/Labels_"+labelFile+".csv")
        return
    
    
    fstat_acc=open("Stats/SkIF_Accuracy.csv", "a")
    fstat_acc.write(filename+','+ str(params[0][1]) + ','+ str(params[1][1]) + ',' + str(params[2][1]) + ',' + str(params[3][1]) + ',' + str(params[4][1]) + ',' + str(params[5][1]) + ',' + str(params[6][1]) + ',' + str(parameter_iteration))
    
    fstat_f1=open("Stats/SkIF_F1.csv", "a")
    fstat_f1.write(filename+','+ str(params[0][1]) + ','+ str(params[1][1]) + ',' + str(params[2][1]) + ',' + str(params[3][1]) + ',' + str(params[4][1]) + ',' + str(params[5][1]) + ',' + str(params[6][1]) + ',' + str(parameter_iteration))
    
    fstat_ari=open("Stats/SkIF_ARI.csv", "a")
    fstat_ari.write(filename+','+ str(params[0][1]) + ','+ str(params[1][1]) + ',' + str(params[2][1]) + ',' + str(params[3][1]) + ',' + str(params[4][1]) + ',' + str(params[5][1]) + ',' + str(params[6][1]) + ',' + str(parameter_iteration))
    
    
    for i in range(10):##
        clustering = IsolationForest(n_estimators=params[0][1], max_samples=params[1][1], 
                                      max_features=params[3][1], bootstrap=params[4][1], 
                                      n_jobs=params[5][1], warm_start=params[6][1]).fit(X)
    
        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        
        
        
        accuracy = metrics.accuracy_score(gt, l)
        fstat_acc.write(','+str(accuracy))
        
        f1 = metrics.f1_score(gt, l)
        fstat_f1.write(','+str(f1))
        
        ari = adjusted_rand_score(gt, l)
        fstat_ari.write(','+str(ari))
        
    flabel=open("IF/Labels_"+labelFile+".csv", 'a')
    flabel.write("Done")
    flabel.close()
    
    
    fstat_acc.write('\n')
    fstat_f1.write('\n')
    fstat_ari.write('\n')
    
    fstat_f1.close()
    fstat_acc.close()
    fstat_ari.close()

def calculate_score(allFiles, Tool, Algo, parameter, parameter_values, all_parameters):
    i_n_estimators=all_parameters[0][1]
    i_max_samples=all_parameters[1][1]
    i_contamination=all_parameters[2][1]
    i_max_features=all_parameters[3][1]
    i_bootstrap=all_parameters[4][1]
    i_n_jobs=all_parameters[5][1]
    i_warm_start = all_parameters[6][1]
    
    dfacc = pd.read_csv("Stats/SkIF_Accuracy.csv")
    dff1 = pd.read_csv("Stats/SkIF_F1.csv")
    runs = []
    for i in range(10):##
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

    df_acc_r = df_acc_r.fillna(0)
    df_acc_m = df_acc_m.fillna(0)
    df_f1_r = df_f1_r.fillna(0)
    df_f1_m = df_f1_m.fillna(0)

    ### Friedman Test
    friedman_test_acc_m = pg.friedman(data=df_acc_m, dv="Accuracy_Median", within=parameter, subject="Filename")
    pvalue_friedman_acc_m = friedman_test_acc_m['p-unc']['Friedman']
    
    friedman_test_f1_m = pg.friedman(data=df_f1_m, dv="F1Score_Median", within=parameter, subject="Filename")
    pvalue_friedman_f1_m = friedman_test_f1_m['p-unc']['Friedman']
    
    friedman_test_acc_r = pg.friedman(data=df_acc_r, dv="Accuracy_Range", within=parameter, subject="Filename")
    pvalue_friedman_acc_r = friedman_test_acc_r['p-unc']['Friedman']
    
    friedman_test_f1_r = pg.friedman(data=df_f1_r, dv="F1Score_Range", within=parameter, subject="Filename")
    pvalue_friedman_f1_r = friedman_test_f1_r['p-unc']['Friedman']
    
    avg_friedman = (pvalue_friedman_acc_m + pvalue_friedman_f1_m + pvalue_friedman_acc_r + pvalue_friedman_f1_r)/4
    return avg_friedman, df_acc_m[parameter].loc[df_acc_m["Accuracy_Median"].idxmax()], df_acc_r[parameter].loc[df_acc_r["Accuracy_Range"].idxmin()], df_f1_m[parameter].loc[df_f1_m["F1Score_Median"].idxmax()], df_f1_r[parameter].loc[df_f1_r["F1Score_Range"].idxmin()]  

if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
    
    parameters = []
    
    n_estimators = [2, 4, 8, 16, 32, 64, 100, 128, 256, 512]##
    max_samples = ['auto', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    contamination = ['auto'] 
    max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bootstrap = [True, False]
    n_jobs = [1, None] 
    warm_start = [True, False]
    
    parameters.append(["n_estimators", 100, n_estimators])
    parameters.append(["max_samples", 'auto', max_samples])
    parameters.append(["contamination", 'auto', contamination])
    parameters.append(["max_features", 1.0, max_features])
    parameters.append(["bootstrap", False, bootstrap])
    parameters.append(["n_jobs", None, n_jobs])
    parameters.append(["warm_start", False, warm_start])
    
    R = ""
    for i in range(9):##
        R += "R"+str(i)+","
    R+="R9"##
    if os.path.exists("Stats/SkIF_Accuracy.csv") == 0:
        fstat_acc=open("Stats/SkIF_Accuracy.csv", "w")
        fstat_acc.write('Filename,n_estimators,max_samples,contamination,max_features,bootstrap,n_jobs,warm_start,Parameter_Iteration,'+R+"\n")
        fstat_acc.close()
        
    if os.path.exists("Stats/SkIF_F1.csv") == 0: 
        fstat_f1=open("Stats/SkIF_F1.csv", "w")
        fstat_f1.write('Filename,n_estimators,max_samples,contamination,max_features,bootstrap,n_jobs,warm_start,Parameter_Iteration,'+R+"\n")
        fstat_f1.close()

    if os.path.exists("Stats/SkIF_ARI.csv") == 0:    
        fstat_ari=open("Stats/SkIF_ARI.csv", "w")
        fstat_ari.write('Filename,n_estimators,max_samples,contamination,max_features,bootstrap,n_jobs,warm_start,Parameter_Iteration,'+R+"\n")
        fstat_ari.close()
    print(master_files)
    for param_iteration in range(len(parameters)):
        # for FileNumber in range(len(master_files)):
        rand_files = random.sample(master_files, 30)
        
        for FileNumber in range(30):
            print(FileNumber, end=' ')
            isolationforest(rand_files[FileNumber], parameters, param_iteration)
            
            

        friedmanValues = [10]*7
        f1_range = [0]*7
        f1_median =[0]*7 
        for i in range(7):
            if len(parameters[i][2]) > 1:
                friedmanValues[i], _, _, f1_median[i], f1_range[i]= calculate_score(master_files, 'Sk', 'IF', parameters[i][0], parameters[i][2], parameters)
            
        # friedmanValues[0], _, _, f1_median[0], f1_range[0]= calculate_score(master_files, 'Sk', 'IF', 'n_estimators', n_estimators, parameters)
        # friedmanValues[1], _, _, f1_median[1], f1_range[1]= calculate_score(master_files, 'Sk', 'IF', 'max_samples', max_samples, parameters)
    
        # friedmanValues[3], _, _, f1_median[3], f1_range[3]= calculate_score(master_files, 'Sk', 'IF', 'max_features', max_features, parameters)
        # friedmanValues[4], _, _, f1_median[4], f1_range[4]= calculate_score(master_files, 'Sk', 'IF', 'bootstrap', bootstrap, parameters)
        # friedmanValues[5], _, _, f1_median[5], f1_range[5]= calculate_score(master_files, 'Sk', 'IF', 'n_jobs', n_jobs, parameters)
        # friedmanValues[6], _, _, f1_median[6], f1_range[6]= calculate_score(master_files, 'Sk', 'IF', 'warm_start', warm_start, parameters)
        # print(friedmanValues)
        index_min = np.argmin(friedmanValues)

        if index_min == 5 and f1_median[index_min] == 0:
            f1_median[index_min] = None
            
        parameters[index_min][1] = f1_median[index_min]
        parameters[index_min][2] = [f1_median[index_min]]
        # parameters[0][1] = 128##
        # parameters[0][2] = [128]##
        print(parameters)        
        
        