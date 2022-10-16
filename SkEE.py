#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 05:01:01 2022

@author: muyeedahmed
"""
import warnings
warnings.filterwarnings('ignore')
import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
from sklearn.covariance import EllipticEnvelope
import numpy as np
from sklearn import metrics
from copy import copy, deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
import math
import random
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
import scipy.stats as stats
from scipy.stats import gmean

datasetFolderDir = 'Dataset/'

def ee(filename, parameters, parameter_iteration):
    print(filename)
    folderpath = datasetFolderDir
    
    if os.path.exists(folderpath+filename+".mat") == 1:
        if os.path.getsize(folderpath+filename+".mat") > 200000: # 200KB
            # print("Didn\'t run -> Too large - ", filename)    
            return
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
        if os.path.getsize(folderpath+filename+".csv") > 200000: # 200KB
            # print("Didn\'t run -> Too large - ", filename)    
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
            runEE(filename, X, gt, passing_param, parameter_iteration)
            print(parameters[p][2][pv], end = ', ')
        print()
      
    
def runEE(filename, X, gt, params, parameter_iteration):
    labelFile = filename + "_" + str(params[0][1]) + "_" + str(params[1][1]) + "_" + str(params[2][1]) + "_" + str(params[3][1])
    if os.path.exists("EE/Labels_Sk_EE_"+labelFile+".csv") == 1:
        return
    sp = params[0][1]
    ac = params[1][1]
    sf = params[2][1]
    cont = params[3][1]
    if cont == "LOF":
        percentage_file = pd.read_csv("Stats/SkPercentage.csv")
        cont  = percentage_file[percentage_file["Filename"] == filename]["LOF"].to_numpy()[0]
    if cont == "IF":
        percentage_file = pd.read_csv("Stats/SkPercentage.csv")
        cont  = percentage_file[percentage_file["Filename"] == filename]["IF"].to_numpy()[0]

    labels = []
    f1 = []
    ari = []
    for i in range(10):
        try:
            clustering = EllipticEnvelope(store_precision=sp, assume_centered=ac, 
                                     support_fraction=sf, contamination=cont).fit(X)
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
    if os.path.exists("../AnomalyAlgoDiagnosis_Labels/Labels_Sk_EE_"+labelFile+".csv") == 0:
        fileLabels=open("../AnomalyAlgoDiagnosis_Labels/Labels_Sk_EE_"+labelFile+".csv", 'a')
        for l in labels:
            fileLabels.write(','.join(str(s) for s in l) + '\n')
        fileLabels.close()
    
    flabel_done=open("EE/Labels_Sk_EE_"+labelFile+".csv", 'a')
    flabel_done.write("Done")
    flabel_done.close()
   
    fstat_f1=open("Stats/SkEE_F1.csv", "a")
    fstat_f1.write(filename+','+ str(params[0][1]) + ','+ str(params[1][1]) + ',' + str(params[2][1]) + ',' + str(params[3][1]) + ',' + str(parameter_iteration) + ',')
    fstat_f1.write(','.join(str(s) for s in f1) + '\n')
    fstat_f1.close()
    
    fstat_ari=open("Stats/SkEE_ARI.csv", "a")
    fstat_ari.write(filename+','+ str(params[0][1]) + ','+ str(params[1][1]) + ',' + str(params[2][1]) + ',' + str(params[3][1]) + ',' + str(parameter_iteration) + ',')
    fstat_ari.write(','.join(str(s) for s in ari) + '\n')
    fstat_ari.close()
    


def calculate_score(allFiles, parameter, parameter_values, all_parameters, p_iter):
    i_store_precision = all_parameters[0][1]
    i_assume_centered = all_parameters[1][1]
    i_support_fraction = all_parameters[2][1]
    i_contamination = all_parameters[3][1]
    
    # dfacc = pd.read_csv("Stats/SkEE_Accuracy.csv")
    dff1 = pd.read_csv("Stats/SkEE_F1.csv")
    dfari =  pd.read_csv("Stats/SkEE_ARI.csv")
    
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
            if parameter == 'store_precision':
                i_store_precision = p
            elif parameter == 'assume_centered':
                i_assume_centered = p
            elif parameter == 'support_fraction':
                i_support_fraction = p
            elif parameter == 'contamination':
                i_contamination = p
            
            
            f1 = dff1[(dff1['Filename']==filename)&
                            (dff1['store_precision']==i_store_precision)&
                            (dff1['assume_centered']==i_assume_centered)&
                            (dff1['support_fraction']==str(i_support_fraction))&
                            (dff1['contamination']==str(i_contamination))]
            
            if f1.empty:
                continue
            ari = dfari[(dfari['Filename']==filename)&
                            (dfari['store_precision']==i_store_precision)&
                            (dfari['assume_centered']==i_assume_centered)&
                            (dfari['support_fraction']==str(i_support_fraction))&
                            (dfari['contamination']==str(i_contamination))]
            
            f1_values = f1[f1_runs].to_numpy()[0]
            ari_values = ari[ari_runs].to_numpy()[0]
            
            f1Diff = (np.percentile(f1_values, 75) - np.percentile(f1_values, 25))/(np.percentile(f1_values, 75) + np.percentile(f1_values, 25))
            if math.isnan(f1Diff):
                f1Diff = 0
            f1_range_all.append([filename, p, f1Diff])
            f1_med_all.append([filename, p, np.percentile(f1_values, 50)])
            
            ari_all.append([filename, p, np.percentile(ari_values, 50)])
            
    df_f1_r = pd.DataFrame(f1_range_all, columns = ['Filename', parameter, 'F1Score_Range'])
    df_f1_m = pd.DataFrame(f1_med_all, columns = ['Filename', parameter, 'F1Score_Median'])
    df_ari_m = pd.DataFrame(ari_all, columns = ['Filename', parameter, 'ARI'])
    
    if parameter == 'support_fraction':
        df_f1_r = df_f1_r.fillna(0)
        df_f1_m = df_f1_m.fillna(0)
        df_ari_m = df_ari_m.fillna(0)
        df_f1_r[parameter] = df_f1_r[parameter].astype(float)
        df_f1_m[parameter] = df_f1_m[parameter].astype(float)
        df_ari_m[parameter] = df_ari_m[parameter].astype(float)
        parameter_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
   
    
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
    mwu_df_f1_range.to_csv("Mann–Whitney U test/MWU_SkEE_F1_Range_"+parameter+"_"+str(p_iter)+".csv")
    
    
    try:
        mwu_f1_range = [[z+1 for z in y] for y in mwu_f1_range]
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


if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
    
    parameters = []
    
    
    store_precision = [True, False]
    assume_centered = [True, False]
    support_fraction = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    contamination = [0.1, 0.2, 0.3, 0.4, 0.5, 'LOF', 'IF']
    
    parameters.append(["store_precision", True, store_precision])
    parameters.append(["assume_centered", False, assume_centered])
    parameters.append(["support_fraction", None, support_fraction])
    parameters.append(["contamination", 0.1, contamination])
    
    
    R = ""
    for i in range(9):
        R += "R"+str(i)+","
    R+="R9"
    ARI_R = ""
    for i in range(44):
        ARI_R += "R"+str(i)+","
    ARI_R+="R44"
    
    if os.path.exists("Stats/SkEE_Accuracy.csv") == 0:
        fstat_acc=open("Stats/SkEE_Accuracy.csv", "w")
        fstat_acc.write('Filename,store_precision,assume_centered,support_fraction,contamination,Parameter_Iteration,'+R+"\n")
        fstat_acc.close()
        
    if os.path.exists("Stats/SkEE_F1.csv") == 0: 
        fstat_f1=open("Stats/SkEE_F1.csv", "w")
        fstat_f1.write('Filename,store_precision,assume_centered,support_fraction,contamination,Parameter_Iteration,'+R+"\n")
        fstat_f1.close()

    if os.path.exists("Stats/SkEE_ARI.csv") == 0:    
        fstat_ari=open("Stats/SkEE_ARI.csv", "w")
        fstat_ari.write('Filename,store_precision,assume_centered,support_fraction,contamination,Parameter_Iteration,'+ARI_R+"\n")
        fstat_ari.close()
    if os.path.exists("Stats/SkEE_Winners.csv") == 0:  
        fstat_winner=open("Stats/SkEE_Winners.csv", "w")
        fstat_winner.write('Parameter,MWU_P,Max_F1,Min_F1_Range,Max_ARI\n')
        fstat_winner.close()
    
    for param_iteration in range(len(parameters)):
        for FileNumber in range(len(master_files)):
            print(FileNumber, end=' ')
            ee(master_files[FileNumber], parameters, param_iteration)
            

        MWU_geo = [10]*len(parameters)
        MWU_min = [10]*len(parameters)
        f1_range = [0]*len(parameters)
        f1_median =[0]*len(parameters) 
        ari = [0]*len(parameters)
        for i in range(len(parameters)):
            if len(parameters[i][2]) > 1:
                mwu_geomean, mwu_min, f1_median[i], f1_range[i], ari[i] = calculate_score(master_files, parameters[i][0], parameters[i][2], parameters, param_iteration)
                
                MWU_geo[i] = mwu_geomean
                MWU_min[i] = mwu_min
        index_min = np.argmin(MWU_geo)

        
        if index_min == 2 and f1_range[index_min] == 0:
            f1_range[index_min] = None
            
        if MWU_min[index_min] > 2:
            print("MWU_min: ", end='')
            print(MWU_min)
            break
        parameters[index_min][1] = f1_range[index_min]
        parameters[index_min][2] = [f1_range[index_min]]
        
        fstat_winner=open("Stats/SkEE_Winners.csv", "a")
        fstat_winner.write(parameters[index_min][0]+','+str(MWU_geo[index_min])+','+str(f1_median[index_min])+','+str(f1_range[index_min])+','+str(ari[index_min])+'\n')
        fstat_winner.close()
        
        print(parameters)              




        
        
        
        
        