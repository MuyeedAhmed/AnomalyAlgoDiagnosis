#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 18:03:34 2022

@author: muyeedahmed
"""


import warnings
warnings.filterwarnings('ignore')

import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
import numpy as np
from sklearn import metrics
from copy import copy, deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.svm import OneClassSVM
# import pingouin as pg
import random
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
# import scikit_posthocs as sp
import scipy.stats as stats
from scipy.stats import gmean
import math
import scipy.stats as ss
import bisect 

import subprocess
import matlab.engine
eng = matlab.engine.start_matlab()

datasetFolderDir = 'Dataset/'

'''
Run Matlab from Python
Install Matlab Engine

python -m pip install matlabengine

Import and run

import matlab.engine
eng = matlab.engine.start_matlab()
eng.mat_test(nargout=0)
'''

def ocsvm(filename, parameters_r, parameters_mat, parameters_sk):
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
            # print("Didn\'t run -> NaN - ", filename)
            return
        
    elif os.path.exists(folderpath+filename+".csv") == 1:
        if os.path.getsize(folderpath+filename+".csv") > 200000: # 200KB
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
    
    DefaultARI, DefaultF1_r, DefaultF1_mat, DefaultF1_sk = runOCSVM(filename, X, gt, parameters_r, parameters_mat, parameters_sk)
    print(DefaultARI, DefaultF1_r, DefaultF1_mat, DefaultF1_sk)
    ## Rearrange "IF" and "LOF"
    mod_parameters_r = deepcopy(parameters_r)
    mod_parameters_mat = deepcopy(parameters_mat)
    mod_parameters_sk = deepcopy(parameters_sk)
    
    percentage_file = pd.read_csv("Stats/SkPercentage.csv")
    lof_cont = percentage_file[percentage_file["Filename"] == filename]["LOF"].to_numpy()[0]
    bisect.insort(mod_parameters_mat[0][2], lof_cont)
    if_cont = percentage_file[percentage_file["Filename"] == filename]["IF"].to_numpy()[0]   
    bisect.insort(mod_parameters_mat[0][2], if_cont)
    
    mod_parameters_mat[0][2][mod_parameters_mat[0][2].index(lof_cont)] = 'LOF'
    mod_parameters_mat[0][2][mod_parameters_mat[0][2].index(if_cont)] = 'IF'

    bisect.insort(mod_parameters_sk[5][2], lof_cont)
    bisect.insort(mod_parameters_sk[5][2], if_cont)
    
    mod_parameters_sk[5][2][mod_parameters_sk[5][2].index(lof_cont)] = 'LOF'
    mod_parameters_sk[5][2][mod_parameters_sk[5][2].index(if_cont)] = 'IF'
    
    bisect.insort(mod_parameters_r[5][2], lof_cont)
    bisect.insort(mod_parameters_r[5][2], if_cont)
    
    mod_parameters_r[5][2][mod_parameters_r[5][2].index(lof_cont)] = 'LOF'
    mod_parameters_r[5][2][mod_parameters_r[5][2].index(if_cont)] = 'IF'
    # ##
    
    #Shortcut
    mod_parameters_r[5][1] = "IF"
    mod_parameters_r[2][1] = "auto"
    
    mod_parameters_mat[0][1] = "IF"
    mod_parameters_mat[1][1] = "auto"
    
    mod_parameters_sk[5][1] = "IF"

    # print(mod_parameters_mat)
    
    
    tools = ['R', 'Matlab', 'Sklearn']
    
    param_r_b, param_mat_b, param_sk_b = deepcopy(mod_parameters_r), deepcopy(mod_parameters_mat), deepcopy(mod_parameters_sk)
    ari_score, f1_score_r, f1_score_mat, f1_score_sk = get_blind_route(X, gt, filename, param_r_b, param_mat_b, param_sk_b, tools)
    print(ari_score, f1_score_r, f1_score_mat, f1_score_sk)
    
    
    
    
    UninformedARI = str(ari_score)
    UninformedF1_r = str(f1_score_r)
    UninformedF1_mat = str(f1_score_mat)
    UninformedF1_sk = str(f1_score_sk)
    
    param_r_u, param_mat_u, param_sk_u = deepcopy(mod_parameters_r), deepcopy(mod_parameters_mat), deepcopy(mod_parameters_sk)
    ari_score, f1_score_r, f1_score_mat, f1_score_sk = get_guided_route(X, gt, filename, param_r_u, param_mat_u, param_sk_u, tools)    
    print(ari_score, f1_score_r, f1_score_mat, f1_score_sk)
    

    InformedARI = str(ari_score)
    InformedF1_r = str(f1_score_r)
    InformedF1_mat = str(f1_score_mat)
    InformedF1_sk = str(f1_score_sk)
    
    
    f_Route_Scores=open("Stats/OCSVM_SvMvR_Route_Scores.csv", "a")
    f_Route_Scores.write(filename+','+str(DefaultARI)+","+str(DefaultF1_r)+","+str(DefaultF1_mat)+","+str(DefaultF1_sk)+","+UninformedARI+","+UninformedF1_r+","+UninformedF1_mat+","+UninformedF1_sk+","+InformedARI+","+InformedF1_r+","+InformedF1_mat+","+InformedF1_sk+"\n")
    f_Route_Scores.close()
    
    f_param=open("Stats/OCSVM_SvMvR_Winner_Parameters.csv", "a")
    f_param.write(filename+','+str(param_r_b)+","+str(param_mat_b)+","+str(param_sk_b)+","+str(param_r_u)+","+str(param_mat_u)+","+str(param_sk_u)+"\n")
    f_param.close()

def get_blind_route(X, gt, filename, parameters_r_copy,parameters_mat_copy, parameters_sk_copy, tools):
    # blind_route_r = []
    # blind_route_mat = []
    # blind_route_sk = []
    if tools == []:
        ari_score, f1_score_r, f1_score_mat, f1_score_sk = runOCSVM(filename, X, gt, parameters_r_copy, parameters_mat_copy, parameters_sk_copy)
        return ari_score, f1_score_r, f1_score_mat, f1_score_sk
    # else:
    #     print()
    current_tool = tools[0]
    
    print(current_tool, end=' - ')
    
    parameters_copy = []
    if current_tool == 'Sklearn':
        parameters_copy = parameters_sk_copy
    elif current_tool == 'Matlab':
        parameters_copy = parameters_mat_copy
    else:
        parameters_copy = parameters_r_copy
            
    
    for p_c in range(len(parameters_copy)):
        # print(p_c, end=', ')
        parameter_route_c = []
        ari_scores_c = []
        # ari_scores_c.append(max_M_ARI)
        passing_param_c = parameters_copy

        default_ari, default_f1_r, default_f1_mat, default_f1_sk = get_blind_route(X, gt, filename, parameters_r_copy, parameters_mat_copy, parameters_sk_copy, tools[1:])
        if default_ari == -1:
            return -1, -1, -1, -1
        parameter_route_c.append([passing_param_c[p_c][1], default_ari, default_f1_r, default_f1_mat, default_f1_sk])
        ari_scores_c.append(default_ari)
        # # blind_route_mat += route_mat
        # blind_route_mat = route_mat
        # blind_route_sk = route_sk

        i_def = passing_param_c[p_c][2].index(passing_param_c[p_c][1])
        if i_def+1 == len(passing_param_c[p_c][2]):
            i_pv_c = i_def-1
        else:
            i_pv_c = i_def+1
        
        while True:
            if i_pv_c >= len(parameters_copy[p_c][2]):
                break
            if i_pv_c < 0:
                break

            passing_param_c[p_c][1] = parameters_copy[p_c][2][i_pv_c]
            
            ari_score, f1_score_r, f1_score_mat, f1_score_sk = get_blind_route(X, gt, filename, parameters_r_copy, parameters_mat_copy, parameters_sk_copy, tools[1:])
            # route_temp += route_mat
            if ari_score >= np.max(ari_scores_c):
                parameter_route_c.append([passing_param_c[p_c][1], ari_score, f1_score_r, f1_score_mat, f1_score_sk])
                ari_scores_c.append(ari_score)
                # blind_route_mat += route_temp
                # blind_route_mat = route_mat
                # blind_route_sk = route_sk
                # route_temp = []
            if ari_score != np.max(ari_scores_c):
                
                if i_pv_c - 1 > i_def:
                    break
                elif i_pv_c - 1 == i_def:
                    i_pv_c = i_def - 1
                else:
                    break
            else:
                if i_pv_c > i_def:
                    i_pv_c += 1
                else:
                    i_pv_c -= 1
        ari_scores_c = np.array(ari_scores_c)
        max_index = np.where(ari_scores_c == max(ari_scores_c))[0][-1]
        
        # default_index = np.where(ari_scores_c == default_ari)[0][0]
        # print(p_c)
        # print(parameters_copy)
        
        # print(max_index)
        # print(parameter_route_c)
        parameters_copy[p_c][1] = parameter_route_c[max_index][0]
        # blind_route_c.append([parameters_copy[p_c][0], max_index, parameter_route_c])
    return parameter_route_c[-1][1], parameter_route_c[-1][2], parameter_route_c[-1][3], parameter_route_c[-1][4]

def get_guided_route(X, gt, filename, parameters_r_copy,parameters_mat_copy, parameters_sk_copy, tools):
    if tools == []:
        ari_score, f1_score_r, f1_score_mat, f1_score_sk = runOCSVM(filename, X, gt, parameters_r_copy, parameters_mat_copy, parameters_sk_copy)
        return ari_score, f1_score_r, f1_score_mat, f1_score_sk
    # else:
    #     print()
    current_tool = tools[0]
    
    print(current_tool, end=' - ')
    
    parameters_copy = []
    if current_tool == 'Sklearn':
        parameters_copy = parameters_sk_copy
    elif current_tool == 'Matlab':
        parameters_copy = parameters_mat_copy
    else:
        parameters_copy = parameters_r_copy
            
    
    for p_c in range(len(parameters_copy)):
        # print(p_c, end=', ')
        parameter_route_c = []
        ari_scores_c = []
        f1_scores_c = []
        passing_param_c = parameters_copy

        default_ari, default_f1_r, default_f1_mat, default_f1_sk = get_guided_route(X, gt, filename, parameters_r_copy, parameters_mat_copy, parameters_sk_copy, tools[1:])
        if default_ari == -1:
            return -1, -1, -1, -1
        parameter_route_c.append([passing_param_c[p_c][1], default_ari, default_f1_r, default_f1_mat, default_f1_sk])
        ari_scores_c.append(default_ari)
        f1_scores_c.append((default_f1_r+ default_f1_mat+ default_f1_sk)/3)
        
        i_def = passing_param_c[p_c][2].index(passing_param_c[p_c][1])
        if i_def+1 == len(passing_param_c[p_c][2]):
            i_pv_c = i_def-1
        else:
            i_pv_c = i_def+1
        
        while True:
            if i_pv_c >= len(parameters_copy[p_c][2]):
                break
            if i_pv_c < 0:
                break

            passing_param_c[p_c][1] = parameters_copy[p_c][2][i_pv_c]
            
            ari_score, f1_score_r, f1_score_mat, f1_score_sk = get_guided_route(X, gt, filename, parameters_r_copy, parameters_mat_copy, parameters_sk_copy, tools[1:])
            if ari_score >= np.max(ari_scores_c) and f1_score_r > np.max(f1_scores_c):
                parameter_route_c.append([passing_param_c[p_c][1], ari_score, f1_score_r, f1_score_mat, f1_score_sk])
                ari_scores_c.append(ari_score)
                f1_scores_c.append((f1_score_r+ f1_score_mat+ f1_score_sk)/3)

            if ari_score != np.max(ari_scores_c):
                
                if i_pv_c - 1 > i_def:
                    break
                elif i_pv_c - 1 == i_def:
                    i_pv_c = i_def - 1
                else:
                    break
            else:
                if i_pv_c > i_def:
                    i_pv_c += 1
                else:
                    i_pv_c -= 1
        ari_scores_c = np.array(ari_scores_c)
        max_index = np.where(ari_scores_c == max(ari_scores_c))[0][-1]
        parameters_copy[p_c][1] = parameter_route_c[max_index][0]
    return parameter_route_c[-1][1], parameter_route_c[-1][2], parameter_route_c[-1][3], parameter_route_c[-1][4]


def runOCSVM(filename, X, gt, param_r, param_mat, param_sk):
    labelFile_r = filename + "_" + str(param_r[0][1]) + "_" + str(param_r[1][1]) + "_" + str(param_r[2][1]) + "_" + str(param_r[3][1]) + "_" + str(param_r[4][1]) + "_" + str(param_r[5][1]) + "_" + str(param_r[6][1]) + "_" + str(param_r[7][1]) + "_" + str(param_r[8][1])
    labelFile_mat = filename + "_" + str(param_mat[0][1]) + "_" + str(param_mat[1][1]) + "_" + str(param_mat[2][1]) + "_" + str(param_mat[3][1]) + "_" + str(param_mat[4][1]) + "_" + str(param_mat[5][1]) + "_" + str(param_mat[6][1]) + "_" + str(param_mat[7][1])
    labelFile_sk = filename + "_" + str(param_sk[0][1]) + "_" + str(param_sk[1][1]) + "_" + str(param_sk[2][1]) + "_" + str(param_sk[3][1]) + "_" + str(param_sk[4][1]) + "_" + str(param_sk[5][1]) + "_" + str(param_sk[6][1]) + "_" + str(param_sk[7][1]) + "_" + str(param_sk[8][1])
    
    
    if os.path.exists("Labels/OCSVM_R/"+labelFile_r+".csv") == 0:
        frr=open("GD_ReRun/ROCSVM.csv", "a")
        frr.write(filename+","+str(param_r[0][1])+","+str(param_r[1][1])+","+str(param_r[2][1])+","+str(param_r[3][1])+","+str(param_r[4][1])+","+str(param_r[5][1])+","+str(param_r[6][1])+","+str(param_r[7][1])+","+str(param_r[8][1])+'\n')
        frr.close()
        try:
            print(param_r[4][1])
            subprocess.call((["/usr/local/bin/Rscript", "--vanilla", "ROCSVM_Rerun.r"]))
            frr=open("GD_ReRun/ROCSVM.csv", "w")
            frr.write('Filename,kernel,degree,gamma,coef0,tolerance,nu,shrinking,cachesize,epsilon\n')
            frr.close()
            if os.path.exists("Labels/OCSVM_R/"+labelFile_r+".csv") == 0: 
                print("-Did not work-")
                return -1, -1, -1, -1
        except:
            print("\nDid not work-r-\n")
            return -1, -1, -1, -1
    if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile_mat+".csv") == 0:        
        frr=open("GD_ReRun/MatOCSVM.csv", "a")
        frr.write(filename+","+str(param_mat[0][1])+","+str(param_mat[1][1])+","+str(param_mat[2][1])+","+str(param_mat[3][1])+","+str(param_mat[4][1])+","+str(param_mat[5][1])+","+str(param_mat[6][1])+","+str(param_mat[7][1])+'\n')
        frr.close()
        try:
            eng.MatOCSVM_Rerun(nargout=0)
            frr=open("GD_ReRun/MatOCSVM.csv", "w")
            frr.write('Filename,ContaminationFraction,KernelScale,Lambda,NumExpansionDimensions,StandardizeData,BetaTolerance,BetaTolerance,GradientTolerance,IterationLimit\n')
            frr.close()
            if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile_mat+".csv") == 0: 
                print("-Did not work-m-")
                return -1, -1, -1, -1
        except:
            print("\nDid not work\n")
            return -1, -1, -1, -1
    if os.path.exists("Labels/OCSVM_Sk/Labels_Sk_OCSVM_"+labelFile_sk+".csv") == 0:
        skf1 = get_sk_f1(filename, param_sk, X, gt)
        if skf1 == -1:
            return -1, -1, -1, -1
    ##
    
    labels_r = pd.read_csv("Labels/OCSVM_R/"+labelFile_r+".csv").to_numpy()
    labels_mat = pd.read_csv("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile_mat+".csv", header=None).to_numpy()
    labels_sk = pd.read_csv("Labels/OCSVM_Sk/Labels_Sk_OCSVM_"+labelFile_sk+".csv", header=None).to_numpy()
    
    ari_mvr = []
    for i in range(len(labels_r)):
        for j in range(len(labels_mat)):
            ari_mvr.append(adjusted_rand_score(np.int64((labels_r[i][1:])*1), labels_mat[j]))
    ari_mvr = np.mean(ari_mvr)

    ari_rvs = []
    for i in range(len(labels_r)):
        for j in range(len(labels_sk)):
            ari_rvs.append(adjusted_rand_score(np.int64((labels_r[i][1:])*1), labels_sk[j]))
    ari_rvs = np.mean(ari_rvs)
    
    ari_mvs = []
    for i in range(len(labels_sk)):
        for j in range(len(labels_mat)):
            ari_mvs.append(adjusted_rand_score(labels_sk[i], labels_mat[j]))
    ari_mvs = np.mean(ari_mvs) 
    
    ari = []  
    ari = (ari_mvr + ari_rvs + ari_mvs)/3
    
    return ari, get_r_f1(filename, param_r, X, gt), get_mat_f1(filename, param_mat, X, gt), get_sk_f1(filename, param_sk, X, gt)

def get_sk_f1(filename, param_sk, X, gt):
    labelFile = filename + "_" + str(param_sk[0][1]) + "_" + str(param_sk[1][1]) + "_" + str(param_sk[2][1]) + "_" + str(param_sk[3][1]) + "_" + str(param_sk[4][1]) + "_" + str(param_sk[5][1]) + "_" + str(param_sk[6][1]) + "_" + str(param_sk[7][1]) + "_" + str(param_sk[8][1])
    
    if os.path.exists("Labels/OCSVM_Sk_Done/Labels_Sk_OCSVM_"+labelFile+".csv") == 1:
        # labels =  pd.read_csv("Labels/OCSVM_Sk/Labels_Sk_OCSVM_"+labelFile+".csv", header=None).to_numpy()
        # # print(labels)
        # # print(gt)
        # return metrics.f1_score(gt, labels[0])
        i_kernel=param_sk[0][1]
        i_degree=param_sk[1][1]
        i_gamma=param_sk[2][1]
        i_coef0=param_sk[3][1]
        i_tol=param_sk[4][1]
        i_nu=param_sk[5][1]
        i_shrinking=param_sk[6][1]
        i_cache_size=param_sk[7][1]
        i_max_iter=param_sk[8][1]
        
        dff1 =  pd.read_csv("Stats/BrokenDown/SkOCSVM_F1_"+filename+".csv")
        f1 = dff1[(dff1['Filename']==filename)&
                    (dff1['kernel']==i_kernel)&
                    (dff1['degree']==i_degree)&
                    (dff1['gamma']==i_gamma)&
                    (dff1['coef0']==i_coef0)&
                    (dff1['tol']==i_tol)&
                    (dff1['nu']==str(i_nu))&
                    (dff1['shrinking']==i_shrinking)&
                    (dff1['cache_size']==i_cache_size)&
                    (dff1['max_iter']==i_max_iter)]
        if f1.empty == 0:
            run_f1_values = f1["R"].to_numpy()
            
            return np.mean(run_f1_values)
  
    labels = []
    f1 = []
    nu_contamination = param_sk[5][1]
    if nu_contamination == "LOF":
        percentage_file = pd.read_csv("Stats/SkPercentage.csv")
        nu_contamination  = percentage_file[percentage_file["Filename"] == filename]["LOF"].to_numpy()[0]
    if nu_contamination == "IF":
        percentage_file = pd.read_csv("Stats/SkPercentage.csv")
        nu_contamination  = percentage_file[percentage_file["Filename"] == filename]["IF"].to_numpy()[0]
    
    try:
        clustering = OneClassSVM(kernel=param_sk[0][1], degree=param_sk[1][1], gamma=param_sk[2][1], coef0=param_sk[3][1], tol=param_sk[4][1], nu=nu_contamination, 
                              shrinking=param_sk[6][1], cache_size=param_sk[7][1], max_iter=param_sk[8][1]).fit(X)
    except:
        return -1
    l = clustering.predict(X)
    l = [0 if x == 1 else 1 for x in l]
    labels.append(l)
    f1.append(metrics.f1_score(gt, l))
         
    if os.path.exists("Labels/OCSVM_Sk/Labels_Sk_OCSVM_"+labelFile+".csv") == 0:
        fileLabels=open("Labels/OCSVM_Sk/Labels_Sk_OCSVM_"+labelFile+".csv", 'a')
        for l in labels:
            fileLabels.write(','.join(str(s) for s in l) + '\n')
        fileLabels.close()
    
    flabel_done=open("Labels/OCSVM_Sk_Done/Labels_Sk_OCSVM_"+labelFile+".csv", 'a')
    flabel_done.write("Done")
    flabel_done.close()
    
    
    fstat_f1=open("Stats/BrokenDown/SkOCSVM_F1_"+filename+".csv", "a")
    fstat_f1.write(filename+','+ str(param_sk[0][1]) + ','+ str(param_sk[1][1]) + ',' + str(param_sk[2][1]) + ',' + str(param_sk[3][1]) + ',' + str(param_sk[4][1]) + ',' + str(param_sk[5][1]) + ',' + str(param_sk[6][1]) + ',' + str(param_sk[7][1]) + ',' + str(param_sk[8][1]) + ',0,')
    fstat_f1.write(','.join(str(s) for s in f1) + '\n')
    fstat_f1.close()

    return np.mean(f1)

def get_mat_f1(filename, param_mat, X, gt):
    labelFile = filename + "_" + str(param_mat[0][1]) + "_" + str(param_mat[1][1]) + "_" + str(param_mat[2][1]) + "_" + str(param_mat[3][1]) + "_" + str(param_mat[4][1]) + "_" + str(param_mat[5][1]) + "_" + str(param_mat[6][1]) + "_" + str(param_mat[7][1])
    
    if os.path.exists("Labels/OCSVM_Matlab_Done/Labels_Mat_OCSVM_"+labelFile+".csv") == 1:
        i_ContaminationFraction = param_mat[0][1]
        i_KernelScale = param_mat[1][1]
        i_Lambda = param_mat[2][1]
        i_NumExpansionDimensions = param_mat[3][1]
        i_StandardizeData = param_mat[4][1]
        i_BetaTolerance = param_mat[5][1]
        i_GradientTolerance = param_mat[6][1]
        i_IterationLimit = param_mat[7][1]
        
        dff1 =  pd.read_csv("Stats/BrokenDown/MatOCSVM_F1_"+filename+".csv")
        f1 = dff1[(dff1['Filename']==filename)&
                    (dff1['ContaminationFraction']==str(i_ContaminationFraction))&
                    (dff1['KernelScale']==str(i_KernelScale))&
                    (dff1['Lambda']==str(i_Lambda))&
                    (dff1['NumExpansionDimensions']==str(i_NumExpansionDimensions))&
                    (dff1['StandardizeData']==i_StandardizeData)&
                    (dff1['BetaTolerance']==i_BetaTolerance)&
                    (dff1['GradientTolerance']==i_GradientTolerance)&
                    (dff1['IterationLimit']==i_IterationLimit)]
        
        if f1.empty == 0:
            runs_f1 = []
            for i in range(10):
                runs_f1.append(('R'+str(i)))
            run_f1_values = f1[runs_f1].to_numpy()
            
            return np.mean(np.mean(run_f1_values))
        # else:
            # print(dff1)
            # # print(dff1.dtypes)
            # print(labelFile)
            # f1 = dff1[(dff1['Filename']==filename)&
            #         (dff1['ContaminationFraction']==str(i_ContaminationFraction))]
            # print(f1)
            # f1 = dff1[(dff1['Filename']==filename)&
            #         (dff1['KernelScale']==str(i_KernelScale))]
            # print(f1)
    if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile+".csv") == 0:        
        frr=open("GD_ReRun/MatOCSVM.csv", "a")
        frr.write(filename+","+str(param_mat[0][1])+","+str(param_mat[1][1])+","+str(param_mat[2][1])+","+str(param_mat[3][1])+","+str(param_mat[4][1])+","+str(param_mat[5][1])+","+str(param_mat[6][1])+","+str(param_mat[7][1])+'\n')
        frr.close()
        return -1
    
    labels = []
    f1 = []
    ari = []
    
    labels =  pd.read_csv("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile+".csv", header=None).to_numpy()
    for i in range(10):
        f1.append(metrics.f1_score(gt, labels[i]))
        
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
          ari.append(adjusted_rand_score(labels[i], labels[j]))
          
    flabel_done=open("Labels/OCSVM_Matlab_Done/Labels_Mat_OCSVM_"+labelFile+".csv", 'a')
    flabel_done.write("Done")
    flabel_done.close()
    
    fstat_f1=open("Stats/BrokenDown/MatOCSVM_F1_"+filename+".csv", "a")
    fstat_f1.write(filename+','+ str(param_mat[0][1]) + ','+ str(param_mat[1][1]) + ',' + str(param_mat[2][1]) + ',' + str(param_mat[3][1]) + ',' + str(param_mat[4][1]) + ',' + str(param_mat[5][1]) + ',' + str(param_mat[6][1]) + ',' + str(param_mat[7][1]) + ',0,')
    fstat_f1.write(','.join(str(s) for s in f1) + '\n')
    fstat_f1.close()
    
    fstat_ari=open("Stats/MatOCSVM_ARI.csv", "a")
    fstat_ari.write(filename+','+ str(param_mat[0][1]) + ','+ str(param_mat[1][1]) + ',' + str(param_mat[2][1]) + ',' + str(param_mat[3][1]) + ',' + str(param_mat[4][1]) + ',' + str(param_mat[5][1]) + ',' + str(param_mat[6][1]) + ',' + str(param_mat[7][1]) + ',0,')
    fstat_ari.write(','.join(str(s) for s in ari) + '\n')
    fstat_ari.close()

    return np.mean(f1)

def get_r_f1(filename, param_r, X, gt):
    labelFile = filename + "_" + str(param_r[0][1]) + "_" + str(param_r[1][1]) + "_" + str(param_r[2][1]) + "_" + str(param_r[3][1]) + "_" + str(param_r[4][1]) + "_" + str(param_r[5][1]) + "_" + str(param_r[6][1]) + "_" + str(param_r[7][1]) + "_" + str(param_r[8][1])
    
    if os.path.exists("Labels/OCSVM_R_Done/"+labelFile+".csv") == 1:
        i_kernel = param_r[0][1]
        i_degree = param_r[1][1]
        i_gamma = param_r[2][1]
        i_coef0 = param_r[3][1]
        i_tolerance = param_r[4][1]
        i_nu = param_r[5][1]
        i_shrinking = param_r[6][1]
        i_cachesize = param_r[7][1]
        i_epsilon = param_r[8][1]
        
        dff1 =  pd.read_csv("Stats/BrokenDown/ROCSVM_F1_"+filename+".csv")
        if i_shrinking == "TRUE":
            i_shrinking = True
        else:
            i_shrinking = False
        f1 = dff1[(dff1['Filename']==filename)&
                    (dff1['kernel']==i_kernel)&
                    (dff1['degree']==i_degree)&
                    (dff1['gamma']==i_gamma)&
                    (dff1['coef0']==i_coef0)&
                    (dff1['tolerance']==i_tolerance)&
                    (dff1['nu']==str(i_nu))&
                    (dff1['shrinking']==i_shrinking)&
                    (dff1['cachesize']==i_cachesize)&
                    (dff1['epsilon']==i_epsilon)]
        if f1.empty == 0:
            run_f1_values = f1["R"].to_numpy()
            return np.mean(run_f1_values)
    

    if os.path.exists("Labels/OCSVM_R/"+labelFile+".csv") == 0:
        frr=open("GD_ReRun/ROCSVM.csv", "a")
        frr.write(filename+","+str(param_r[0][1])+","+str(param_r[1][1])+","+str(param_r[2][1])+","+str(param_r[3][1])+","+str(param_r[4][1])+","+str(param_r[5][1])+","+str(param_r[6][1])+","+str(param_r[7][1])+","+str(param_r[8][1])+'\n')
        frr.close()
        return -1
    
    labels = []
    f1 = []
    
    labels =  pd.read_csv("Labels/OCSVM_R/"+labelFile+".csv").to_numpy()
    
    f1.append(metrics.f1_score(gt, np.int64((labels[0][1:])*1)))
    
    flabel_done=open("Labels/OCSVM_R_Done/"+labelFile+".csv", 'a')
    flabel_done.write("Done")
    flabel_done.close()
    
    fstat_f1=open("Stats/BrokenDown/ROCSVM_F1_"+filename+".csv", "a")
    fstat_f1.write(filename+','+ str(param_r[0][1]) + ','+ str(param_r[1][1]) + ',' + str(param_r[2][1]) + ',' + str(param_r[3][1]) + ',' + str(param_r[4][1]) + ',' + str(param_r[5][1]) + ',' + str(param_r[6][1]) + ',' + str(param_r[7][1]) + ',' + str(param_r[8][1]) + ',0,')
    fstat_f1.write(','.join(str(s) for s in f1) + '\n')
    fstat_f1.close()
    
    return np.mean(f1)
    
def plot_ari_f1():
    Route_Scores = pd.read_csv("Stats/OCSVM_SvMvR_Route_Scores.csv")
    
    Route_Scores['DefaultF1'] = Route_Scores[['DefaultF1_r', 'DefaultF1_mat', 'DefaultF1_sk']].mean(axis=1)
    Route_Scores['UninformedF1'] = Route_Scores[['UninformedF1_r', 'UninformedF1_mat', 'UninformedF1_sk']].mean(axis=1)
    Route_Scores['InformedF1'] = Route_Scores[['InformedF1_r', 'InformedF1_mat', 'InformedF1_sk']].mean(axis=1)
    
    fig = plt.Figure()
    
    plt.plot(Route_Scores["DefaultF1"], Route_Scores["DefaultARI"], '.', color='red', marker = 'd', markersize = 4, alpha=.5)
    plt.plot(Route_Scores["UninformedF1"], Route_Scores["UninformedARI"], '.', color = 'green', marker = 'v', markersize = 4, alpha=.5)
    plt.plot(Route_Scores["InformedF1"], Route_Scores["InformedARI"], '.', color = 'blue', marker = '^', markersize = 4, alpha=.5)
     
    plt.plot(Route_Scores["DefaultF1"].mean(), Route_Scores["DefaultARI"].mean(), '.', color='red', marker = 'd', markersize = 12, markeredgecolor='black', markeredgewidth=1.5)
    plt.plot(Route_Scores["UninformedF1"].mean(), Route_Scores["UninformedARI"].mean(), '.', color = 'green', marker = 'v', markersize = 12, markeredgecolor='black', markeredgewidth=1.5)
    plt.plot(Route_Scores["InformedF1"].mean(), Route_Scores["InformedARI"].mean(), '.', color = 'blue', marker = '^', markersize = 12, markeredgecolor='black', markeredgewidth=1.5)
    
    plt.legend(['Default Setting', 'Uninformed Route', 'Informed Route'])
    plt.title("Isolation Forest - Inconsistency")
    plt.xlabel("Average Performance (F1 Score)")
    plt.ylabel("Determinism (ARI)")
    plt.savefig("Fig/OCSVM_SvMvR_GD_Comparison.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    
    # ## Calculate Percentage
    
    ui_win_performance_r = 0
    ui_lose_performance_r = 0
    i_win_performance_r = 0    
    ui_win_performance_mat = 0
    ui_lose_performance_mat = 0
    i_win_performance_mat = 0
    ui_win_performance_sk = 0
    ui_lose_performance_sk = 0
    i_win_performance_sk = 0    
    
    ui_win_nd = 0
    i_win_nd = 0
    
    for index, row in Route_Scores.iterrows():
        if row["UninformedF1_r"] > row["DefaultF1_r"]:
            ui_win_performance_r += 1
        elif row["UninformedF1_r"] < row["DefaultF1_r"]:
            ui_lose_performance_r += 1
        if row["UninformedF1_mat"] > row["DefaultF1_mat"]:
            ui_win_performance_mat += 1
        elif row["UninformedF1_mat"] < row["DefaultF1_mat"]:
            ui_lose_performance_mat += 1    
        if row["UninformedF1_sk"] > row["DefaultF1_sk"]:
            ui_win_performance_sk += 1
        elif row["UninformedF1_sk"] < row["DefaultF1_sk"]:
            ui_lose_performance_sk += 1
        
        
        if row["UninformedARI"] > row["DefaultARI"]:
            ui_win_nd += 1
    
        if row["InformedF1_r"] > row["DefaultF1_r"]:
            i_win_performance_r += 1
        if row["InformedF1_mat"] > row["DefaultF1_mat"]:
            i_win_performance_mat += 1
        if row["InformedF1_sk"] > row["DefaultF1_sk"]:
            i_win_performance_sk += 1
            
        if row["InformedARI"] > row["DefaultARI"]:
            i_win_nd += 1
    
    print(Route_Scores['DefaultARI'].mean(), " & ", Route_Scores['UninformedARI'].mean(), " & ", Route_Scores['InformedARI'].mean())
    print("- & ",ui_win_nd, " & ",i_win_nd)
    print("- & 0 & 0")
    print("r")
    print(Route_Scores['DefaultF1_r'].mean(), " & ", Route_Scores['UninformedF1_r'].mean(), " & ", Route_Scores['InformedF1_r'].mean())
    print("- & ",ui_win_performance_r, " & ",i_win_performance_r)
    print("- & ",ui_lose_performance_r, " & 0 ")
    print("matlab")
    print(Route_Scores['DefaultF1_mat'].mean(), " & ", Route_Scores['UninformedF1_mat'].mean(), " & ", Route_Scores['InformedF1_may'].mean())
    print("- & ",ui_win_performance_mat, " & ",i_win_performance_mat)
    print("- & ",ui_lose_performance_mat, " & 0 ")
    print("Sklearn")
    print(Route_Scores['DefaultF1_sk'].mean(), " & ", Route_Scores['UninformedF1_sk'].mean(), " & ", Route_Scores['InformedF1_sk'].mean())
    print("- & ",ui_win_performance_sk, " & ",i_win_performance_sk)
    print("- & ",ui_lose_performance_sk, " & 0 ")
    
if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    if os.path.exists("Stats/OCSVM_SvMvR_Route_Scores.csv"):
        done_files = pd.read_csv("Stats/OCSVM_SvMvR_Route_Scores.csv")
        done_files = done_files["Filename"].to_numpy()
        print(done_files)
    master_files = [x for x in master_files if x not in done_files]
    master_files.sort()
    
    parameters_r = []
    
    kernel = ['linear', 'polynomial', 'radial', 'sigmoid']
    degree = [3, 4, 5, 6]
    gamma = ['scale', 'auto']
    coef0 = [0, 0.1, 0.2, 0.3, 0.4]
    tolerance = [0.1, 0.01, 0.001, 0.0001]
    nu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    shrinking = ["TRUE", "FALSE"]
    cachesize = [50, 100, 200, 400]
    epsilon = [0.1, 0.2, 0.01, 0.05]
    
    parameters_r.append(["kernel", 'radial', kernel])
    parameters_r.append(["degree", 3, degree])
    parameters_r.append(["gamma","scale",gamma])
    parameters_r.append(["coef0", 0, coef0])
    parameters_r.append(["tolerance", 0.001, tolerance])
    parameters_r.append(["nu", 0.5, nu])
    parameters_r.append(["shrinking", "TRUE", shrinking])
    parameters_r.append(["cachesize", 200, cachesize])
    parameters_r.append(["epsilon",0.1,epsilon])
    
    
    parameters_mat = []
    
    ContaminationFraction = [0.05, 0.1, 0.15, 0.2, 0.25]
    KernelScale = [1, "auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    Lambda = ["auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    NumExpansionDimensions = ["auto", 2^12, 2^15, 2^17, 2^19]
    StandardizeData = [0, 1]
    BetaTolerance = [1e-2, 1e-3, 1e-4, 1e-5]
    GradientTolerance = [1e-3, 1e-4, 1e-5, 1e-6]
    IterationLimit = [100, 200, 500, 1000, 2000]
    
    parameters_mat.append(["ContaminationFraction", 0.1, ContaminationFraction])
    parameters_mat.append(["KernelScale", 1, KernelScale])
    parameters_mat.append(["Lambda", 'auto', Lambda])
    parameters_mat.append(["NumExpansionDimensions", 'auto', NumExpansionDimensions])
    parameters_mat.append(["StandardizeData", 0, StandardizeData])
    parameters_mat.append(["BetaTolerance", 0.0001, BetaTolerance])
    parameters_mat.append(["GradientTolerance", 0.0001, GradientTolerance])
    parameters_mat.append(["IterationLimit", 1000, IterationLimit])
    
    
    parameters_sk = []
    kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    degree = [3, 4, 5, 6] # Kernel poly only
    gamma = ['scale', 'auto'] # Kernel ???rbf???, ???poly??? and ???sigmoid???
    coef0 = [0.0, 0.1, 0.2, 0.3, 0.4] # Kernel ???poly??? and ???sigmoid???
    tol = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    nu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    shrinking = [True, False]
    cache_size = [50, 100, 200, 400]
    max_iter = [50, 100, 150, 200, 250, 300, -1]
    
    parameters_sk.append(["kernel", 'rbf', kernel])
    parameters_sk.append(["degree", 3, degree])
    parameters_sk.append(["gamma", 'scale', gamma])
    parameters_sk.append(["coef0", 0.0, coef0])
    parameters_sk.append(["tol", 0.001, tol])
    parameters_sk.append(["nu", 0.5, nu])
    parameters_sk.append(["shrinking", True, shrinking])
    parameters_sk.append(["cache_size", 200, cache_size])
    parameters_sk.append(["max_iter", -1, max_iter])
    
    # if os.path.exists("Stats/Inconsistency/OCSVM_mvr.csv") == 0:
    #     fmvr=open("Stats/Inconsistency/OCSVM_mvr.csv", "w")
    #     fmvr.write('Mat,R,ARI\n')
    #     fmvr.close()
    # if os.path.exists("Stats/Inconsistency/OCSVM_rvs.csv") == 0:
    #     frvs=open("Stats/Inconsistency/OCSVM_rvs.csv", "w")
    #     frvs.write('R,Sk,ARI\n')
    #     frvs.close()
    # if os.path.exists("Stats/Inconsistency/OCSVM_mvs.csv") == 0:
    #     fmvs=open("Stats/Inconsistency/OCSVM_mvs.csv", "w")
    #     fmvs.write('Mat,Sk,ARI\n')
    #     fmvs.close()
    if os.path.exists("Stats/OCSVM_SvMvR_Route_Scores.csv") == 0:  
        f_Route_Scores=open("Stats/OCSVM_SvMvR_Route_Scores.csv", "w")
        f_Route_Scores.write('Filename,DefaultARI,DefaultF1_r,DefaultF1_mat,DefaultF1_sk,UninformedARI,UninformedF1_r,UninformedF1_mat,UninformedF1_sk,InformedARI,InformedF1_r,InformedF1_mat,InformedF1_sk\n')
        f_Route_Scores.close()
    if os.path.exists("Stats/OCSVM_SvMvR_Winner_Parameters.csv") == 0:
        f_param=open("Stats/OCSVM_SvMvR_Winner_Parameters.csv", "w")
        f_param.write("Filename,R_Uni,Mat_Uni,Sk_Uni,R_Bi,Mat_Bi,Sk_Bi\n")
        f_param.close()
    
    frr=open("GD_ReRun/ROCSVM.csv", "w")
    frr.write('Filename,kernel,degree,gamma,coef0,tolerance,nu,shrinking,cachesize,epsilon\n')
    frr.close()
    
    frr=open("GD_ReRun/MatOCSVM.csv", "w")
    frr.write('Filename,ContaminationFraction,KernelScale,Lambda,NumExpansionDimensions,StandardizeData,BetaTolerance,BetaTolerance,GradientTolerance,IterationLimit\n')
    frr.close()
    
    
    
    total_r = pd.read_csv("Stats/ROCSVM_F1.csv")       
    total_mat = pd.read_csv("Stats/MatOCSVM_F1.csv")       
    total_sk = pd.read_csv("Stats/SkOCSVM_F1.csv")       
    
    for FileNumber in range(len(master_files)):
        broken_r = total_r[(total_r['Filename']==master_files[FileNumber])]
        broken_r.to_csv("Stats/BrokenDown/ROCSVM_F1_"+master_files[FileNumber]+".csv", index=False)
        broken_mat = total_mat[(total_mat['Filename']==master_files[FileNumber])]
        broken_mat.to_csv("Stats/BrokenDown/MatOCSVM_F1_"+master_files[FileNumber]+".csv", index=False)
        
        broken_sk = total_sk[(total_sk['Filename']==master_files[FileNumber])]
        broken_sk.to_csv("Stats/BrokenDown/SkOCSVM_F1_"+master_files[FileNumber]+".csv", index=False)
        
        print(FileNumber, end=' ')
        ocsvm(master_files[FileNumber], parameters_r, parameters_mat, parameters_sk)
        
        broken_r = pd.read_csv("Stats/BrokenDown/ROCSVM_F1_"+master_files[FileNumber]+".csv")
        total_r = pd.merge(total_r, broken_r, how='outer')
        broken_mat = pd.read_csv("Stats/BrokenDown/MatOCSVM_F1_"+master_files[FileNumber]+".csv")
        total_mat = pd.merge(total_mat, broken_mat, how='outer')
        broken_sk = pd.read_csv("Stats/BrokenDown/SkOCSVM_F1_"+master_files[FileNumber]+".csv")
        total_sk = pd.merge(total_sk, broken_sk, how='outer')
        
    total_r.to_csv("Stats/ROCSVM_F1.csv", index=False)       
    total_mat.to_csv("Stats/MatOCSVM_F1.csv", index=False)       
    total_sk.to_csv("Stats/SkOCSVM_F1.csv", index=False)      
    
    # ocsvm("KnuggetChase3", parameters_r, parameters_mat, parameters_sk)
    # plot_ari_f1() 

    