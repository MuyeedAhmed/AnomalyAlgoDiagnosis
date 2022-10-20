#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:14:21 2022

@author: muyeedahmed
"""



import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
import numpy as np
from sklearn import metrics
from copy import copy, deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
import scipy.stats as stats
from scipy.stats import gmean
import math

datasetFolderDir = 'Dataset/'



def isolationforest(filename, parameters, parameter_iteration):
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

    for p in range(len(parameters)):
        passing_param = deepcopy(parameters)
        print(parameters[p][0], end=': ')
        for pv in range(len(parameters[p][2])):
            passing_param[p][1] = parameters[p][2][pv]
            runIF(filename, X, gt, passing_param, parameter_iteration)
            print(parameters[p][2][pv], end = ', ')
        print()
       
    
    
def runIF(filename, X, gt, params, parameter_iteration):
    
    labelFile = filename + "_" + str(params[0][1]) + "_" + str(params[1][1]) + "_" + str(params[2][1]) + "_" + str(params[3][1])

    if os.path.exists("IF_R/"+labelFile+".csv") == 0:
        print(labelFile)
        return
    if os.path.exists("IF_R_Done/"+labelFile+".csv"):
        return
    
    
    labels = []
    f1 = []
    ari = []
    
    
    labels =  pd.read_csv("IF_R/"+labelFile+".csv").to_numpy()
    for i in range(10):
        f1.append(metrics.f1_score(gt, np.int64((labels[i][1:])*1)))
        
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
          ari.append(adjusted_rand_score(np.int64((labels[i][1:])*1), np.int64((labels[j][1:])*1)))
          
    flabel_done=open("IF_R_Done/"+labelFile+".csv", 'a')
    flabel_done.write("Done")
    flabel_done.close()
    
    fstat_f1=open("Stats/RIF_F1.csv", "a")
    fstat_f1.write(filename+','+ str(params[0][1]) + ','+ str(params[1][1]) + ',' + str(params[2][1]) + ',' + str(params[3][1]) + ',' + str(parameter_iteration) + ',')
    fstat_f1.write(','.join(str(s) for s in f1) + '\n')
    fstat_f1.close()
    
    fstat_ari=open("Stats/RIF_ARI.csv", "a")
    fstat_ari.write(filename+','+ str(params[0][1]) + ','+ str(params[1][1]) + ',' + str(params[2][1]) + ',' + str(params[3][1]) + ',' + str(parameter_iteration) + ',')
    fstat_ari.write(','.join(str(s) for s in ari) + '\n')
    fstat_ari.close()

        
if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
        
    
    parameters = []
    ntrees = [2, 4, 8, 16, 32, 64, 100, 128, 256, 512]
    standardize_data = ["TRUE","FALSE"]
    sample_size = ['auto',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,"NULL"]
    ncols_per_tree = ['def',0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    parameters.append(["ntrees",512,ntrees])
    parameters.append(["standardize_data","TRUE",standardize_data])
    parameters.append(["sample_size",'auto',sample_size])
    parameters.append(["ncols_per_tree",'def',ncols_per_tree])
    
    
    R = ""
    for i in range(9):
        R += "R"+str(i)+","
    R+="R9"
    ARI_R = ""
    for i in range(44):
        ARI_R += "R"+str(i)+","
    ARI_R+="R44"
        
    if os.path.exists("Stats/RIF_F1.csv") == 0: 
        fstat_f1=open("Stats/RIF_F1.csv", "w")
        fstat_f1.write('Filename,ntrees,standardize_data,sample_size,ncols_per_tree,Parameter_Iteration,'+R+"\n")
        fstat_f1.close()
        
    if os.path.exists("Stats/RIF_ARI.csv") == 0:    
        fstat_ari=open("Stats/RIF_ARI.csv", "w")
        fstat_ari.write('Filename,ntrees,standardize_data,sample_size,ncols_per_tree,Parameter_Iteration,'+ARI_R+"\n")
        fstat_ari.close()
        
    if os.path.exists("Stats/RIF_Winners.csv") == 0:
        fstat_winner=open("Stats/RIF_Winners.csv", "w")
        fstat_winner.write('Parameter,MWU_P,Max_F1,Min_F1_Range,Max_ARI\n')
        fstat_winner.close()
    
    # # # for param_iteration in range(len(parameters)):
    # winners = pd.read_csv("Stats/RIF_Winners.csv")
    # for param_iteration in range(len(parameters)):
    #     for i in range(winners.shape[0]):
    #         if parameters[param_iteration][0] == winners.loc[i,'Parameter']:
    #             parameters[param_iteration][1] = winners.loc[i,'Min_F1_Range']
    #             try:
    #                 parameters[param_iteration][1] = int(parameters[param_iteration][1])
    #             except:
    #                 pass 
    #             parameters[param_iteration][2] = [winners.loc[i,'Min_F1_Range']]                
    
    # param_iteration = len(parameters)
    
    for FileNumber in range(len(master_files)):
        print(FileNumber, end=' ')
        isolationforest(master_files[FileNumber], parameters, 0)

    # MWU_geo = [10]*len(parameters)
    # MWU_min = [10]*len(parameters)
    # f1_range = [0]*len(parameters)
    # f1_median =[0]*len(parameters)
    # ari = [0]*len(parameters)
    # for i in range(len(parameters)):
    #     if len(parameters[i][2]) > 1:
    #         mwu_geomean, mwu_min, f1_median[i], f1_range[i], ari[i] = calculate_score(master_files, parameters[i][0], parameters[i][2], parameters, param_iteration)
            
    #         MWU_geo[i] = mwu_geomean
    #         MWU_min[i] = mwu_min
    # index_min = np.argmin(MWU_geo)

    # if index_min == 5 and f1_range[index_min] == 0:
    #     f1_range[index_min] = None
    # if MWU_min[index_min] > 2:
    #     print("MWU_min: ", end='')
    #     print(MWU_min)
    #     # break
    # else:
    #     parameters[index_min][1] = f1_range[index_min]
    #     parameters[index_min][2] = [f1_range[index_min]]
    
    #     fstat_winner=open("Stats/RIF_Winners.csv", "a")
    #     fstat_winner.write('\n'+parameters[index_min][0]+','+str(MWU_geo[index_min])+','+str(f1_median[index_min])+','+str(f1_range[index_min])+','+str(ari[index_min])+'\n')
    #     fstat_winner.close()
    
    # print(parameters)
        
        
        