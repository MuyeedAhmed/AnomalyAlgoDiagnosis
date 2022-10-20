#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 04:09:12 2022

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



def ocsvm(filename, parameters, parameter_iteration):
    print(filename)
    folderpath = datasetFolderDir
    
    if os.path.exists(folderpath+filename+".mat") == 1:
        # if os.path.getsize(folderpath+filename+".mat") > 200000: # 200KB
        #     # print("Didn\'t run -> Too large - ", filename)    
        #     return
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
        # if os.path.getsize(folderpath+filename+".csv") > 200000: # 200KB
        #     # print("Didn\'t run -> Too large - ", filename)    
        #     return
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
            runOCSVM(filename, X, gt, passing_param, parameter_iteration)
            print(parameters[p][2][pv], end = ', ')
        print()
    
       
    
    
def runOCSVM(filename, X, gt, params, parameter_iteration):
        
    labelFile = filename + "_" + str(params[0][1]) + "_" + str(params[1][1]) + "_" + str(params[1][1]) + "_" + str(params[3][1]) + "_" + str(params[4][1]) + "_" + str(params[5][1]) + "_" + str(params[6][1]) + "_" + str(params[7][1]) + "_" + str(params[8][1])

    if os.path.exists("OCSVM_R/"+labelFile+".csv") == 0:
        # print(labelFile)
        return
    else:
        print(labelFile)
        return
    if os.path.exists("OCSVM_R_Done/Labels_R_OCSVM_"+labelFile+".csv"):
        return
    
    labels = []
    f1 = []
    
    
    labels = pd.read_csv("OCSVM_R/Labels_R_OCSVM_"+labelFile+".csv", header=None).to_numpy()
    
    f1.append(metrics.f1_score(gt, labels[0]))
        
   
          
    flabel_done=open("OCSVM_R_Done/Labels_R_OCSVM_"+labelFile+".csv", 'a')
    flabel_done.write("Done")
    flabel_done.close()
    
    fstat_f1=open("Stats/ROCSVM_F1.csv", "a")
    fstat_f1.write(filename+','+ str(params[0][1]) + ','+ str(params[1][1]) + ',' + str(params[2][1]) + ',' + str(params[3][1]) + ',' + str(params[4][1]) + ',' + str(params[5][1]) + ',' + str(params[6][1]) + ',' + str(params[7][1]) + ',' + str(params[8][1]) + ',' + str(parameter_iteration) + ',')
    fstat_f1.write(','.join(str(s) for s in f1) + '\n')
    fstat_f1.close()
    
    

# def calculate_score(allFiles, parameter, parameter_values, all_parameters, p_iter):
#     i_ContaminationFraction = all_parameters[0][1]
#     i_KernelScale = all_parameters[1][1]
#     i_Lambda = all_parameters[2][1]
#     i_NumExpansionDimensions = all_parameters[3][1]
#     i_StandardizeData = all_parameters[4][1]
#     i_BetaTolerance = all_parameters[5][1]
#     i_GradientTolerance = all_parameters[6][1]
#     i_IterationLimit = all_parameters[7][1]
    
#     dff1 = pd.read_csv("Stats/ROCSVM_F1.csv")
    
    
#     f1_runs = "R"
    
#     f1_med_all = []
    
#     for filename in allFiles:
#         for p in parameter_values:
#             if parameter == 'ContaminationFraction':
#                 i_ContaminationFraction = p
#             elif parameter == 'KernelScale':
#                 i_KernelScale = p
#             elif parameter == 'Lambda':
#                 i_Lambda = p
#             elif parameter == 'NumExpansionDimensions':
#                 i_NumExpansionDimensions = p
#             elif parameter == 'StandardizeData':
#                 i_StandardizeData = p
#             elif parameter == 'BetaTolerance':
#                 i_BetaTolerance = p
#             elif parameter == 'GradientTolerance':
#                 i_GradientTolerance = p
#             elif parameter == 'IterationLimit':
#                 i_IterationLimit = p
                

#             f1 = dff1[(dff1['Filename']==filename)&
#                         (dff1['ContaminationFraction']==str(i_ContaminationFraction))&
#                         (dff1['KernelScale']==str(i_KernelScale))&
#                         (dff1['Lambda']==str(i_Lambda))&
#                         (dff1['NumExpansionDimensions']==str(i_NumExpansionDimensions))&
#                         (dff1['StandardizeData']==i_StandardizeData)&
#                         (dff1['BetaTolerance']==i_BetaTolerance)&
#                         (dff1['GradientTolerance']==i_GradientTolerance)&
#                         (dff1['IterationLimit']==i_IterationLimit)]
#             if f1.empty:
#                 continue
            
                        
#             f1_values = f1[f1_runs].to_numpy()[0]
            
#             f1_med_all.append([filename, p, np.percentile(f1_values, 50)])
            
#     df_f1_m = pd.DataFrame(f1_med_all, columns = ['Filename', parameter, 'F1Score_Median'])
    
    
    
#     ### Mann–Whitney U test

#     mwu_f1_range = [[0 for i in range(len(parameter_values))] for j in range(len(parameter_values))]
#     i = 0
#     for p1 in parameter_values:
#         p1_values = df_f1_r[df_f1_r[parameter] == p1]['F1Score_Range'].to_numpy()
#         j = 0
#         for p2 in parameter_values:
#             p2_values = df_f1_r[df_f1_r[parameter] == p2]['F1Score_Range'].to_numpy()
#             if len(p1_values) == 0 or len(p2_values) == 0:
#                 mwu_f1_range[i][j] = None
#                 continue
#             _, mwu_f1_range[i][j] = stats.mannwhitneyu(x=p1_values, y=p2_values, alternative = 'greater')
#             j += 1
#         i += 1
#     mwu_df_f1_range = pd.DataFrame(mwu_f1_range, columns = parameter_values)
#     mwu_df_f1_range.index = parameter_values
#     mwu_df_f1_range.to_csv("Mann–Whitney U test/MWU_ROCSVM_F1_Range_"+parameter+"_"+str(p_iter)+".csv")
    
#     try:
#         mwu_f1_range = [[z+1 for z in y] for y in mwu_f1_range]
#         mwu_geomean = gmean(gmean(mwu_f1_range))
#         mwu_min = np.min(mwu_f1_range)
#     except:
#         mwu_geomean = 11
#         mwu_min = 11
    
#     f1_m_grouped = df_f1_m.groupby(parameter)[["F1Score_Median"]].median().reset_index()
#     f1_r_grouped = df_f1_r.groupby(parameter)[["F1Score_Range"]].median().reset_index()
#     ari_m_grouped = df_ari_m.groupby(parameter)[["ARI"]].median().reset_index()
    
#     parameter_value_max_f1_median = f1_m_grouped[parameter].loc[f1_m_grouped["F1Score_Median"].idxmax()]
#     parameter_value_min_f1_range = f1_r_grouped[parameter].loc[f1_r_grouped["F1Score_Range"].idxmin()]  
#     parameter_value_max_ari = ari_m_grouped[parameter].loc[ari_m_grouped["ARI"].idxmax()]
    
#     return mwu_geomean, mwu_min, parameter_value_max_f1_median, parameter_value_min_f1_range, parameter_value_max_ari
    
        
if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
        
    parameters = []
    
    kernel = ['linear', 'polynomial', 'radial', 'sigmoid']
    degree = [3, 4, 5, 6]
    gamma = ['scale', 'auto']
    coef0 = [0.0, 0.1, 0.2, 0.3, 0.4]
    tolerance = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    nu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    shrinking = [True, False]
    cachesize = [50, 100, 200, 400]
    epsilon = [0.1, 0.2, 0.01, 0.05]
    
    parameters.append(["kernel", 'radial', kernel])
    parameters.append(["degree", 3, degree])
    parameters.append(["gamma","auto",gamma])
    parameters.append(["coef0", 0.0, coef0])
    parameters.append(["tolerance", 0.001, tolerance])
    parameters.append(["nu", 0.5, nu])
    parameters.append(["shrinking", True, shrinking])
    parameters.append(["cachesize", 200, cachesize])
    parameters.append(["epsilon",0.1,epsilon])
        
    
    R="R"
        
    if os.path.exists("Stats/ROCSVM_F1.csv") == 0: 
        fstat_f1=open("Stats/ROCSVM_F1.csv", "w")
        fstat_f1.write('Filename,kernel,degree,gamma,coef0,tolerance,nu,shrinking,cachesize,epsilon,Parameter_Iteration,'+R+"\n")
        fstat_f1.close()
        
        
    if os.path.exists("Stats/ROCSVM_Winners.csv") == 0:
        fstat_winner=open("Stats/ROCSVM_Winners.csv", "w")
        fstat_winner.write('Parameter,MWU_P,Max_F1,Min_F1_Range,Max_ARI\n')
        fstat_winner.close()
    
    
    # winners = pd.read_csv("Stats/ROCSVM_Winners.csv")
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
        ocsvm(master_files[FileNumber], parameters, 0)
        break

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

    
    # if MWU_min[index_min] > 2:
    #     print("MWU_min: ", end='')
    #     print(MWU_min)
    #     # break
    # else:
    #     parameters[index_min][1] = f1_range[index_min]
    #     parameters[index_min][2] = [f1_range[index_min]]
    
    #     fstat_winner=open("Stats/ROCSVM_Winners.csv", "a")
    #     fstat_winner.write('\n'+parameters[index_min][0]+','+str(MWU_geo[index_min])+','+str(f1_median[index_min])+','+str(f1_range[index_min])+','+str(ari[index_min])+'\n')
    #     fstat_winner.close()
    
    # print(parameters)
        
        
        