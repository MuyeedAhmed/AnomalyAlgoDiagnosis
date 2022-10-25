#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 17:52:36 2022

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
# import pingouin as pg
import random
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
# import scikit_posthocs as sp
import scipy.stats as stats
from scipy.stats import gmean
import math

datasetFolderDir = 'Dataset/'


def isolationforest(filename, parameters, parameter_iteration, parameter_rankings):
    print(filename)
    folderpath = datasetFolderDir
    
    parameters_this_file = deepcopy(parameters)
    
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
    
    ari_total = []
    
    for p_i in range(len(parameters_this_file)):
        p = np.where(parameter_rankings == p_i)
        p = p[0][0]
    #     continue
    # return
    # if True:
        ari_param = []
        ari_scores = []
        passing_param = deepcopy(parameters_this_file)

        default_ari = runIF(filename, X, gt, passing_param, parameter_iteration)

        ari_param.append([passing_param[p][1], default_ari])
        ari_scores.append(default_ari)
        # print(parameters[p][0], end=': ')
        i_def = passing_param[p][2].index(passing_param[p][1])
        if i_def+1 == len(parameters_this_file[p][2]):
            i_pv = i_def-1    
        else:
            i_pv = i_def+1
        
        while True:
            print(ari_param)
            if i_pv >= len(parameters_this_file[p][2]):
                break
            if i_pv < 0:
                break
            # print(parameters[p][2][i_pv], end = '- ')
            
            
            passing_param[p][1] = parameters_this_file[p][2][i_pv]
            ari_score = runIF(filename, X, gt, passing_param, parameter_iteration)
            if i_pv > i_def:
                ari_param.append([passing_param[p][1], ari_score])
                ari_scores.append(ari_score)
            else:
                ari_param.insert(0, [passing_param[p][1], ari_score])
                ari_scores.insert(0, ari_score)
            
            if ari_score != np.max(ari_scores):
                
                if i_pv - 1 > i_def:
                    break
                elif i_pv - 1 == i_def:
                    i_pv = i_def - 1
                else:
                    break
            else:
                if i_pv > i_def:
                    i_pv += 1
                else:
                    i_pv -= 1
        print()
        max_index = ari_scores.index(max(ari_scores))
        default_index = ari_scores.index(default_ari)
        parameters_this_file[p][1] = ari_param[max_index][0]
        ari_total.append([parameters_this_file[p][0], max_index, default_index, ari_param])
    print(ari_total)
    param_names = []
    for i in range(len(ari_total)):
        param_names.append(ari_total[i][0])
    fig = plt.Figure()
    start = end = 0
    for i in range(len(ari_total)):
        default = ari_total[i][2]
        start = end
        end = ari_total[i][1] - (default-start)
        
        ari_p = ari_total[i][3]
        # print()
        ari_scores = []
        ari_x = []
        for j in range(len(ari_p)):
            ari_scores.append(ari_p[j][1])
            ari_x.append(j-(default-start))
        # print(ari_scores, ari_x)
        plt.plot(ari_x, ari_scores, 'o-')
    
    plt.legend(param_names)
    
    plt.savefig("Fig/GD/SkIF"+filename+".pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
def runIF(filename, X, gt, params, parameter_iteration):

    labelFile = filename + "_" + str(params[0][1]) + "_" + str(params[1][1]) + "_" + str(params[2][1]) + "_" + str(params[3][1]) + "_" + str(params[4][1]) + "_" + str(params[5][1]) + "_" + str(params[6][1])

    if os.path.exists("IF/Labels_Sk_IF_"+labelFile+".csv") == 1:
        
        dfari =  pd.read_csv("Stats/SkIF_ARI.csv")
        i_n_estimators=params[0][1]
        i_max_samples=params[1][1]
        i_contamination=params[2][1]
        i_max_features=params[3][1]
        i_bootstrap=params[4][1]
        i_n_jobs=params[5][1]
        i_warm_start = params[6][1]
        ari = dfari[(dfari['Filename']==filename)&
                    (dfari['n_estimators']==i_n_estimators)&
                    (dfari['max_samples']==str(i_max_samples))&
                    (dfari['max_features']==i_max_features)&
                    (dfari['bootstrap']==i_bootstrap)&
                    (dfari['n_jobs']==str(i_n_jobs))&
                    (dfari['warm_start']==i_warm_start)]
        if ari.empty == 0:
            runs_ari = []
            for i in range(45):
                runs_ari.append(('R'+str(i)))
            run_values = ari[runs_ari].to_numpy()
               
            return np.mean(np.mean(run_values))
  
    # return 0
        
    
    labels = []
    f1 = []
    ari = []
    for i in range(10):
        clustering = IsolationForest(n_estimators=params[0][1], max_samples=params[1][1], 
                                      max_features=params[3][1], bootstrap=params[4][1], 
                                      n_jobs=params[5][1], warm_start=params[6][1]).fit(X)
    
        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        labels.append(l)
        f1.append(metrics.f1_score(gt, l))
        
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
          ari.append(adjusted_rand_score(labels[i], labels[j]))      
    if os.path.exists("../AnomalyAlgoDiagnosis_Labels/Labels_Sk_IF_"+labelFile+".csv") == 0:
        fileLabels=open("../AnomalyAlgoDiagnosis_Labels/Labels_Sk_IF_"+labelFile+".csv", 'a')
        for l in labels:
            fileLabels.write(','.join(str(s) for s in l) + '\n')
        fileLabels.close()
    
    flabel_done=open("IF/Labels_Sk_IF_"+labelFile+".csv", 'a')
    flabel_done.write("Done")
    flabel_done.close()
    
    fstat_f1=open("Stats/SkIF_F1.csv", "a")
    fstat_f1.write(filename+','+ str(params[0][1]) + ','+ str(params[1][1]) + ',' + str(params[2][1]) + ',' + str(params[3][1]) + ',' + str(params[4][1]) + ',' + str(params[5][1]) + ',' + str(params[6][1]) + ',' + str(parameter_iteration) + ',')
    fstat_f1.write(','.join(str(s) for s in f1) + '\n')
    fstat_f1.close()
    
    fstat_ari=open("Stats/SkIF_ARI.csv", "a")
    fstat_ari.write(filename+','+ str(params[0][1]) + ','+ str(params[1][1]) + ',' + str(params[2][1]) + ',' + str(params[3][1]) + ',' + str(params[4][1]) + ',' + str(params[5][1]) + ',' + str(params[6][1]) + ',' + str(parameter_iteration) + ',')
    fstat_ari.write(','.join(str(s) for s in ari) + '\n')
    fstat_ari.close()
    
    return np.mean(ari)

def calculate_score(allFiles, parameter, parameter_values, all_parameters, p_iter):
    i_n_estimators=all_parameters[0][1]
    i_max_samples=all_parameters[1][1]
    i_contamination=all_parameters[2][1]
    i_max_features=all_parameters[3][1]
    i_bootstrap=all_parameters[4][1]
    i_n_jobs=all_parameters[5][1]
    i_warm_start = all_parameters[6][1]
    
    # dfacc = pd.read_csv("Stats/SkIF_Accuracy.csv")
    dff1 = pd.read_csv("Stats/SkIF_F1.csv")
    dfari =  pd.read_csv("Stats/SkIF_ARI.csv")
    
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
                        
            # accuracy = dfacc[(dfacc['Filename']==filename)&
            #                 (dfacc['n_estimators']==i_n_estimators)&
            #                 (dfacc['max_samples']==str(i_max_samples))&
            #                 (dfacc['max_features']==i_max_features)&
            #                 (dfacc['bootstrap']==i_bootstrap)&
            #                 (dfacc['n_jobs']==str(i_n_jobs))&
            #                 (dfacc['warm_start']==i_warm_start)]
                

            f1 = dff1[(dff1['Filename']==filename)&
                            (dff1['n_estimators']==i_n_estimators)&
                            (dff1['max_samples']==str(i_max_samples))&
                            (dff1['max_features']==i_max_features)&
                            (dff1['bootstrap']==i_bootstrap)&
                            (dff1['n_jobs']==str(i_n_jobs))&
                            (dff1['warm_start']==i_warm_start)]
            if f1.empty:
                continue
            ari = dfari[(dfari['Filename']==filename)&
                            (dfari['n_estimators']==i_n_estimators)&
                            (dfari['max_samples']==str(i_max_samples))&
                            (dfari['max_features']==i_max_features)&
                            (dfari['bootstrap']==i_bootstrap)&
                            (dfari['n_jobs']==str(i_n_jobs))&
                            (dfari['warm_start']==i_warm_start)]
            
            # accuracy_values = accuracy[runs].to_numpy()[0]
            
            f1_values = f1[f1_runs].to_numpy()[0]
            ari_values = ari[ari_runs].to_numpy()[0]
            
            # accDiff = (np.percentile(accuracy_values, 75) - np.percentile(accuracy_values, 25))/(np.percentile(accuracy_values, 75) + np.percentile(accuracy_values, 25))
            # accuracy_range_all.append([filename, p, accDiff])
            # accuracy_med_all.append([filename, p, np.percentile(accuracy_values, 50)])
            
            f1Diff = (np.percentile(f1_values, 75) - np.percentile(f1_values, 25))/(np.percentile(f1_values, 75) + np.percentile(f1_values, 25))
            if math.isnan(f1Diff):
                f1Diff = 0
            f1_range_all.append([filename, p, f1Diff])
            f1_med_all.append([filename, p, np.percentile(f1_values, 50)])
            
            ari_all.append([filename, p, np.percentile(ari_values, 50)])
            
    # df_acc_r = pd.DataFrame(accuracy_range_all, columns = ['Filename', parameter, 'Accuracy_Range'])
    # df_acc_m = pd.DataFrame(accuracy_med_all, columns = ['Filename', parameter, 'Accuracy_Median'])
    df_f1_r = pd.DataFrame(f1_range_all, columns = ['Filename', parameter, 'F1Score_Range'])
    df_f1_m = pd.DataFrame(f1_med_all, columns = ['Filename', parameter, 'F1Score_Median'])
    df_ari_m = pd.DataFrame(ari_all, columns = ['Filename', parameter, 'ARI'])

    if parameter == 'n_jobs':
        df_f1_r = df_f1_r.fillna(0)
        df_f1_m = df_f1_m.fillna(0)
        df_ari_m = df_ari_m.fillna(0)
        df_f1_r[parameter] = df_f1_r[parameter].astype(int)
        df_f1_m[parameter] = df_f1_m[parameter].astype(int)
        df_ari_m[parameter] = df_ari_m[parameter].astype(int)
        parameter_values = [1, 0]


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
    mwu_df_f1_range.to_csv("Mann–Whitney U test/MWU_SkIF_F1_Range_"+parameter+"_"+str(p_iter)+".csv")
    
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
    
    # R = ""
    # for i in range(9):
    #     R += "R"+str(i)+","
    # R+="R9"
    # ARI_R = ""
    # for i in range(44):
    #     ARI_R += "R"+str(i)+","
    # ARI_R+="R44"

        
    # if os.path.exists("Stats/SkIF_F1.csv") == 0: 
    #     fstat_f1=open("Stats/SkIF_F1.csv", "w")
    #     fstat_f1.write('Filename,n_estimators,max_samples,contamination,max_features,bootstrap,n_jobs,warm_start,Parameter_Iteration,'+R+"\n")
    #     fstat_f1.close()

    # if os.path.exists("Stats/SkIF_ARI.csv") == 0:    
    #     fstat_ari=open("Stats/SkIF_ARI.csv", "w")
    #     fstat_ari.write('Filename,n_estimators,max_samples,contamination,max_features,bootstrap,n_jobs,warm_start,Parameter_Iteration,'+ARI_R+"\n")
    #     fstat_ari.close()
    # if os.path.exists("Stats/SkIF_Winners.csv") == 0:  
    #     fstat_winner=open("Stats/SkIF_Winners.csv", "w")
    #     fstat_winner.write('Parameter,MWU_P,Max_F1,Min_F1_Range,Max_ARI\n')
    #     fstat_winner.close()
    
    
    # for param_iteration in range(len(parameters)):
    
    parameter_rankings = pd.read_csv("Mann–Whitney U test/MWU_SkIF_Ranking.csv")
    import scipy.stats as ss
    parameter_rankings["Ranking"] = (ss.rankdata(parameter_rankings["MWU_Score"])-1)
    # print(type(parameter_rankings["Ranking"].to_numpy()))
    # print(np.array(parameter_rankings["Ranking"]))
    for i in range(len(parameters)):
        param_rank = parameter_rankings[parameter_rankings["ParameterName"] == parameters[i][0]]["Ranking"].to_numpy()
        parameters[i].append(int(param_rank[0]))
    # print(parameters)
    for FileNumber in range(len(master_files)):
        print(FileNumber, end=' ')
        isolationforest(master_files[FileNumber], parameters, 0, parameter_rankings["Ranking"].to_numpy())
        if FileNumber == 5:
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

        # if index_min == 5 and f1_range[index_min] == 0:
        #     f1_range[index_min] = None
        # if MWU_min[index_min] > 2:
        #     print("MWU_min: ", end='')
        #     print(MWU_min)
        #     break
        # parameters[index_min][1] = f1_range[index_min]
        # parameters[index_min][2] = [f1_range[index_min]]
        
        # fstat_winner=open("Stats/SkIF_Winners.csv", "a")
        # fstat_winner.write(parameters[index_min][0]+','+str(MWU_geo[index_min])+','+str(f1_median[index_min])+','+str(f1_range[index_min])+','+str(ari[index_min])+'\n')
        # fstat_winner.close()
        
        # print(parameters)        
        
        
        
        