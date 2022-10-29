#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 04:51:08 2022

@author: muyeedahmed
"""


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

datasetFolderDir = 'Dataset/'


def ee(filename, parameters, parameter_iteration, parameter_rankings):
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
    
    # ## Rearrange "IF" and "LOF"
    mod_parameters = deepcopy(parameters)
    percentage_file = pd.read_csv("Stats/SkPercentage.csv")
    lof_cont = percentage_file[percentage_file["Filename"] == filename]["LOF"].to_numpy()[0]
    bisect.insort(mod_parameters[3][2], lof_cont)
    if_cont = percentage_file[percentage_file["Filename"] == filename]["IF"].to_numpy()[0]   
    bisect.insort(mod_parameters[3][2], if_cont)
    
    mod_parameters[3][2][mod_parameters[3][2].index(lof_cont)] = 'LOF'
    mod_parameters[3][2][mod_parameters[3][2].index(if_cont)] = 'IF'
    print(mod_parameters)
    # ##
    
    blind_route = get_blind_route(X, gt, filename, deepcopy(mod_parameters), parameter_iteration, parameter_rankings)
    guided_route = get_guided_route(X, gt, filename, deepcopy(mod_parameters), parameter_iteration, parameter_rankings)
    
    DefaultARI = str(blind_route[0][3][0][1])
    DefaultF1 = str(blind_route[0][3][0][2])
    
    UninformedARI = str(blind_route[-1][3][-1][1])
    UninformedF1 = str(blind_route[-1][3][-1][2])
    
    InformedARI = str(guided_route[-1][3][-1][1])
    InformedF1 = str(guided_route[-1][3][-1][2])
    
    f_Route_Scores=open("Stats/SkEE_Route_Scores.csv", "a")
    f_Route_Scores.write(filename+','+DefaultARI+","+DefaultF1+","+UninformedARI+","+UninformedF1+","+InformedARI+","+InformedF1+"\n")
    f_Route_Scores.close()
    
    
    
def get_blind_route(X, gt, filename, parameters_this_file, parameter_iteration, parameter_rankings):
    blind_route = []
    
    for p_i in range(len(parameters_this_file)):
        p = np.where(parameter_rankings == p_i)
        p = p[0][0]
        
        parameter_route = []
        ari_scores = []
        passing_param = deepcopy(parameters_this_file)

        default_f1, default_ari = runEE(filename, X, gt, passing_param, parameter_iteration)

        parameter_route.append([passing_param[p][1], default_ari, default_f1])
        ari_scores.append(default_ari)
        # print(parameters[p][0], end=': ')
        i_def = passing_param[p][2].index(passing_param[p][1])
        if i_def+1 == len(parameters_this_file[p][2]):
            i_pv = i_def-1    
        else:
            i_pv = i_def+1
        
        while True:
            if i_pv >= len(parameters_this_file[p][2]):
                break
            if i_pv < 0:
                break
            # print(parameters[p][2][i_pv], end = '- ')

            passing_param[p][1] = parameters_this_file[p][2][i_pv]
            f1_score, ari_score = runEE(filename, X, gt, passing_param, parameter_iteration)

            if ari_score > np.max(ari_scores):
                # if i_pv > i_def:
                parameter_route.append([passing_param[p][1], ari_score, f1_score])
                ari_scores.append(ari_score)
            
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
        # print()
        max_index = ari_scores.index(max(ari_scores))
        default_index = ari_scores.index(default_ari)
        parameters_this_file[p][1] = parameter_route[max_index][0]
        blind_route.append([parameters_this_file[p][0], max_index, default_index, parameter_route])
    print(blind_route)
    
    ## Without F1 Score
    fig = plt.Figure()
    start = end = 0
    for i in range(len(blind_route)):
        default = blind_route[i][2]
        start = end
        end = blind_route[i][1] - (default-start)
        
        ari_p = blind_route[i][3]
        # print()
        ari_scores = []
        ari_x = []
        for j in range(len(ari_p)):
            ari_scores.append(ari_p[j][1])
            ari_x.append(j-(default-start))
        for k in range(len(ari_scores)-1):
            plt.plot(ari_x[k+1], ari_scores[k+1], marker=(3, 0, get_angle(ari_x[k+1], ari_scores[k+1], ari_x[k], ari_scores[k])), color='black')
        plt.plot(ari_x, ari_scores, '-')
        
        if i == 0:
            plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][2]][0]), (0.1, blind_route[i][3][blind_route[i][2]][1]), ha='left')

        if start != end:
            plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][1]][0]), (end+0.1, blind_route[i][3][blind_route[i][1]][1]), ha='left')
    # plt.legend(param_names)
    plt.ylabel("Cross-run ARI")
    plt.xticks(ticks= [])
    plt.title(filename)
    plt.savefig("Fig/GD/Routes/"+filename+"_SkEE_Trajectory.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    
    ## With F1 Score
    fig = plt.Figure()
    start = end = 0
    for i in range(len(blind_route)):
        default = blind_route[i][2]
        start = end
        end = blind_route[i][1] - (default-start)
        
        ari_p = blind_route[i][3]
        ari_scores = []
        ari_x = []
        for j in range(len(ari_p)):
            ari_scores.append(ari_p[j][1])
            ari_x.append(ari_p[j][2])
        for k in range(len(ari_scores)-1):
            plt.plot(ari_x[k+1], ari_scores[k+1], marker=(3, 0, get_angle(ari_x[k+1], ari_scores[k+1], ari_x[k], ari_scores[k])), color='black')
        plt.plot(ari_x, ari_scores, '-')
        
        if i == 0:
            plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][2]][0]), (blind_route[i][3][blind_route[i][2]][2], blind_route[i][3][blind_route[i][2]][1]), ha='left')

        if start != end:
            plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][1]][0]), (blind_route[i][3][blind_route[i][1]][2], blind_route[i][3][blind_route[i][1]][1]), ha='left')
    plt.ylabel("Cross-run ARI")
    plt.xlabel("F1 Score")
    
    plt.title(filename)
    plt.savefig("Fig/GD/Routes/"+filename+"_SkEE_Trajectory_W_F1.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    return blind_route
    
def get_guided_route(X, gt, filename, parameters_this_file, parameter_iteration, parameter_rankings):
    guided_route = []
    
    for p_i in range(len(parameters_this_file)):
        p = np.where(parameter_rankings == p_i)
        p = p[0][0]
        
        parameter_route = []
        ari_scores = []
        f1_scores = []
        passing_param = deepcopy(parameters_this_file)

        default_f1, default_ari = runEE(filename, X, gt, passing_param, parameter_iteration)

        parameter_route.append([passing_param[p][1], default_ari, default_f1])
        ari_scores.append(default_ari)
        f1_scores.append(default_f1)
        # print(parameters[p][0], end=': ')
        i_def = passing_param[p][2].index(passing_param[p][1])
        if i_def+1 == len(parameters_this_file[p][2]):
            i_pv = i_def-1    
        else:
            i_pv = i_def+1
        
        while True:
            if i_pv >= len(parameters_this_file[p][2]):
                break
            if i_pv < 0:
                break
            # print(parameters[p][2][i_pv], end = '- ')

            passing_param[p][1] = parameters_this_file[p][2][i_pv]
            f1_score, ari_score = runEE(filename, X, gt, passing_param, parameter_iteration)

            if ari_score > np.max(ari_scores) and f1_score > np.max(f1_scores):
                # if i_pv > i_def:
                parameter_route.append([passing_param[p][1], ari_score, f1_score])
                ari_scores.append(ari_score)
                f1_scores.append(f1_score)
            
            if ari_score != np.max(ari_scores) and f1_score != np.max(f1_scores):
                
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
        max_index = ari_scores.index(max(ari_scores))
        default_index = ari_scores.index(default_ari)
        parameters_this_file[p][1] = parameter_route[max_index][0]
        guided_route.append([parameters_this_file[p][0], max_index, default_index, parameter_route])
    print(guided_route)
    print(guided_route[-1][3][-1][1], guided_route[-1][3][-1][2])
    ## With F1 Score
    fig = plt.Figure()
    start = end = 0
    for i in range(len(guided_route)):
        default = guided_route[i][2]
        start = end
        end = guided_route[i][1] - (default-start)
        
        ari_p = guided_route[i][3]
        ari_scores = []
        ari_x = []
        for j in range(len(ari_p)):
            ari_scores.append(ari_p[j][1])
            ari_x.append(ari_p[j][2])
        for k in range(len(ari_scores)-1):
            plt.plot(ari_x[k+1], ari_scores[k+1], marker=(3, 0, get_angle(ari_x[k+1], ari_scores[k+1], ari_x[k], ari_scores[k])), color='black')
        plt.plot(ari_x, ari_scores, '-')
        
        if i == 0:
            plt.annotate(guided_route[i][0]+" = "+str(guided_route[i][3][guided_route[i][2]][0]), (guided_route[i][3][guided_route[i][2]][2], guided_route[i][3][guided_route[i][2]][1]), ha='left')

        if start != end:
            plt.annotate(guided_route[i][0]+" = "+str(guided_route[i][3][guided_route[i][1]][0]), (guided_route[i][3][guided_route[i][1]][2], guided_route[i][3][guided_route[i][1]][1]), ha='left')
    # plt.legend(param_names)
    plt.ylabel("Cross-run ARI")
    plt.xlabel("F1 Score")
    
    plt.title(filename)
    plt.savefig("Fig/GD/Routes/"+filename+"_SkEE_Guided_Trajectory_W_F1.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()    
    
    return guided_route
    
def get_angle(p1x, p1y, p2x, p2y):
    dx = p2x - p1x
    dy = p2y - p1y
    theta = math.atan2(dy, dx)
    angle = math.degrees(theta)  # angle is in (-180, 180]
    if angle < 0:
        angle = 360 + angle
    return angle

def runEE(filename, X, gt, params, parameter_iteration):

    labelFile = filename + "_" + str(params[0][1]) + "_" + str(params[1][1]) + "_" + str(params[2][1]) + "_" + str(params[3][1])
    
    if os.path.exists("Labels/EE_Sk_Done/Labels_Sk_EE_"+labelFile+".csv") == 1:
        i_store_precision=params[0][1]
        i_assume_centered=params[1][1]
        i_support_fraction=params[2][1]
        i_contamination=params[3][1]
        
        dfari =  pd.read_csv("Stats/SkEE_ARI.csv")
        ari = dfari[(dfari['Filename']==filename)&
                    (dfari['store_precision']==i_store_precision)&
                    (dfari['assume_centered']==i_assume_centered)&
                    (dfari['support_fraction']==str(i_support_fraction))&
                    (dfari['contamination']==str(i_contamination))]
        
        dff1 =  pd.read_csv("Stats/SkEE_F1.csv")
        f1 = dff1[(dff1['Filename']==filename)&
                    (dff1['store_precision']==i_store_precision)&
                    (dff1['assume_centered']==i_assume_centered)&
                    (dff1['support_fraction']==str(i_support_fraction))&
                    (dff1['contamination']==str(i_contamination))]
        if ari.empty == 0 and f1.empty == 0:
            runs_ari = []
            for i in range(45):
                runs_ari.append(('R'+str(i)))
            run_ari_values = ari[runs_ari].to_numpy()
            
            runs_f1 = []
            for i in range(10):
                runs_f1.append(('R'+str(i)))
            run_f1_values = f1[runs_f1].to_numpy()
            
            return np.mean(np.mean(run_f1_values)), np.mean(np.mean(run_ari_values))
  
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
        clustering = EllipticEnvelope(store_precision=sp, assume_centered=ac, 
                                     support_fraction=sf, contamination=cont).fit(X)
    
        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        labels.append(l)
        f1.append(metrics.f1_score(gt, l))
        
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
          ari.append(adjusted_rand_score(labels[i], labels[j]))      
    if os.path.exists("../AnomalyAlgoDiagnosis_Labels/EE_Sk/Labels_Sk_EE_"+labelFile+".csv") == 0:
        fileLabels=open("../AnomalyAlgoDiagnosis_Labels/EE_Sk/Labels_Sk_EE_"+labelFile+".csv", 'a')
        for l in labels:
            fileLabels.write(','.join(str(s) for s in l) + '\n')
        fileLabels.close()
    
    flabel_done=open("Labels/EE_Sk_Done/Labels_Sk_EE_"+labelFile+".csv", 'a')
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

    return np.mean(f1), np.mean(ari)


def plot_ari_f1():
    Route_Scores = pd.read_csv("Stats/SkEE_Route_Scores.csv")
    
    fig = plt.Figure()
    
    plt.plot(Route_Scores["DefaultF1"], Route_Scores["DefaultARI"], '.', color='red', marker = 'd', markersize = 4, alpha=.5)
    plt.plot(Route_Scores["UninformedF1"], Route_Scores["UninformedARI"], '.', color = 'green', marker = 'v', markersize = 4, alpha=.5)
    plt.plot(Route_Scores["InformedF1"], Route_Scores["InformedARI"], '.', color = 'blue', marker = '^', markersize = 4, alpha=.5)
     
    plt.plot(Route_Scores["DefaultF1"].mean(), Route_Scores["DefaultARI"].mean(), '.', color='red', marker = 'd', markersize = 12, markeredgecolor='black', markeredgewidth=1.5)
    plt.plot(Route_Scores["UninformedF1"].mean(), Route_Scores["UninformedARI"].mean(), '.', color = 'green', marker = 'v', markersize = 12, markeredgecolor='black', markeredgewidth=1.5)
    plt.plot(Route_Scores["InformedF1"].mean(), Route_Scores["InformedARI"].mean(), '.', color = 'blue', marker = '^', markersize = 12, markeredgecolor='black', markeredgewidth=1.5)
    
    plt.legend(['Default Setting', 'Uninformed Route', 'Informed Route'])
    plt.title("Scikit-learn - Robust Covariance")
    plt.xlabel("Performance (F1 Score)")
    plt.ylabel("Determinism (ARI)")
    plt.savefig("Fig/SkEE_GD_Comparison.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    
    # ## Calculate Percentage
    
    ui_win_performance = 0
    ui_lose_performance = 0
    ui_win_nd = 0
    i_win_performance = 0
    i_win_nd = 0
    
    
    for index, row in Route_Scores.iterrows():
        if row["UninformedF1"] > row["DefaultF1"]:
            ui_win_performance += 1
        elif row["UninformedF1"] < row["DefaultF1"]:
            ui_lose_performance += 1
        if row["UninformedARI"] > row["DefaultARI"]:
            ui_win_nd += 1
    
        if row["InformedF1"] > row["DefaultF1"]:
            i_win_performance += 1
        if row["InformedARI"] > row["DefaultARI"]:
            i_win_nd += 1

    print(f"Default & {Route_Scores['DefaultARI'].mean()} & -  & -  & {Route_Scores['DefaultF1'].mean()}  & -  & -  \\\\ \\hline")
    print(f"Uninformed & {Route_Scores['UninformedARI'].mean()} & {ui_win_nd}  & 0  & {Route_Scores['UninformedF1'].mean()}  & {ui_win_performance}  & {ui_lose_performance} \\\\ ")
    print(f"Informed & {Route_Scores['InformedARI'].mean()} & {i_win_nd}  & 0  & {Route_Scores['InformedF1'].mean()}  & {i_win_performance}  & 0 \\\\ ")
    
    
    
    
if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
    
    # if os.path.exists("Stats/SkEE_F1.csv") == 0: 
    #     R = ""
    #     for i in range(9):
    #         R += "R"+str(i)+","
    #     R+="R9"
    #     fstat_f1=open("Stats/SkEE_F1.csv", "w")
    #     fstat_f1.write('Filename,store_precision,assume_centered,support_fraction,contamination,Parameter_Iteration,'+R+"\n")
    #     fstat_f1.close()

    # if os.path.exists("Stats/SkEE_ARI.csv") == 0:    
    #     ARI_R = ""
    #     for i in range(44):
    #         ARI_R += "R"+str(i)+","
    #     ARI_R+="R44"
    #     fstat_ari=open("Stats/SkEE_ARI.csv", "w")
    #     fstat_ari.write('Filename,store_precision,assume_centered,support_fraction,contamination,Parameter_Iteration,'+ARI_R+"\n")
    #     fstat_ari.close()
        
    parameters = []
    store_precision = [True, False]
    assume_centered = [True, False]
    support_fraction = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    contamination = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    parameters.append(["store_precision", True, store_precision])
    parameters.append(["assume_centered", False, assume_centered])
    parameters.append(["support_fraction", None, support_fraction])
    parameters.append(["contamination", 0.1, contamination])
    
    
    f_Route_Scores=open("Stats/SkEE_Route_Scores.csv", "w")
    f_Route_Scores.write('Filename,DefaultARI,DefaultF1,UninformedARI,UninformedF1,InformedARI,InformedF1\n')
    f_Route_Scores.close()

    ## Ranking
    parameter_rankings = pd.read_csv("Mann–Whitney U test/MWU_SkEE_Ranking.csv")
    parameter_rankings["Ranking"] = (ss.rankdata(parameter_rankings["MWU_Score"])-1)
    for i in range(len(parameters)):
        param_rank = parameter_rankings[parameter_rankings["ParameterName"] == parameters[i][0]]["Ranking"].to_numpy()
        parameters[i].append(int(param_rank[0]))
    ##
        
    for FileNumber in range(len(master_files)):
        print(FileNumber, end=' ')
        ee(master_files[FileNumber], parameters, 0, parameter_rankings["Ranking"].to_numpy())

    plot_ari_f1() 
        
        
        