#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 04:44:18 2022

@author: muyeedahmed
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 21:24:10 2022

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


def ee(filename, parameters_mat, parameters_sk):
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
    
    DefaultARI, DefaultF1_sk, DefaultF1_mat = runEE(filename, X, gt, parameters_sk, parameters_mat)
    
    # ## Rearrange "IF" and "LOF"
    mod_parameters_mat = deepcopy(parameters_mat)
    mod_parameters_sk = deepcopy(parameters_sk)
    percentage_file = pd.read_csv("Stats/SkPercentage.csv")
    lof_cont = percentage_file[percentage_file["Filename"] == filename]["LOF"].to_numpy()[0]
    bisect.insort(mod_parameters_sk[3][2], lof_cont)
    if_cont = percentage_file[percentage_file["Filename"] == filename]["IF"].to_numpy()[0]   
    bisect.insort(mod_parameters_sk[3][2], if_cont)
    
    mod_parameters_sk[3][2][mod_parameters_sk[3][2].index(lof_cont)] = 'LOF'
    mod_parameters_sk[3][2][mod_parameters_sk[3][2].index(if_cont)] = 'IF'
    # ##
    
    # blind_route_sk, blind_route_mat, blind = get_blind_route(X, gt, filename, deepcopy(mod_parameters_sk), deepcopy(mod_parameters_mat))
    # informed_route_sk, informed_route_mat = get_informed_route(X, gt, filename, deepcopy(mod_parameters_sk), deepcopy(mod_parameters_mat))
    
    # print(blind_route_sk)
    # print()
    # print(blind_route_mat)
    # print()
    # print(informed_route_sk)
    # print()
    # print(informed_route_mat)
    
    
    blind = [['Matlab_Method', 1, 0, [['fmcd', 0.06970256081940562, 0.0, 0.09886815212133457], ['ogk', 0.5884146150541621, 0.0, 0.027027027027027]]], 
             ['Matlab_NumOGKIterations', 1, 0, [[2, 0.5884146150541621, 0.0, 0.027027027027027], [3, 0.686269153780207, 0.0, 0.029411764705882297]]], 
             
             ['Sklearn_assume_centered', 1, 0, [[False, 0.686269153780207, 0.0, 0.029411764705882297], [True, 0.7099423110825727, 0.0, 0.029411764705882297]]], 
             ['Sklearn_contamination', 1, 0, [['IF', 0.7099423110825727, 0.0, 0.029411764705882297], ['IF', 0.7354250701081788, 0.0256410256410256, 0.029411764705882297]]], 
             ['Matlab_NumOGKIterations', 1, 0, [[3, 0.7354250701081788, 0.0256410256410256, 0.029411764705882297], [2, 0.8490755852376717, 0.0256410256410256, 0.027027027027027]]], 
             ['Matlab_UnivariateEstimator', 1, 0, [['tauscale', 0.8490755852376717, 0.0256410256410256, 0.027027027027027], ['qn', 0.8655554078622951, 0.0256410256410256, 0.028169014084506998]]], 
             
             ['Sklearn_contamination', 0, 0, [[0.2, 0.8655554078622951, 0.0256410256410256, 0.028169014084506998]]]]
     
    
    informed_route_sk = [['store_precision', 0, 0, [[True, 0.09427886575351131, 0.0, 0.11456723819010786]]], 
                          ['assume_centered', 0, 0, [[False, 0.09427886575351131, 0.0, 0.11456723819010786]]], 
                          ['support_fraction', 1, 0, [[None, 0.09427886575351131, 0.0, 0.11456723819010786], [0.1, 0.1014925467129616, 0.0037037037037037, 0.11456723819010786]]], 
                          ['contamination', 0, 0, [[0.1, 0.1014925467129616, 0.0037037037037037, 0.11456723819010786]]]]
    
    # informed_route_mat = [['Method', 0, 0, [['fmcd', 0.06970256081940562, 0.0, 0.09886815212133457]]], ['OutlierFraction', 0, 0, [[0.5, 0.06970256081940562, 0.0, 0.09886815212133457]]], ['NumTrials', 0, 0, [[500, 0.06970256081940562, 0.0, 0.09886815212133457]]], ['BiasCorrection', 1, 0, [[1, 0.06970256081940562, 0.0, 0.09886815212133457], [0, 0.09427886575351131, 0.0, 0.11456723819010786]]], ['Method', 0, 0, [['fmcd', 0.09427886575351131, 0.0, 0.11456723819010786]]], ['OutlierFraction', 0, 0, [[0.5, 0.09427886575351131, 0.0, 0.11456723819010786]]], ['NumTrials', 0, 0, [[500, 0.09427886575351131, 0.0, 0.11456723819010786]]], ['BiasCorrection', 0, 0, [[0, 0.09427886575351131, 0.0, 0.11456723819010786]]], ['Method', 0, 0, [['fmcd', 0.09427886575351131, 0.0, 0.11456723819010786]]], ['OutlierFraction', 0, 0, [[0.5, 0.09427886575351131, 0.0, 0.11456723819010786]]], ['NumTrials', 0, 0, [[500, 0.09427886575351131, 0.0, 0.11456723819010786]]], ['BiasCorrection', 0, 0, [[0, 0.09427886575351131, 0.0, 0.11456723819010786]]], ['Method', 0, 0, [['fmcd', 0.09427886575351131, 0.0, 0.11456723819010786]]], ['OutlierFraction', 0, 0, [[0.5, 0.09427886575351131, 0.0, 0.11456723819010786]]], ['NumTrials', 0, 0, [[500, 0.09427886575351131, 0.0, 0.11456723819010786]]], ['BiasCorrection', 0, 0, [[0, 0.09427886575351131, 0.0, 0.11456723819010786]]], ['Method', 0, 0, [['fmcd', 0.08482566199042128, 0.0, 0.11456723819010786]]], ['OutlierFraction', 0, 0, [[0.5, 0.08482566199042128, 0.0, 0.11456723819010786]]], ['NumTrials', 0, 0, [[500, 0.08482566199042128, 0.0, 0.11456723819010786]]], ['BiasCorrection', 0, 0, [[0, 0.08482566199042128, 0.0, 0.11456723819010786]]], ['Method', 0, 0, [['fmcd', 0.1014925467129616, 0.0037037037037037, 0.11456723819010786]]], ['OutlierFraction', 0, 0, [[0.5, 0.1014925467129616, 0.0037037037037037, 0.11456723819010786]]], ['NumTrials', 0, 0, [[500, 0.1014925467129616, 0.0037037037037037, 0.11456723819010786]]], ['BiasCorrection', 0, 0, [[0, 0.1014925467129616, 0.0037037037037037, 0.11456723819010786]]], ['Method', 0, 0, [['fmcd', 0.1014925467129616, 0.0037037037037037, 0.11456723819010786]]], ['OutlierFraction', 0, 0, [[0.5, 0.1014925467129616, 0.0037037037037037, 0.11456723819010786]]], ['NumTrials', 0, 0, [[500, 0.1014925467129616, 0.0037037037037037, 0.11456723819010786]]], ['BiasCorrection', 0, 0, [[0, 0.1014925467129616, 0.0037037037037037, 0.11456723819010786]]]]
    # print(blind)
    draw_graph(filename, blind,informed_route_sk, 'S')
    # draw_graph(filename, blind_route_mat,informed_route_mat, 'M')
    
    
def draw_graph(filename, blind_route, informed_route, tool):
    ## Without F1 Score
    fig = plt.Figure()
    plt.rcParams['figure.figsize'] = [10, 8]
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
        plt.plot(ari_x, ari_scores, '-')
        for k in range(len(ari_scores)-1):
            plt.plot(ari_x[k+1], ari_scores[k+1], marker=(3, 0, get_angle(ari_x[k], ari_scores[k], ari_x[k+1], ari_scores[k+1])-90), color='black', markersize=8)
            
        
        if i == 0:
            plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][2]][0]), (0.1, blind_route[i][3][blind_route[i][2]][1]), ha='left', fontsize=15)

        if start != end:
            if blind_route[i][0] == "Matlab_Method":
                plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][1]][0]), (end+0.1, blind_route[i][3][blind_route[i][1]][1]-0.02), ha='left', fontsize=15)
            if blind_route[i][0] == "Matlab_NumOGKIterations" and end == 2:
                plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][1]][0]), (end, blind_route[i][3][blind_route[i][1]][1]-0.04), ha='left', fontsize=15)
            if blind_route[i][0] == "Sklearn_assume_centered":
                plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][1]][0]), (end+0.1, blind_route[i][3][blind_route[i][1]][1]-0.03), ha='left', fontsize=15)
            if blind_route[i][0] == "Sklearn_contamination":
                plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][1]][0]), (end+0.1, blind_route[i][3][blind_route[i][1]][1]-0.02), ha='left', fontsize=15)
            if blind_route[i][0] == "Matlab_NumOGKIterations" and end == 5:
                plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][1]][0]), (end-0.1, blind_route[i][3][blind_route[i][1]][1]-0.01), ha='right', fontsize=15)
            if blind_route[i][0] == "Matlab_UnivariateEstimator":
                plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][1]][0]), (end-0.1, blind_route[i][3][blind_route[i][1]][1]+0.01), ha='right', fontsize=15)
    
            # plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][1]][0]), (end+0.1, blind_route[i][3][blind_route[i][1]][1]), ha='left')
    # plt.legend(param_names)
    plt.ylabel("Mutual ARI", fontsize=15)
    plt.xticks([], fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.title(filename, fontsize = 15)
    plt.savefig("Fig/GD/Routes_Inconsistency/"+filename+"_EE_SvM_"+tool+"_Trajectory.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    # ## With F1 Score
    # fig = plt.Figure()
    # start = end = 0
    # for i in range(len(blind_route)):
    #     default = blind_route[i][2]
    #     start = end
    #     end = blind_route[i][1] - (default-start)
        
    #     ari_p = blind_route[i][3]
    #     ari_scores = []
    #     ari_x = []
    #     for j in range(len(ari_p)):
    #         print(ari_p)
    #         ari_scores.append(ari_p[j][1])
    #         if tool == 'S':
    #             ari_x.append(ari_p[j][2])
    #         elif tool == 'M':
    #             ari_x.append(ari_p[j][3])
    #     plt.plot(ari_x, ari_scores, '-')
    #     for k in range(len(ari_scores)-1):
    #         plt.plot(ari_x[k+1], ari_scores[k+1], marker=(3, 0, get_angle(ari_x[k], ari_scores[k], ari_x[k+1], ari_scores[k+1])-90), color='black', markersize=8)
        
    #     if i == 0:
    #         plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][2]][0]), (blind_route[i][3][blind_route[i][2]][2], blind_route[i][3][blind_route[i][2]][1]), ha='left')

    #     if start != end:
            
    #         plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][1]][0]), (blind_route[i][3][blind_route[i][1]][2], blind_route[i][3][blind_route[i][1]][1]), ha='left')
    # plt.ylabel("Cross-run ARI")
    # plt.xlabel("F1 Score")
    
    # plt.title(filename)
    # plt.savefig("Fig/GD/Routes_Inconsistency/"+filename+"_EE_SvM_"+tool+"_Trajectory_W_F1.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()
    
    # ## Informed F1 Score
    # fig = plt.Figure()
    # start = end = 0
    # for i in range(len(informed_route)):
    #     default = informed_route[i][2]
    #     start = end
    #     end = informed_route[i][1] - (default-start)
        
    #     ari_p = informed_route[i][3]
    #     ari_scores = []
    #     ari_x = []
    #     for j in range(len(ari_p)):
    #         ari_scores.append(ari_p[j][1])
    #         if tool == 'S':
    #             ari_x.append(ari_p[j][2])
    #         elif tool == 'M':
    #             ari_x.append(ari_p[j][3])
    #     plt.plot(ari_x, ari_scores, '-')
        
    #     for k in range(len(ari_scores)-1):
    #         plt.plot(ari_x[k+1], ari_scores[k+1], marker=(3, 0, get_angle(ari_x[k], ari_scores[k], ari_x[k+1], ari_scores[k+1])-90), color='black', markersize=8)

        
    #     if i == 0:
    #         plt.annotate(informed_route[i][0]+" = "+str(informed_route[i][3][informed_route[i][2]][0]), (informed_route[i][3][informed_route[i][2]][2], informed_route[i][3][informed_route[i][2]][1]), ha='left')

    #     if start != end:
    #         plt.annotate(informed_route[i][0]+" = "+str(informed_route[i][3][informed_route[i][1]][0]), (informed_route[i][3][informed_route[i][1]][2], informed_route[i][3][informed_route[i][1]][1]), ha='left')
    # plt.ylabel("Cross-run ARI")
    # plt.xlabel("F1 Score")
    
    # plt.title(filename)
    # plt.savefig("Fig/GD/Routes_Inconsistency/"+filename+"_EE_SvM_"+tool+"_Guided_Trajectory_W_F1.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()
    
def get_angle(p1x, p1y, p2x, p2y):
    dx = p2x - p1x
    dy = p2y - p1y
    theta = math.atan2(dy, dx)
    angle = math.degrees(theta)  # angle is in (-180, 180]
    if angle < 0:
        angle = 360 + angle
    return angle

def get_blind_route(X, gt, filename, paramaters_sk_copy,paramaters_mat_copy_pre):
    blind = []
    blind_route_sk = []
    blind_route_mat = []
    route_temp = []
    for p_sk in range(len(paramaters_sk_copy)):
        parameter_route_sk = []
        ari_scores_sk = []
        passing_param_sk = deepcopy(paramaters_sk_copy)

        default_ari, default_f1_sk, default_f1_mat, route_mat = get_blind_route_mat(X, gt, filename, passing_param_sk, paramaters_mat_copy_pre)
        if default_ari == -1:
            return [], []
        parameter_route_sk.append([passing_param_sk[p_sk][1], default_ari, default_f1_sk, default_f1_mat])
        ari_scores_sk.append(default_ari)
        blind_route_mat += route_mat
        blind += route_mat
        i_def = passing_param_sk[p_sk][2].index(passing_param_sk[p_sk][1])
        if i_def+1 == len(passing_param_sk[p_sk][2]):
            i_pv_sk = i_def-1    
        else:
            i_pv_sk = i_def+1
        
        while True:
            if i_pv_sk >= len(paramaters_sk_copy[p_sk][2]):
                break
            if i_pv_sk < 0:
                break

            passing_param_sk[p_sk][1] = paramaters_sk_copy[p_sk][2][i_pv_sk]
            
            ari_score, f1_score_sk, f1_score_mat, route_mat = get_blind_route_mat(X, gt, filename, passing_param_sk, paramaters_mat_copy_pre)
            route_temp += route_mat
            blind += route_mat
            if ari_score >= np.max(ari_scores_sk):
                parameter_route_sk.append([passing_param_sk[p_sk][1], ari_score, f1_score_sk, f1_score_mat])
                ari_scores_sk.append(ari_score)
                blind.append(["Sklearn_"+paramaters_sk_copy[p_sk][0], -1,-1, [passing_param_sk[p_sk][1], ari_score, f1_score_sk, f1_score_mat]])
                # print(route_mat)
                blind_route_mat += route_temp
                route_temp = []
            
            if ari_score != np.max(ari_scores_sk):
                
                if i_pv_sk - 1 > i_def:
                    break
                elif i_pv_sk - 1 == i_def:
                    i_pv_sk = i_def - 1
                else:
                    break
            else:
                if i_pv_sk > i_def:
                    i_pv_sk += 1
                else:
                    i_pv_sk -= 1
        # print()
        ari_scores_sk = np.array(ari_scores_sk)
        max_index = np.where(ari_scores_sk == max(ari_scores_sk))[0][-1]
        
        default_index = np.where(ari_scores_sk == default_ari)[0][0]
        # default_index = ari_scores_sk.index(default_ari)
        paramaters_sk_copy[p_sk][1] = parameter_route_sk[max_index][0]
        blind_route_sk.append(["Sklearn_"+paramaters_sk_copy[p_sk][0], max_index, default_index, parameter_route_sk])
    # print("Sklearn")
    # print(blind_route_sk)
    return blind_route_sk, blind_route_mat, blind
    
    
def get_blind_route_mat(X, gt, filename, passing_param_sk, paramaters_mat_copy):
    blind_route = []
    
    for p in range(len(paramaters_mat_copy)):
        
        if paramaters_mat_copy[0][1] == "fmcd" and p != 0:
            if p == 4 or p == 5 or p == 6 or p == 7 or p == 8:
                continue
            
        if paramaters_mat_copy[0][1] == "ogk" and p != 0:
            if p == 1 or p == 2 or p == 3 or p==6 or p == 7 or p == 8:
                continue
        
        if paramaters_mat_copy[0][1] == "olivehawkins" and p != 0:
            if p == 3 or p == 4 or p == 5:
                continue
        
        parameter_route = []
        ari_scores = []
        passing_param = deepcopy(paramaters_mat_copy)

        default_ari, default_f1_sk, default_f1_mat = runEE(filename, X, gt, passing_param_sk, passing_param)
        if default_ari == -1:
            return -1, -1, -1, []
        parameter_route.append([passing_param[p][1], default_ari, default_f1_sk, default_f1_mat])
        ari_scores.append(default_ari)
        i_def = passing_param[p][2].index(passing_param[p][1])
        if i_def+1 == len(paramaters_mat_copy[p][2]):
            i_pv = i_def-1    
        else:
            i_pv = i_def+1
        
        while True:
            if i_pv >= len(paramaters_mat_copy[p][2]):
                break
            if i_pv < 0:
                break

            passing_param[p][1] = paramaters_mat_copy[p][2][i_pv]
            ari_score, f1_score_sk, f1_score_mat = runEE(filename, X, gt, passing_param_sk, passing_param)
            if ari_score >= np.max(ari_scores):
                parameter_route.append([passing_param[p][1], ari_score, f1_score_sk, f1_score_mat])
                ari_scores.append(ari_score)
            if np.max(ari_scores) == 1:
                break
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
        ari_scores = np.array(ari_scores)
        max_index = np.where(ari_scores == max(ari_scores))[0][-1]
        default_index = np.where(ari_scores == default_ari)[0][0]
        # default_index = ari_scores.index(default_ari)
        paramaters_mat_copy[p][1] = parameter_route[max_index][0]
        blind_route.append([paramaters_mat_copy[p][0], max_index, default_index, parameter_route])
    # print("Matlab")
    # print(blind_route)
    # print()
    return blind_route[-1][3][-1][1], blind_route[-1][3][-1][2], blind_route[-1][3][-1][3], blind_route


def get_informed_route(X, gt, filename, paramaters_sk_copy,paramaters_mat_copy_pre):
    informed_route_sk = []
    informed_route_mat = []
    route_temp = []
    for p_sk in range(len(paramaters_sk_copy)):
        parameter_route_sk = []
        ari_scores_sk = []
        f1_scores_sk = []
        passing_param_sk = deepcopy(paramaters_sk_copy)

        default_ari, default_f1_sk, default_f1_mat, route_mat = get_informed_route_mat(X, gt, filename, passing_param_sk, paramaters_mat_copy_pre)
        if default_ari == -1:
            return [], []
        parameter_route_sk.append([passing_param_sk[p_sk][1], default_ari, default_f1_sk, default_f1_mat])
        ari_scores_sk.append(default_ari)
        f1_scores_sk.append(default_f1_sk)
        informed_route_mat += route_mat
        
        i_def = passing_param_sk[p_sk][2].index(passing_param_sk[p_sk][1])
        if i_def+1 == len(passing_param_sk[p_sk][2]):
            i_pv_sk = i_def-1    
        else:
            i_pv_sk = i_def+1
        
        while True:
            if i_pv_sk >= len(paramaters_sk_copy[p_sk][2]):
                break
            if i_pv_sk < 0:
                break

            passing_param_sk[p_sk][1] = paramaters_sk_copy[p_sk][2][i_pv_sk]
            
            
            ari_score, f1_score_sk, f1_score_mat, route_mat = get_informed_route_mat(X, gt, filename, passing_param_sk, paramaters_mat_copy_pre)
            route_temp += route_mat
            if ari_score >= np.max(ari_scores_sk) and f1_score_sk >= np.max(f1_scores_sk):
                parameter_route_sk.append([passing_param_sk[p_sk][1], ari_score, f1_score_sk, f1_score_mat])
                ari_scores_sk.append(ari_score)
                f1_scores_sk.append(f1_score_sk)
                informed_route_mat += route_temp
                route_temp = []
            if ari_score != np.max(ari_scores_sk):
                
                if i_pv_sk - 1 > i_def:
                    break
                elif i_pv_sk - 1 == i_def:
                    i_pv_sk = i_def - 1
                else:
                    break
            else:
                if i_pv_sk > i_def:
                    i_pv_sk += 1
                else:
                    i_pv_sk -= 1
        # print()
        ari_scores_sk = np.array(ari_scores_sk)
        max_index = np.where(ari_scores_sk == max(ari_scores_sk))[0][-1]
        
        default_index = np.where(ari_scores_sk == default_ari)[0][0]
        # default_index = ari_scores_sk.index(default_ari)
        paramaters_sk_copy[p_sk][1] = parameter_route_sk[max_index][0]
        informed_route_sk.append([paramaters_sk_copy[p_sk][0], max_index, default_index, parameter_route_sk])
    # print("Sklearn")
    # print(blind_route_sk)
    return informed_route_sk, informed_route_mat
    
    
def get_informed_route_mat(X, gt, filename, passing_param_sk, paramaters_mat_copy):
    informed_route = []
    
    for p in range(len(paramaters_mat_copy)):
        
        if paramaters_mat_copy[0][1] == "fmcd" and p != 0:
            if p == 4 or p == 5 or p == 6 or p == 7 or p == 8:
                continue
            
        if paramaters_mat_copy[0][1] == "ogk" and p != 0:
            if p == 1 or p == 2 or p == 3 or p==6 or p == 7 or p == 8:
                continue
        
        if paramaters_mat_copy[0][1] == "olivehawkins" and p != 0:
            if p == 3 or p == 4 or p == 5:
                continue
        
        parameter_route = []
        ari_scores = []
        f1_scores = []
        passing_param = deepcopy(paramaters_mat_copy)

        default_ari, default_f1_sk, default_f1_mat = runEE(filename, X, gt, passing_param_sk, passing_param)
        if default_ari == -1:
            return -1, -1, -1, []
        parameter_route.append([passing_param[p][1], default_ari, default_f1_sk, default_f1_mat])
        ari_scores.append(default_ari)
        f1_scores.append(default_f1_mat)
        i_def = passing_param[p][2].index(passing_param[p][1])
        if i_def+1 == len(paramaters_mat_copy[p][2]):
            i_pv = i_def-1    
        else:
            i_pv = i_def+1
        
        while True:
            if i_pv >= len(paramaters_mat_copy[p][2]):
                break
            if i_pv < 0:
                break

            passing_param[p][1] = paramaters_mat_copy[p][2][i_pv]
            ari_score, f1_score_sk, f1_score_mat = runEE(filename, X, gt, passing_param_sk, passing_param)
            if ari_score >= np.max(ari_scores) and f1_score_mat >= np.max(f1_scores):
                parameter_route.append([passing_param[p][1], ari_score, f1_score_sk, f1_score_mat])
                ari_scores.append(ari_score)
                f1_scores.append(f1_score_mat)
            if np.max(ari_scores) == 1:
                break
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
        ari_scores = np.array(ari_scores)
        max_index = np.where(ari_scores == max(ari_scores))[0][-1]
        default_index = np.where(ari_scores == default_ari)[0][0]
        paramaters_mat_copy[p][1] = parameter_route[max_index][0]
        informed_route.append([paramaters_mat_copy[p][0], max_index, default_index, parameter_route])
    return informed_route[-1][3][-1][1], informed_route[-1][3][-1][2], informed_route[-1][3][-1][3], informed_route


def runEE(filename, X, gt, param_sk, param_mat):
    labelFile_sk = filename + "_" + str(param_sk[0][1]) + "_" + str(param_sk[1][1]) + "_" + str(param_sk[2][1]) + "_" + str(param_sk[3][1])
    labelFile_mat = filename + "_" + str(param_mat[0][1]) + "_" + str(param_mat[1][1]) + "_" + str(param_mat[2][1]) + "_" + str(param_mat[3][1]) + "_" + str(param_mat[4][1]) + "_" + str(param_mat[5][1]) + "_" + str(param_mat[6][1]) + "_" + str(param_mat[7][1]) + "_" + str(param_mat[8][1])

    if os.path.exists("Labels/EE_Sk/Labels_Sk_EE_"+labelFile_sk+".csv") == 0:
        skf1 = get_sk_f1(filename, param_sk, X, gt)
        if skf1 == -1:
            return -1, -1, -1
    if os.path.exists("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile_mat+".csv") == 0:        
        frr=open("GD_ReRun/MatEE.csv", "a")
        frr.write(filename+","+str(param_mat[0][1])+","+str(param_mat[1][1])+","+str(param_mat[2][1])+","+str(param_mat[3][1])+","+str(param_mat[4][1])+","+str(param_mat[5][1])+","+str(param_mat[6][1])+","+str(param_mat[7][1])+","+str(param_mat[8][1])+'\n')
        frr.close()
        return -1, -1, -1
    labels_sk =  pd.read_csv("Labels/EE_Sk/Labels_Sk_EE_"+labelFile_sk+".csv", header=None).to_numpy()


    labels_mat =  pd.read_csv("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile_mat+".csv", header=None).to_numpy()
    
    ari = []
    
    for i in range(len(labels_sk)):
        for j in range(len(labels_mat)):
            ari.append(adjusted_rand_score(labels_sk[i], labels_mat[j]))
    # print(ari)
    return np.mean(ari), get_sk_f1(filename, param_sk, X, gt), get_mat_f1(filename, param_mat, X, gt)

def get_sk_f1(filename, param_sk, X, gt):
    labelFile = filename + "_" + str(param_sk[0][1]) + "_" + str(param_sk[1][1]) + "_" + str(param_sk[2][1]) + "_" + str(param_sk[3][1])
    
    if os.path.exists("Labels/EE_Sk_Done/Labels_Sk_EE_"+labelFile+".csv") == 1:
        i_store_precision=param_sk[0][1]
        i_assume_centered=param_sk[1][1]
        i_support_fraction=param_sk[2][1]
        i_contamination=param_sk[3][1]
        
        dff1 =  pd.read_csv("Stats/SkEE_F1.csv")
        f1 = dff1[(dff1['Filename']==filename)&
                    (dff1['store_precision']==i_store_precision)&
                    (dff1['assume_centered']==i_assume_centered)&
                    (dff1['support_fraction']==str(i_support_fraction))&
                    (dff1['contamination']==str(i_contamination))]
        if f1.empty == 0:
            runs_f1 = []
            for i in range(10):
                runs_f1.append(('R'+str(i)))
            run_f1_values = f1[runs_f1].to_numpy()
            
            return np.mean(np.mean(run_f1_values))
  
    sp = param_sk[0][1]
    ac = param_sk[1][1]
    sf = param_sk[2][1]
    cont = param_sk[3][1]
    if cont == "LOF":
        percentage_file = pd.read_csv("Stats/SkPercentage.csv")
        cont  = percentage_file[percentage_file["Filename"] == filename]["LOF"].to_numpy()[0]
    if cont == "IF":
        percentage_file = pd.read_csv("Stats/SkPercentage.csv")
        cont  = percentage_file[percentage_file["Filename"] == filename]["IF"].to_numpy()[0]
    if cont == 0:
        return -1
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

    return np.mean(f1)
def get_mat_f1(filename, param_mat, X, gt):
    labelFile = filename + "_" + str(param_mat[0][1]) + "_" + str(param_mat[1][1]) + "_" + str(param_mat[2][1]) + "_" + str(param_mat[3][1]) + "_" + str(param_mat[4][1]) + "_" + str(param_mat[5][1]) + "_" + str(param_mat[6][1]) + "_" + str(param_mat[7][1]) + "_" + str(param_mat[8][1])

    if os.path.exists("Labels/EE_Matlab_Done/Labels_Mat_EE_"+labelFile+".csv") == 1:
        i_Method = param_mat[0][1]
        i_OutlierFraction = param_mat[1][1]
        i_NumTrials = param_mat[2][1]
        i_BiasCorrection = param_mat[3][1]
        i_NumOGKIterations = param_mat[4][1]
        i_UnivariateEstimator = param_mat[5][1]
        i_ReweightingMethod = param_mat[6][1]
        i_NumConcentrationSteps = param_mat[7][1]
        i_StartMethod = param_mat[8][1]
        
            
        
        dff1 =  pd.read_csv("Stats/MatEE_F1.csv")
        f1 = dff1[(dff1['Filename']==filename)&
                    (dff1['Method']==i_Method)&
                    (dff1['OutlierFraction']==i_OutlierFraction)&
                    (dff1['NumTrials']==i_NumTrials)&
                    (dff1['BiasCorrection']==i_BiasCorrection)&
                    (dff1['NumOGKIterations']==i_NumOGKIterations)&
                    (dff1['UnivariateEstimator']==i_UnivariateEstimator)&
                    (dff1['ReweightingMethod']==i_ReweightingMethod)&
                    (dff1['NumConcentrationSteps']==i_NumConcentrationSteps)&
                    (dff1['StartMethod']==i_StartMethod)]
        
        if f1.empty == 0:
            runs_f1 = []
            for i in range(10):
                runs_f1.append(('R'+str(i)))
            run_f1_values = f1[runs_f1].to_numpy()
            
            return np.mean(np.mean(run_f1_values))

    
    
    labels = []
    f1 = []
    ari = []
    
    
    labels =  pd.read_csv("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile+".csv", header=None).to_numpy()
    for i in range(10):
        f1.append(metrics.f1_score(gt, labels[i]))
        

    return np.mean(f1)



if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
    
    parameters_mat = []

    Method = ["olivehawkins", "fmcd", "ogk"];
    OutlierFraction = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5];
    NumTrials = [2, 5, 10, 25, 50, 100, 250, 500, 750, 1000];
    BiasCorrection = [1, 0];
    NumOGKIterations = [1, 2, 3];
    UnivariateEstimator = ["tauscale", "qn"];
    ReweightingMethod = ["rfch", "rmvn"];
    NumConcentrationSteps = [2, 5, 10, 15, 20];
    StartMethod = ["elemental","classical", "medianball"];    
    
    parameters_mat.append(["Method", "fmcd", Method])
    parameters_mat.append(["OutlierFraction", 0.5, OutlierFraction])
    parameters_mat.append(["NumTrials", 500, NumTrials])
    parameters_mat.append(["BiasCorrection", 1, BiasCorrection])
    parameters_mat.append(["NumOGKIterations", 2, NumOGKIterations])
    parameters_mat.append(["UnivariateEstimator", "tauscale", UnivariateEstimator])
    parameters_mat.append(["ReweightingMethod", "rfch", ReweightingMethod])
    parameters_mat.append(["NumConcentrationSteps", 10, NumConcentrationSteps])
    parameters_mat.append(["StartMethod", "classical", StartMethod])
    
    
    parameters_sk = []
    store_precision = [True, False]
    assume_centered = [True, False]
    support_fraction = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    contamination = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    parameters_sk.append(["store_precision", True, store_precision])
    parameters_sk.append(["assume_centered", False, assume_centered])
    parameters_sk.append(["support_fraction", None, support_fraction])
    parameters_sk.append(["contamination", 0.1, contamination])
    
    

    # for FileNumber in range(len(master_files)):
    #     print(FileNumber, end=' ')
    #     MatEE_Route_Scores = pd.read_csv("Stats/MatEE_Route_Scores.csv")
    #     fileroute = MatEE_Route_Scores[MatEE_Route_Scores["Filename"] == master_files[FileNumber]]
        
    #     if fileroute.empty:
    #         continue
    #     if fileroute["UninformedF1"].values[0] == 0 and fileroute["InformedF1"].values[0] == 0:
    #         continue

    ee("vertebral", parameters_mat, parameters_sk)
    
    # plot_ari_f1() 

    