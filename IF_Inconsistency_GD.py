#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 18:01:52 2022

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
import scipy.stats as ss
import bisect 

datasetFolderDir = 'Dataset/'


def isolationforest(filename, parameters_r, parameters_mat, parameters_sk):
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
    
    DefaultARI, DefaultF1_r, DefaultF1_mat, DefaultF1_sk = runIF(filename, X, gt, parameters_r, parameters_mat, parameters_sk)
    print(DefaultARI, DefaultF1_r, DefaultF1_mat, DefaultF1_sk)
    # ## Rearrange "IF" and "LOF"
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
    
    auto_1 = min(256, X.shape[0])/X.shape[0]
    bisect.insort(mod_parameters_mat[2][2], auto_1)
    mod_parameters_mat[2][2][mod_parameters_mat[2][2].index(auto_1)] = 'auto'
    
    
    auto_1 = min(256, X.shape[0])/X.shape[0]
    bisect.insort(mod_parameters_sk[1][2], auto_1)
    mod_parameters_sk[1][2][mod_parameters_sk[1][2].index(auto_1)] = 'auto'
    # ##
    
    
    
    blind_route_r, blind_route_mat, blind_route_sk = get_blind_route(X, gt, filename, deepcopy(mod_parameters_r), deepcopy(mod_parameters_mat), deepcopy(mod_parameters_sk))
    informed_route_r, informed_route_mat, informed_route_sk = get_guided_route(X, gt, filename, deepcopy(mod_parameters_r), deepcopy(mod_parameters_mat), deepcopy(mod_parameters_sk))
    print(blind_route_r)
    print(blind_route_mat)
    
    print(blind_route_sk)
    
    UninformedARI = str(blind_route_r[-1][3][-1][1])
    UninformedF1_r = str(blind_route_r[-1][3][-1][2])
    UninformedF1_mat = str(blind_route_mat[-1][3][-1][3])
    UninformedF1_sk = str(blind_route_sk[-1][3][-1][4])
    
    InformedARI = str(informed_route_r[-1][3][-1][1])
    InformedF1_r = str(informed_route_r[-1][3][-1][2])
    InformedF1_mat = str(informed_route_mat[-1][3][-1][3])
    InformedF1_sk = str(informed_route_sk[-1][3][-1][4])
    
    # draw_graph(filename, blind_route_sk,informed_route_sk, 'R')
    # draw_graph(filename, blind_route_mat,informed_route_mat, 'M')
    # draw_graph(filename, blind_route_sk,informed_route_sk, 'S')
    
    f_Route_Scores=open("Stats/IF_SvMvR_Route_Scores.csv", "a")
    f_Route_Scores.write(filename+','+str(DefaultARI)+","+str(DefaultF1_r)+","+str(DefaultF1_mat)+","+str(DefaultF1_sk)+","+UninformedARI+","+UninformedF1_r+","+UninformedF1_mat+","+UninformedF1_sk+","+InformedARI+","+InformedF1_r+","+InformedF1_mat+","+InformedF1_sk+"\n")
    f_Route_Scores.close()
    
def draw_graph(filename, blind_route, informed_route, tool):
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
        plt.plot(ari_x, ari_scores, '-')
        for k in range(len(ari_scores)-1):
            plt.plot(ari_x[k+1], ari_scores[k+1], marker=(3, 0, get_angle(ari_x[k], ari_scores[k], ari_x[k+1], ari_scores[k+1])-90), color='black', markersize=8)
            
        
        # if i == 0:
        #     plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][2]][0]), (0.1, blind_route[i][3][blind_route[i][2]][1]), ha='left')

        # if start != end:
        #     plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][1]][0]), (end+0.1, blind_route[i][3][blind_route[i][1]][1]), ha='left')
    # plt.legend(param_names)
    plt.ylabel("Cross-run ARI")
    plt.xticks(ticks= [])
    plt.title(filename)
    plt.savefig("Fig/GD/Routes_Inconsistency/"+filename+"_IF_SvMvR_"+tool+"_Trajectory.pdf", bbox_inches="tight", pad_inches=0)
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
            if tool == 'S':
                ari_x.append(ari_p[j][2])
            elif tool == 'M':
                ari_x.append(ari_p[j][3])
        plt.plot(ari_x, ari_scores, '-')
        for k in range(len(ari_scores)-1):
            plt.plot(ari_x[k+1], ari_scores[k+1], marker=(3, 0, get_angle(ari_x[k], ari_scores[k], ari_x[k+1], ari_scores[k+1])-90), color='black', markersize=8)
        
        # if i == 0:
        #     plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][2]][0]), (blind_route[i][3][blind_route[i][2]][2], blind_route[i][3][blind_route[i][2]][1]), ha='left')

        # if start != end:
        #     plt.annotate(blind_route[i][0]+" = "+str(blind_route[i][3][blind_route[i][1]][0]), (blind_route[i][3][blind_route[i][1]][2], blind_route[i][3][blind_route[i][1]][1]), ha='left')
    plt.ylabel("Cross-run ARI")
    plt.xlabel("F1 Score")
    
    plt.title(filename)
    plt.savefig("Fig/GD/Routes_Inconsistency/"+filename+"_IF_SvMvR_"+tool+"_Trajectory_W_F1.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    
    ## Informed F1 Score
    fig = plt.Figure()
    start = end = 0
    for i in range(len(informed_route)):
        default = informed_route[i][2]
        start = end
        end = informed_route[i][1] - (default-start)
        
        ari_p = informed_route[i][3]
        ari_scores = []
        ari_x = []
        for j in range(len(ari_p)):
            ari_scores.append(ari_p[j][1])
            if tool == 'S':
                ari_x.append(ari_p[j][2])
            elif tool == 'M':
                ari_x.append(ari_p[j][3])
        plt.plot(ari_x, ari_scores, '-')
        
        for k in range(len(ari_scores)-1):
            plt.plot(ari_x[k+1], ari_scores[k+1], marker=(3, 0, get_angle(ari_x[k], ari_scores[k], ari_x[k+1], ari_scores[k+1])-90), color='black', markersize=8)

        
        # if i == 0:
        #     plt.annotate(informed_route[i][0]+" = "+str(informed_route[i][3][informed_route[i][2]][0]), (informed_route[i][3][informed_route[i][2]][2], informed_route[i][3][informed_route[i][2]][1]), ha='left')

        # if start != end:
        #     plt.annotate(informed_route[i][0]+" = "+str(informed_route[i][3][informed_route[i][1]][0]), (informed_route[i][3][guided_route[i][1]][2], informed_route[i][3][informed_route[i][1]][1]), ha='left')
    # plt.legend(param_names)
    plt.ylabel("Cross-run ARI")
    plt.xlabel("F1 Score")
    
    plt.title(filename)
    plt.savefig("Fig/GD/Routes_Inconsistency/"+filename+"_IF_SvMvR_"+tool+"_Guided_Trajectory_W_F1.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    
def get_angle(p1x, p1y, p2x, p2y):
    dx = p2x - p1x
    dy = p2y - p1y
    theta = math.atan2(dy, dx)
    angle = math.degrees(theta)  # angle is in (-180, 180]
    if angle < 0:
        angle = 360 + angle
    return angle

def get_blind_route(X, gt, filename, paramaters_r_copy,paramaters_mat_copy, paramaters_sk_copy):
    blind_route_r = []
    blind_route_mat = []
    blind_route_sk = []
    # route_temp = []
    print("R - ", end='')
    for p_r in range(len(paramaters_r_copy)):
        print(p_r)
        parameter_route_r = []
        ari_scores_r = []
        passing_param_r = deepcopy(paramaters_r_copy)

        default_ari, default_f1_r, default_f1_mat, default_f1_sk, route_mat, route_sk = get_blind_route_mat(X, gt, filename, passing_param_r, paramaters_mat_copy, paramaters_sk_copy)
        if default_ari == -1:
            return [], [], []
        parameter_route_r.append([passing_param_r[p_r][1], default_ari, default_f1_r, default_f1_mat, default_f1_sk])
        ari_scores_r.append(default_ari)
        # blind_route_mat += route_mat
        blind_route_mat = route_mat
        blind_route_sk = route_sk
        i_def = passing_param_r[p_r][2].index(passing_param_r[p_r][1])
        if i_def+1 == len(passing_param_r[p_r][2]):
            i_pv_r = i_def-1
        else:
            i_pv_r = i_def+1
        
        while True:
            if i_pv_r >= len(paramaters_r_copy[p_r][2]):
                break
            if i_pv_r < 0:
                break

            passing_param_r[p_r][1] = paramaters_r_copy[p_r][2][i_pv_r]
            
            ari_score, f1_score_r, f1_score_mat, f1_score_sk, route_mat, route_sk = get_blind_route_mat(X, gt, filename, passing_param_r, paramaters_mat_copy, paramaters_sk_copy)
            # route_temp += route_mat
            if ari_score >= np.max(ari_scores_r):
                parameter_route_r.append([passing_param_r[p_r][1], ari_score, f1_score_r, f1_score_mat, f1_score_sk])
                ari_scores_r.append(ari_score)
                # blind_route_mat += route_temp
                blind_route_mat = route_mat
                blind_route_sk = route_sk
                route_temp = []
            if ari_score != np.max(ari_scores_r):
                
                if i_pv_r - 1 > i_def:
                    break
                elif i_pv_r - 1 == i_def:
                    i_pv_r = i_def - 1
                else:
                    break
            else:
                if i_pv_r > i_def:
                    i_pv_r += 1
                else:
                    i_pv_r -= 1
        ari_scores_r = np.array(ari_scores_r)
        max_index = np.where(ari_scores_r == max(ari_scores_r))[0][-1]
        
        default_index = np.where(ari_scores_r == default_ari)[0][0]
        paramaters_r_copy[p_r][1] = parameter_route_r[max_index][0]
        blind_route_r.append([paramaters_r_copy[p_r][0], max_index, default_index, parameter_route_r])
    return blind_route_r, blind_route_mat, blind_route_sk
    
    
def get_blind_route_mat(X, gt, filename, passing_param_r, paramaters_mat_copy, paramaters_sk_copy):
    blind_route_mat = []
    blind_route_sk = []
    print("mat - ", end='')
    for p in range(len(paramaters_mat_copy)):
        print(p)
        parameter_route_mat = []
        ari_scores_mat = []
        passing_param_mat = deepcopy(paramaters_mat_copy)

        default_ari, default_f1_r, default_f1_mat, default_f1_sk, route_sk = get_blind_route_sk(X, gt, filename, passing_param_r, passing_param_mat, paramaters_sk_copy)
        if default_ari == -1:
            return -1, -1, -1, -1, [], []
        blind_route_sk = route_sk
        parameter_route_mat.append([passing_param_mat[p][1], default_ari, default_f1_r, default_f1_mat, default_f1_sk])
        ari_scores_mat.append(default_ari)
        i_def = passing_param_mat[p][2].index(passing_param_mat[p][1])
        if i_def+1 == len(paramaters_mat_copy[p][2]):
            i_pv = i_def-1    
        else:
            i_pv = i_def+1
        
        while True:
            if i_pv >= len(paramaters_mat_copy[p][2]):
                break
            if i_pv < 0:
                break

            passing_param_mat[p][1] = paramaters_mat_copy[p][2][i_pv]
            ari_score, f1_score_r, f1_score_mat, f1_score_sk, route_sk = get_blind_route_sk(X, gt, filename, passing_param_r, passing_param_mat, paramaters_sk_copy)
            if ari_score >= np.max(ari_scores_mat):
                parameter_route_mat.append([passing_param_mat[p][1], ari_score, f1_score_r, f1_score_mat, f1_score_sk])
                ari_scores_mat.append(ari_score)
                blind_route_sk = route_sk
            if np.max(ari_scores_mat) == 1:
                break
            if ari_score != np.max(ari_scores_mat):
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
        ari_scores_mat = np.array(ari_scores_mat)
        max_index = np.where(ari_scores_mat == max(ari_scores_mat))[0][-1]
        default_index = np.where(ari_scores_mat == default_ari)[0][0]
        paramaters_mat_copy[p][1] = parameter_route_mat[max_index][0]
        blind_route_mat.append([paramaters_mat_copy[p][0], max_index, default_index, parameter_route_mat])
    return blind_route_mat[-1][3][-1][1], blind_route_mat[-1][3][-1][2], blind_route_mat[-1][3][-1][3], blind_route_mat[-1][3][-1][4], blind_route_mat, blind_route_sk

def get_blind_route_sk(X, gt, filename, passing_param_r, passing_param_mat, paramaters_sk_copy):
    blind_route_sk = []

    for p in range(len(paramaters_sk_copy)):
        parameter_route_sk = []
        ari_scores_sk = []
        passing_param_sk = deepcopy(paramaters_sk_copy)

        default_ari, default_f1_r, default_f1_mat, default_f1_sk = runIF(filename, X, gt, passing_param_r, passing_param_mat, passing_param_sk)
        if default_ari == -1:
            return -1, -1, -1, -1, []
        parameter_route_sk.append([passing_param_sk[p][1], default_ari, default_f1_r, default_f1_mat, default_f1_sk])
        ari_scores_sk.append(default_ari)
        i_def = passing_param_sk[p][2].index(passing_param_sk[p][1])
        if i_def+1 == len(paramaters_sk_copy[p][2]):
            i_pv = i_def-1    
        else:
            i_pv = i_def+1
        
        while True:
            if i_pv >= len(paramaters_sk_copy[p][2]):
                break
            if i_pv < 0:
                break

            passing_param_sk[p][1] = paramaters_sk_copy[p][2][i_pv]
            ari_score, f1_score_r, f1_score_mat, f1_score_sk = runIF(filename, X, gt, passing_param_r, passing_param_mat, passing_param_sk)
            if ari_score >= np.max(ari_scores_sk):
                parameter_route_sk.append([passing_param_sk[p][1], ari_score, f1_score_r, f1_score_mat, f1_score_sk])
                ari_scores_sk.append(ari_score)
            if np.max(ari_scores_sk) == 1:
                break
            if ari_score != np.max(ari_scores_sk):
                
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
        ari_scores_sk = np.array(ari_scores_sk)
        max_index = np.where(ari_scores_sk == max(ari_scores_sk))[0][-1]
        default_index = np.where(ari_scores_sk == default_ari)[0][0]
        paramaters_sk_copy[p][1] = parameter_route_sk[max_index][0]
        blind_route_sk.append([paramaters_sk_copy[p][0], max_index, default_index, parameter_route_sk])
    return blind_route_sk[-1][3][-1][1], blind_route_sk[-1][3][-1][2], blind_route_sk[-1][3][-1][3], blind_route_sk[-1][3][-1][4], blind_route_sk


def get_guided_route(X, gt, filename, paramaters_r_copy,paramaters_mat_copy, paramaters_sk_copy):
    guided_route_r = []
    guided_route_mat = []
    guided_route_sk = []
    # route_temp = []
    print("R - ", end='')
    for p_r in range(len(paramaters_r_copy)):
        print(p_r)
        parameter_route_r = []
        ari_scores_r = []
        f1_scores_r = []
        passing_param_r = deepcopy(paramaters_r_copy)

        default_ari, default_f1_r, default_f1_mat, default_f1_sk, route_mat, route_sk = get_guided_route_mat(X, gt, filename, passing_param_r, paramaters_mat_copy, paramaters_sk_copy)
        if default_ari == -1:
            return [], [], []
        parameter_route_r.append([passing_param_r[p_r][1], default_ari, default_f1_r, default_f1_mat, default_f1_sk])
        ari_scores_r.append(default_ari)
        f1_scores_r.append(default_f1_r)
        # guided_route_mat += route_mat
        guided_route_mat = route_mat
        guided_route_sk = route_sk
        i_def = passing_param_r[p_r][2].index(passing_param_r[p_r][1])
        if i_def+1 == len(passing_param_r[p_r][2]):
            i_pv_r = i_def-1    
        else:
            i_pv_r = i_def+1
        
        while True:
            if i_pv_r >= len(paramaters_r_copy[p_r][2]):
                break
            if i_pv_r < 0:
                break

            passing_param_r[p_r][1] = paramaters_r_copy[p_r][2][i_pv_r]
            
            ari_score, f1_score_r, f1_score_mat, f1_score_sk, route_mat, route_sk = get_guided_route_mat(X, gt, filename, passing_param_r, paramaters_mat_copy, paramaters_sk_copy)
            # route_temp += route_mat
            if ari_score >= np.max(ari_scores_r) and f1_score_r > np.max(f1_scores_r):
                parameter_route_r.append([passing_param_r[p_r][1], ari_score, f1_score_r, f1_score_mat, f1_score_sk])
                ari_scores_r.append(ari_score)
                f1_scores_r.append(f1_score_r)
                # guided_route_mat += route_temp
                guided_route_mat = route_mat
                guided_route_sk = route_sk
                route_temp = []
            if ari_score != np.max(ari_scores_r):
                
                if i_pv_r - 1 > i_def:
                    break
                elif i_pv_r - 1 == i_def:
                    i_pv_r = i_def - 1
                else:
                    break
            else:
                if i_pv_r > i_def:
                    i_pv_r += 1
                else:
                    i_pv_r -= 1
        ari_scores_r = np.array(ari_scores_r)
        max_index = np.where(ari_scores_r == max(ari_scores_r))[0][-1]
        
        default_index = np.where(ari_scores_r == default_ari)[0][0]
        paramaters_r_copy[p_r][1] = parameter_route_r[max_index][0]
        guided_route_r.append([paramaters_r_copy[p_r][0], max_index, default_index, parameter_route_r])
    return guided_route_r, guided_route_mat, guided_route_sk
    
    
def get_guided_route_mat(X, gt, filename, passing_param_r, paramaters_mat_copy, paramaters_sk_copy):
    guided_route_mat = []
    guided_route_sk = []
    print("mat - ", end='')
    for p in range(len(paramaters_mat_copy)):
        print(p)
        parameter_route_mat = []
        ari_scores_mat = []
        f1_scores_mat = []
        passing_param_mat = deepcopy(paramaters_mat_copy)

        default_ari, default_f1_r, default_f1_mat, default_f1_sk, route_sk = get_guided_route_sk(X, gt, filename, passing_param_r, passing_param_mat, paramaters_sk_copy)
        if default_ari == -1:
            return -1, -1, -1, -1, [], []
        guided_route_sk = route_sk
        parameter_route_mat.append([passing_param_mat[p][1], default_ari, default_f1_r, default_f1_mat, default_f1_sk])
        ari_scores_mat.append(default_ari)
        f1_scores_mat.append(default_f1_mat)
        i_def = passing_param_mat[p][2].index(passing_param_mat[p][1])
        if i_def+1 == len(paramaters_mat_copy[p][2]):
            i_pv = i_def-1    
        else:
            i_pv = i_def+1
        
        while True:
            if i_pv >= len(paramaters_mat_copy[p][2]):
                break
            if i_pv < 0:
                break

            passing_param_mat[p][1] = paramaters_mat_copy[p][2][i_pv]
            ari_score, f1_score_r, f1_score_mat, f1_score_sk, route_sk = get_guided_route_sk(X, gt, filename, passing_param_r, passing_param_mat, paramaters_sk_copy)
            if ari_score >= np.max(ari_scores_mat) and f1_score_mat >= np.max(f1_scores_mat):
                parameter_route_mat.append([passing_param_mat[p][1], ari_score, f1_score_r, f1_score_mat, f1_score_sk])
                ari_scores_mat.append(ari_score)
                f1_scores_mat.append(f1_score_mat)
                guided_route_sk = route_sk
            if np.max(ari_scores_mat) == 1:
                break
            if ari_score != np.max(ari_scores_mat):
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
        ari_scores_mat = np.array(ari_scores_mat)
        max_index = np.where(ari_scores_mat == max(ari_scores_mat))[0][-1]
        default_index = np.where(ari_scores_mat == default_ari)[0][0]
        paramaters_mat_copy[p][1] = parameter_route_mat[max_index][0]
        guided_route_mat.append([paramaters_mat_copy[p][0], max_index, default_index, parameter_route_mat])
    return guided_route_mat[-1][3][-1][1], guided_route_mat[-1][3][-1][2], guided_route_mat[-1][3][-1][3], guided_route_mat[-1][3][-1][4], guided_route_mat, guided_route_sk

def get_guided_route_sk(X, gt, filename, passing_param_r, passing_param_mat, paramaters_sk_copy):
    guided_route_sk = []

    for p in range(len(paramaters_sk_copy)):
        parameter_route_sk = []
        ari_scores_sk = []
        f1_scores_sk = []
        passing_param_sk = deepcopy(paramaters_sk_copy)

        default_ari, default_f1_r, default_f1_mat, default_f1_sk = runIF(filename, X, gt, passing_param_r, passing_param_mat, passing_param_sk)
        if default_ari == -1:
            return -1, -1, -1, -1, []
        parameter_route_sk.append([passing_param_sk[p][1], default_ari, default_f1_r, default_f1_mat, default_f1_sk])
        ari_scores_sk.append(default_ari)
        f1_scores_sk.append(default_f1_sk)
        i_def = passing_param_sk[p][2].index(passing_param_sk[p][1])
        if i_def+1 == len(paramaters_sk_copy[p][2]):
            i_pv = i_def-1    
        else:
            i_pv = i_def+1
        
        while True:
            if i_pv >= len(paramaters_sk_copy[p][2]):
                break
            if i_pv < 0:
                break

            passing_param_sk[p][1] = paramaters_sk_copy[p][2][i_pv]
            ari_score, f1_score_r, f1_score_mat, f1_score_sk = runIF(filename, X, gt, passing_param_r, passing_param_mat, passing_param_sk)
            if ari_score >= np.max(ari_scores_sk) and f1_score_sk >= np.max(f1_scores_sk):
                parameter_route_sk.append([passing_param_sk[p][1], ari_score, f1_score_r, f1_score_mat, f1_score_sk])
                ari_scores_sk.append(ari_score)
                f1_scores_sk.append(f1_score_sk)

            if np.max(ari_scores_sk) == 1:
                break
            if ari_score != np.max(ari_scores_sk):
                
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
        ari_scores_sk = np.array(ari_scores_sk)
        max_index = np.where(ari_scores_sk == max(ari_scores_sk))[0][-1]
        default_index = np.where(ari_scores_sk == default_ari)[0][0]
        paramaters_sk_copy[p][1] = parameter_route_sk[max_index][0]
        guided_route_sk.append([paramaters_sk_copy[p][0], max_index, default_index, parameter_route_sk])
    return guided_route_sk[-1][3][-1][1], guided_route_sk[-1][3][-1][2], guided_route_sk[-1][3][-1][3], guided_route_sk[-1][3][-1][4], guided_route_sk




def runIF(filename, X, gt, param_r, param_mat, param_sk):
    labelFile_r = filename + "_" + str(param_r[0][1]) + "_" + str(param_r[1][1]) + "_" + str(param_r[2][1]) + "_" + str(param_r[3][1])
    labelFile_mat = filename + "_" + str(param_mat[0][1]) + "_" + str(param_mat[1][1]) + "_" + str(param_mat[2][1])
    labelFile_sk = filename + "_" + str(param_sk[0][1]) + "_" + str(param_sk[1][1]) + "_" + str(param_sk[2][1]) + "_" + str(param_sk[3][1]) + "_" + str(param_sk[4][1]) + "_" + str(param_sk[5][1]) + "_" + str(param_sk[6][1])

    
    if os.path.exists("Labels/IF_R/"+labelFile_r+".csv") == 0:        
        frr=open("GD_ReRun/RIF.csv", "a")
        frr.write(filename+","+str(param_r[0][1])+","+str(param_r[1][1])+","+str(param_r[2][1])+","+str(param_r[3][1])+'\n')
        frr.close()
        return -1, -1, -1, -1
    if os.path.exists("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile_mat+".csv") == 0:        
        frr=open("GD_ReRun/MatIF.csv", "a")
        frr.write(filename+","+str(param_mat[0][1])+","+str(param_mat[1][1])+","+str(param_mat[2][1])+'\n')
        frr.close()
        return -1, -1, -1, -1
    if os.path.exists("Labels/IF_Sk/Labels_Sk_IF_"+labelFile_sk+".csv") == 0:
        skf1 = get_sk_f1(filename, param_sk, X, gt)
        if skf1 == -1:
            return -1, -1, -1, -1
    ##
    dfmvr = pd.read_csv("Stats/Inconsistency/IF_mvr.csv")
    dfmvr = dfmvr[(dfmvr['Mat']==labelFile_mat)&
                    (dfmvr['R']==labelFile_r)]
    if dfmvr.empty:
        labels_r = pd.read_csv("Labels/IF_R/"+labelFile_r+".csv").to_numpy()
        labels_mat = pd.read_csv("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile_mat+".csv", header=None).to_numpy()
        
        ari_mvr = []
        for i in range(len(labels_r)):
            for j in range(len(labels_mat)):
                ari_mvr.append(adjusted_rand_score(np.int64((labels_r[i][1:])*1), labels_mat[j]))
        ari_mvr = np.mean(ari_mvr)
        fmvr=open("Stats/Inconsistency/IF_mvr.csv", "a")
        fmvr.write(labelFile_mat+','+labelFile_r+ ','+ str(ari_mvr) + '\n')
        fmvr.close()
    else:
        ari_mvr = dfmvr['ARI'].to_numpy()[0]
    ##
    dfrvs = pd.read_csv("Stats/Inconsistency/IF_rvs.csv")
    dfrvs = dfrvs[(dfrvs['R']==labelFile_r)&
                    (dfrvs['Sk']==labelFile_sk)]
    if dfrvs.empty:
        labels_sk = pd.read_csv("Labels/IF_Sk/Labels_Sk_IF_"+labelFile_sk+".csv", header=None).to_numpy()
        labels_r = pd.read_csv("Labels/IF_R/"+labelFile_r+".csv").to_numpy()

        ari_rvs = []
        for i in range(len(labels_r)):
            for j in range(len(labels_sk)):
                ari_rvs.append(adjusted_rand_score(np.int64((labels_r[i][1:])*1), labels_sk[j]))
        ari_rvs = np.mean(ari_rvs)
        frvs=open("Stats/Inconsistency/IF_rvs.csv", "a")
        frvs.write(labelFile_r+','+labelFile_sk + ','+ str(ari_rvs) + '\n')
        frvs.close()
    else:
        ari_rvs = dfrvs['ARI'].to_numpy()[0]
    ##
    dfmvs = pd.read_csv("Stats/Inconsistency/IF_mvs.csv")
    dfmvs = dfmvs[(dfmvs['Mat']==labelFile_mat)&
                    (dfmvs['Sk']==labelFile_sk)]
    if dfmvs.empty:
        labels_sk = pd.read_csv("Labels/IF_Sk/Labels_Sk_IF_"+labelFile_sk+".csv", header=None).to_numpy()
        labels_mat = pd.read_csv("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile_mat+".csv", header=None).to_numpy()
        
        ari_mvs = []
        for i in range(len(labels_sk)):
            for j in range(len(labels_mat)):
                ari_mvs.append(adjusted_rand_score(labels_sk[i], labels_mat[j]))
        ari_mvs = np.mean(ari_mvs) 
        fmvs=open("Stats/Inconsistency/IF_mvs.csv", "a")
        fmvs.write(labelFile_mat+','+labelFile_sk + ','+ str(ari_mvs) + '\n')
        fmvs.close()
    else:
        ari_mvs = dfmvs['ARI'].to_numpy()[0]
    
    ari = []  
    ari = (ari_mvr + ari_rvs + ari_mvs)/3
    
    return ari, get_r_f1(filename, param_r, X, gt), get_mat_f1(filename, param_mat, X, gt), get_sk_f1(filename, param_sk, X, gt)

def get_sk_f1(filename, param_sk, X, gt):
    labelFile = filename + "_" + str(param_sk[0][1]) + "_" + str(param_sk[1][1]) + "_" + str(param_sk[2][1]) + "_" + str(param_sk[3][1]) + "_" + str(param_sk[4][1]) + "_" + str(param_sk[5][1]) + "_" + str(param_sk[6][1])
    
    if os.path.exists("Labels/IF_Sk_Done/Labels_Sk_IF_"+labelFile+".csv") == 1:
        i_n_estimators=param_sk[0][1]
        i_max_samples=param_sk[1][1]
        i_contamination=param_sk[2][1]
        i_max_features=param_sk[3][1]
        i_bootstrap=param_sk[4][1]
        i_n_jobs=param_sk[5][1]
        i_warm_start = param_sk[6][1]
        
        dff1 =  pd.read_csv("Stats/SkIF_F1.csv")
        f1 = dff1[(dff1['Filename']==filename)&
                    (dff1['n_estimators']==i_n_estimators)&
                    (dff1['max_samples']==str(i_max_samples))&
                    (dff1['max_features']==i_max_features)&
                    (dff1['bootstrap']==i_bootstrap)&
                    (dff1['n_jobs']==str(i_n_jobs))&
                    (dff1['warm_start']==i_warm_start)]
        if f1.empty == 0:
            runs_f1 = []
            for i in range(10):
                runs_f1.append(('R'+str(i)))
            run_f1_values = f1[runs_f1].to_numpy()
            
            return np.mean(np.mean(run_f1_values))
  
    labels = []
    f1 = []
    ari = []
    for i in range(10):
        clustering = IsolationForest(n_estimators=param_sk[0][1], max_samples=param_sk[1][1], 
                                      max_features=param_sk[3][1], bootstrap=param_sk[4][1], 
                                      n_jobs=param_sk[5][1], warm_start=param_sk[6][1]).fit(X)
    
        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        labels.append(l)
        f1.append(metrics.f1_score(gt, l))
        
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
          ari.append(adjusted_rand_score(labels[i], labels[j]))      
    if os.path.exists("Labels/IF_Sk/Labels_Sk_IF_"+labelFile+".csv") == 0:
        fileLabels=open("Labels/IF_Sk/Labels_Sk_IF_"+labelFile+".csv", 'a')
        for l in labels:
            fileLabels.write(','.join(str(s) for s in l) + '\n')
        fileLabels.close()
    
    flabel_done=open("Labels/IF_Sk_Done/Labels_Sk_IF_"+labelFile+".csv", 'a')
    flabel_done.write("Done")
    flabel_done.close()
    
    fstat_f1=open("Stats/SkIF_F1.csv", "a")
    fstat_f1.write(filename+','+ str(param_sk[0][1]) + ','+ str(param_sk[1][1]) + ',' + str(param_sk[2][1]) + ',' + str(param_sk[3][1]) + ',' + str(param_sk[4][1]) + ',' + str(param_sk[5][1]) + ',' + str(param_sk[6][1]) + ',0,')
    fstat_f1.write(','.join(str(s) for s in f1) + '\n')
    fstat_f1.close()
    
    fstat_ari=open("Stats/SkIF_ARI.csv", "a")
    fstat_ari.write(filename+','+ str(param_sk[0][1]) + ','+ str(param_sk[1][1]) + ',' + str(param_sk[2][1]) + ',' + str(param_sk[3][1]) + ',' + str(param_sk[4][1]) + ',' + str(param_sk[5][1]) + ',' + str(param_sk[6][1]) + ',0,')
    fstat_ari.write(','.join(str(s) for s in ari) + '\n')
    fstat_ari.close()

    return np.mean(f1)

def get_mat_f1(filename, param_mat, X, gt):
    labelFile = filename + "_" + str(param_mat[0][1]) + "_" + str(param_mat[1][1]) + "_" + str(param_mat[2][1])
    
    if os.path.exists("Labels/IF_Matlab_Done/Labels_Mat_IF_"+labelFile+".csv") == 1:
        i_ContaminationFraction=param_mat[0][1]
        i_NumLearners=param_mat[1][1]
        i_NumObservationsPerLearner=param_mat[2][1]
        
        dff1 =  pd.read_csv("Stats/MatIF_F1.csv")
        f1 = dff1[(dff1['Filename']==filename)&
                    (dff1['ContaminationFraction']==str(i_ContaminationFraction))&
                    (dff1['NumLearners']==i_NumLearners)&
                    (dff1['NumObservationsPerLearner']==str(i_NumObservationsPerLearner))]
        
        if f1.empty == 0:
            runs_f1 = []
            for i in range(10):
                runs_f1.append(('R'+str(i)))
            run_f1_values = f1[runs_f1].to_numpy()
            
            return np.mean(np.mean(run_f1_values))

    if os.path.exists("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile+".csv") == 0:        
        frr=open("GD_ReRun/MatIF.csv", "a")
        frr.write(filename+","+str(param_mat[0][1])+","+str(param_mat[1][1])+","+str(param_mat[2][1])+'\n')
        frr.close()
        return -1
    
    labels = []
    f1 = []
    ari = []
    
    labels =  pd.read_csv("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile+".csv", header=None).to_numpy()
    for i in range(10):
        f1.append(metrics.f1_score(gt, labels[i]))
        
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
          ari.append(adjusted_rand_score(labels[i], labels[j]))
          
    flabel_done=open("Labels/IF_Matlab_Done/Labels_Mat_IF_"+labelFile+".csv", 'a')
    flabel_done.write("Done")
    flabel_done.close()
    
    fstat_f1=open("Stats/MatIF_F1.csv", "a")
    fstat_f1.write(filename+','+ str(param_mat[0][1]) + ','+ str(param_mat[1][1]) + ',' + str(param_mat[2][1]) + ',0,')
    fstat_f1.write(','.join(str(s) for s in f1) + '\n')
    fstat_f1.close()
    
    fstat_ari=open("Stats/MatIF_ARI.csv", "a")
    fstat_ari.write(filename+','+ str(param_mat[0][1]) + ','+ str(param_mat[1][1]) + ',' + str(param_mat[2][1]) + ',0,')
    fstat_ari.write(','.join(str(s) for s in ari) + '\n')
    fstat_ari.close()

    return np.mean(f1)

def get_r_f1(filename, param_r, X, gt):
    labelFile = filename + "_" + str(param_r[0][1]) + "_" + str(param_r[1][1]) + "_" + str(param_r[2][1]) + "_" + str(param_r[3][1])

    if os.path.exists("Labels/IF_R_Done/"+labelFile+".csv") == 1:
        i_ntrees=param_r[0][1]
        i_standardize_data=param_r[1][1]
        i_sample_size=param_r[2][1]
        i_ncols_per_tree=param_r[3][1]
        
        if i_standardize_data == "TRUE":
            i_standardize_data = True
        else:
            i_standardize_data = False
        dff1 =  pd.read_csv("Stats/RIF_F1.csv")
        f1 = dff1[(dff1['Filename']==filename)&
                    (dff1['ntrees']==i_ntrees)&
                    (dff1['standardize_data']==i_standardize_data)&
                    (dff1['sample_size']==str(i_sample_size))&
                    (dff1['ncols_per_tree']==str(i_ncols_per_tree))]
        
        if f1.empty == 0:
            runs_f1 = []
            for i in range(10):
                runs_f1.append(('R'+str(i)))
            run_f1_values = f1[runs_f1].to_numpy()
            return np.mean(np.mean(run_f1_values))

    

    if os.path.exists("Labels/IF_R/"+labelFile+".csv") == 0:
        frr=open("GD_ReRun/RIF.csv", "a")
        frr.write(filename+","+str(param_r[0][1])+","+str(param_r[1][1])+","+str(param_r[2][1])+","+str(param_r[3][1])+'\n')
        frr.close()
        return -1
    
    labels = []
    f1 = []
    ari = []
    
    labels =  pd.read_csv("Labels/IF_R/"+labelFile+".csv").to_numpy()
    for i in range(10):
        f1.append(metrics.f1_score(gt, np.int64((labels[i][1:])*1)))
        
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
          ari.append(adjusted_rand_score(np.int64((labels[i][1:])*1), np.int64((labels[j][1:])*1)))
          
    flabel_done=open("Labels/IF_R_Done/"+labelFile+".csv", 'a')
    flabel_done.write("Done")
    flabel_done.close()
    
    fstat_f1=open("Stats/RIF_F1.csv", "a")
    fstat_f1.write(filename+','+ str(param_r[0][1]) + ','+ str(param_r[1][1]) + ',' + str(param_r[2][1]) + ',' + str(param_r[3][1]) + ',0,')
    fstat_f1.write(','.join(str(s) for s in f1) + '\n')
    fstat_f1.close()
    
    fstat_ari=open("Stats/RIF_ARI.csv", "a")
    fstat_ari.write(filename+','+ str(param_r[0][1]) + ','+ str(param_r[1][1]) + ',' + str(param_r[2][1]) + ',' + str(param_r[3][1]) + ',0,')
    fstat_ari.write(','.join(str(s) for s in ari) + '\n')
    fstat_ari.close()

    return np.mean(f1)
    
def plot_ari_f1():
    Route_Scores = pd.read_csv("Stats/IF_SvMvR_Route_Scores.csv")
    
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
    
    plt.legend(['Default Setting', 'Uninformed Route', 'Informed Route'], fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.title("Isolation Forest - Inconsistency", fontsize = 15)
    plt.xlabel("Average Performance (F1 Score)", fontsize = 15)
    plt.ylabel("Consistency (Mean Mutual ARI)", fontsize = 15)
    plt.savefig("Fig/IF_SvMvR_GD_Comparison.pdf", bbox_inches="tight", pad_inches=0)
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
    print(Route_Scores['DefaultF1_mat'].mean(), " & ", Route_Scores['UninformedF1_mat'].mean(), " & ", Route_Scores['InformedF1_mat'].mean())
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
    
    master_files.sort()
    
    
    parameters_r = []
    
    ntrees = [2, 4, 8, 16, 32, 64, 100, 128, 256, 512]
    standardize_data = ["TRUE","FALSE"]
    sample_size = ['auto',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,"NULL"]
    ncols_per_tree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,'def']
    
    parameters_r.append(["ntrees",512,ntrees])
    parameters_r.append(["standardize_data","TRUE",standardize_data])
    parameters_r.append(["sample_size",'auto',sample_size])
    parameters_r.append(["ncols_per_tree",'def',ncols_per_tree])
    
    
    parameters_mat = []
    
    # ContaminationFraction = [0, 0.05, 0.1, 0.15, 0.2, 0.25];
    # NumLearners = [1, 2, 4, 8, 16, 32, 64, 100, 128, 256, 512];
    # NumObservationsPerLearner = [0.05, 0.1, 0.2, 0.5, 1];
    
    # parameters_mat.append(["ContaminationFraction", 0.1, ContaminationFraction])
    # parameters_mat.append(["NumLearners", 100, NumLearners])
    # parameters_mat.append(["NumObservationsPerLearner", 'auto', NumObservationsPerLearner])
    
    
    # parameters_sk = []
    # n_estimators = [2, 4, 8, 16, 32, 64, 100, 128, 256, 512]##
    # max_samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # contamination = ['auto'] 
    # max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # bootstrap = [True, False]
    # n_jobs = [1, None] 
    # warm_start = [True, False]
    
    # parameters_sk.append(["n_estimators", 100, n_estimators])
    # parameters_sk.append(["max_samples", 'auto', max_samples])
    # parameters_sk.append(["contamination", 'auto', contamination])
    # parameters_sk.append(["max_features", 1.0, max_features])
    # parameters_sk.append(["bootstrap", False, bootstrap])
    # parameters_sk.append(["n_jobs", None, n_jobs])
    # parameters_sk.append(["warm_start", False, warm_start])
    
    # if os.path.exists("Stats/Inconsistency/IF_mvr.csv") == 0:
    #     fmvr=open("Stats/Inconsistency/IF_mvr.csv", "w")
    #     fmvr.write('Mat,R,ARI\n')
    #     fmvr.close()
    # if os.path.exists("Stats/Inconsistency/IF_rvs.csv") == 0:
    #     frvs=open("Stats/Inconsistency/IF_rvs.csv", "w")
    #     frvs.write('R,Sk,ARI\n')
    #     frvs.close()
    # if os.path.exists("Stats/Inconsistency/IF_mvs.csv") == 0:
    #     fmvs=open("Stats/Inconsistency/IF_mvs.csv", "w")
    #     fmvs.write('Mat,Sk,ARI\n')
    #     fmvs.close()
    
    # f_Route_Scores=open("Stats/IF_SvMvR_Route_Scores.csv", "w")
    # f_Route_Scores.write('Filename,DefaultARI,DefaultF1_r,DefaultF1_mat,DefaultF1_sk,UninformedARI,UninformedF1_r,UninformedF1_mat,UninformedF1_sk,InformedARI,InformedF1_r,InformedF1_mat,InformedF1_sk\n')
    # f_Route_Scores.close()
    
    # frr=open("GD_ReRun/RIF.csv", "w")
    # frr.write('Filename,ntrees,standardize_data,sample_size,ncols_per_tree\n')
    # frr.close()
    
    # frr=open("GD_ReRun/MatIF.csv", "w")
    # frr.write('Filename,ContaminationFraction,NumLearners,NumObservationsPerLearner\n')
    # frr.close()
    
    # for FileNumber in range(len(master_files)):
    #     print(FileNumber, end=' ')
    #     isolationforest(master_files[FileNumber], parameters_r, parameters_mat, parameters_sk)

    # isolationforest("KnuggetChase3", parameters_r, parameters_mat, parameters_sk)
    plot_ari_f1() 

    