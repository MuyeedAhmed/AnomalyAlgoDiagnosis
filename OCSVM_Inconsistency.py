#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:42:30 2022

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


def get_ari_sk_mat(filename, param_sk, param_mat):
    labelFile_sk = filename + "_" + str(param_sk[0][1]) + "_" + str(param_sk[1][1]) + "_" + str(param_sk[2][1]) + "_" + str(param_sk[3][1]) + "_" + str(param_sk[4][1]) + "_" + str(param_sk[5][1]) + "_" + str(param_sk[6][1]) + "_" + str(param_sk[7][1]) + "_" + str(param_sk[8][1])
    labelFile_mat = filename + "_" + str(param_mat[0][1]) + "_" + str(param_mat[1][1]) + "_" + str(param_mat[2][1]) + "_" + str(param_mat[3][1]) + "_" + str(param_mat[4][1]) + "_" + str(param_mat[5][1]) + "_" + str(param_mat[6][1]) + "_" + str(param_mat[7][1])

    if os.path.exists("Labels/OCSVM_Sk/Labels_Sk_OCSVM_"+labelFile_sk+".csv") == 0:
        # print(labelFile_sk)
        return 0

    labels_sk =  pd.read_csv("Labels/OCSVM_Sk/Labels_Sk_OCSVM_"+labelFile_sk+".csv", header=None).to_numpy()

    labels_mat =  pd.read_csv("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile_mat+".csv", header=None).to_numpy()
    
    ari = []
    
    # for i in range(len(labels_sk)):
    for j in range(len(labels_mat)):
        ari.append(adjusted_rand_score(labels_sk[0], labels_mat[j]))
    return ari


def get_ari_sk_r(filename, param_sk, param_r):
    labelFile_sk = filename + "_" + str(param_sk[0][1]) + "_" + str(param_sk[1][1]) + "_" + str(param_sk[2][1]) + "_" + str(param_sk[3][1]) + "_" + str(param_sk[4][1]) + "_" + str(param_sk[5][1]) + "_" + str(param_sk[6][1]) + "_" + str(param_sk[7][1]) + "_" + str(param_sk[8][1])
    labelFile_r = filename + "_" + str(param_r[0][1]) + "_" + str(param_r[1][1]) + "_" + str(param_r[2][1]) + "_" + str(param_r[3][1]) + "_" + str(param_r[4][1]) + "_" + str(param_r[5][1]) + "_" + str(param_r[6][1]) + "_" + str(param_r[7][1]) + "_" + str(param_r[8][1])

    if os.path.exists("Labels/OCSVM_Sk/Labels_Sk_OCSVM_"+labelFile_sk+".csv") == 0:
        # print(labelFile_sk)
        return 0
    

    labels_sk =  pd.read_csv("Labels/OCSVM_Sk/Labels_Sk_OCSVM_"+labelFile_sk+".csv", header=None).to_numpy()    
    
    labels_r = pd.read_csv("Labels/OCSVM_R/"+labelFile_r+".csv").to_numpy()
    labels_r = np.int64((labels_r[0][1:])*1)
    
    
    ari = adjusted_rand_score(labels_sk[0], labels_r)
    return ari


def get_ari_r_mat(filename, param_mat, param_r):
    labelFile_r = filename + "_" + str(param_r[0][1]) + "_" + str(param_r[1][1]) + "_" + str(param_r[2][1]) + "_" + str(param_r[3][1]) + "_" + str(param_r[4][1]) + "_" + str(param_r[5][1]) + "_" + str(param_r[6][1]) + "_" + str(param_r[7][1]) + "_" + str(param_r[8][1])
    labelFile_mat = filename + "_" + str(param_mat[0][1]) + "_" + str(param_mat[1][1]) + "_" + str(param_mat[2][1]) + "_" + str(param_mat[3][1]) + "_" + str(param_mat[4][1]) + "_" + str(param_mat[5][1]) + "_" + str(param_mat[6][1]) + "_" + str(param_mat[7][1])

    if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile_mat+".csv") == 0:
        # print(labelFile_sk)
        return 0
    if os.path.exists("Labels/OCSVM_R/"+labelFile_r+".csv") == 0:
        # print(labelFile_sk)
        return 0

    labels_r = pd.read_csv("Labels/OCSVM_R/"+labelFile_r+".csv").to_numpy()
    labels_r = np.int64((labels_r[0][1:])*1)

    labels_mat =  pd.read_csv("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile_mat+".csv", header=None).to_numpy()
    
    ari = []
    
    for j in range(len(labels_mat)):
        ari.append(adjusted_rand_score(labels_r, labels_mat[j]))
    return ari


def run_mat_r():
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
    
    parameters_mat = []
    
    ContaminationFraction = [0, 0.05, 0.1, 0.15, 0.2, 0.25, "LOF", "IF"];
    KernelScale = [1, "auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5];
    Lambda = ["auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5];
    NumExpansionDimensions = ["auto", 2^12, 2^15, 2^17, 2^19];
    StandardizeData = [0, 1];
    BetaTolerance = [1e-2, 1e-3, 1e-4, 1e-5];
    GradientTolerance = [1e-3, 1e-4, 1e-5, 1e-6];
    IterationLimit = [100, 200, 500, 1000, 2000];
    
    parameters_mat.append(["ContaminationFraction", 0, ContaminationFraction])
    parameters_mat.append(["KernelScale", 1, KernelScale])
    parameters_mat.append(["Lambda", 'auto', Lambda])
    parameters_mat.append(["NumExpansionDimensions", 'auto', NumExpansionDimensions])
    parameters_mat.append(["StandardizeData", 0, StandardizeData])
    parameters_mat.append(["BetaTolerance", 1e-4, BetaTolerance])
    parameters_mat.append(["GradientTolerance", 1e-4, GradientTolerance])
    parameters_mat.append(["IterationLimit", 1000, IterationLimit])
    
    
    parameters_r = []
    
    kernel = ['linear', 'polynomial', 'radial', 'sigmoid']
    degree = [3, 4, 5, 6]
    gamma = ['scale', 'auto']
    coef0 = [0, 0.1, 0.2, 0.3, 0.4]
    tolerance = [0.1, 0.01, 0.001, 0.0001, 0.00001]
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
    
    
    df = pd.DataFrame(columns = ['Filename', 'Configuration', 'Mean ARI', 'Min ARI'])
    
    for file in master_files:
        ari = get_ari_r_mat(file, parameters_mat, parameters_r)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Default", 
                        'Mean ARI' : ari_mean, 
                        'Min ARI' : ari_min}, ignore_index=True)
    
    
    parameters_mat[0][1] = 0.2
    parameters_r[5][1] = 0.2
    for file in master_files:
        ari = get_ari_r_mat(file, parameters_mat, parameters_r)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 1", 
                        'Mean ARI' : ari_mean, 
                        'Min ARI' : ari_min}, ignore_index=True)
        
    
    parameters_mat[0][1] = "IF"
    parameters_mat[1][1] = 'auto'
    parameters_r[2][1] = "auto"
    parameters_r[5][1] = "IF"
    for file in master_files:
        ari = get_ari_r_mat(file, parameters_mat, parameters_r)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 2", 
                        'Mean ARI' : ari_mean, 
                        'Min ARI' : ari_min}, ignore_index=True)
    
    
    parameters_r[2][1] = "scale"
    parameters_r[0][1] = "polynomial"
    for file in master_files:
        ari = get_ari_r_mat(file, parameters_mat, parameters_r)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 3", 
                        'Mean ARI' : ari_mean, 
                        'Min ARI' : ari_min}, ignore_index=True)
    
        
    
    
    
    fig = plt.Figure()
    axmean = sns.boxplot(x="Configuration", y="Mean ARI", data=df)
    axmean.set(xlabel=None)
    plt.title("One Class SVM - Matlab VS R")
    plt.savefig("Fig/BoxPlot/OCSVM_MatR_MeanARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    df.to_csv("Stats/OCSVM_MvR.csv")
    fig = plt.Figure()
    axmin = sns.boxplot(x="Configuration", y="Min ARI", data=df)
    axmin.set(xlabel=None)
    plt.title("One Class SVM - Matlab VS R")
    plt.savefig("Fig/BoxPlot/OCSVM_MatR_MinARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
def run_sk_r():
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
    
    parameters_sk = []
    
    kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    degree = [3, 4, 5, 6] # Kernel poly only
    gamma = ['scale', 'auto'] # Kernel ‘rbf’, ‘poly’ and ‘sigmoid’
    coef0 = [0.0, 0.1, 0.2, 0.3, 0.4] # Kernel ‘poly’ and ‘sigmoid’
    tol = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    nu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    shrinking = [True, False]
    cache_size = [50, 100, 200, 400]
    max_iter = [50, 100, 150, 200, 250, 300, -1]
    
    parameters_sk.append(["kernel", 'rbf', kernel])
    parameters_sk.append(["degree", 3, degree])
    parameters_sk.append(["gamma", 'scale', gamma])
    parameters_sk.append(["coef0", 0.0, coef0])
    parameters_sk.append(["tol", 1e-3, tol])
    parameters_sk.append(["nu", 0.5, nu])
    parameters_sk.append(["shrinking", True, shrinking])
    parameters_sk.append(["cache_size", 200, cache_size])
    parameters_sk.append(["max_iter", -1, max_iter])
    
    
    parameters_r = []
    
    kernel = ['linear', 'polynomial', 'radial', 'sigmoid']
    degree = [3, 4, 5, 6]
    gamma = ['scale', 'auto']
    coef0 = [0, 0.1, 0.2, 0.3, 0.4]
    tolerance = [0.1, 0.01, 0.001, 0.0001, 0.00001]
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
    
    
    df = pd.DataFrame(columns = ['Filename', 'Configuration', 'Mean ARI', 'Min ARI'])
    
    for file in master_files:
        ari = get_ari_sk_r(file, parameters_sk, parameters_r)
        if ari == 0:
            continue
        ari_mean = ari
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Default", 
                        'Mean ARI' : ari_mean, 
                        'Min ARI' : 0}, ignore_index=True)
    
    
    parameters_r[5][1] = 0.2
    parameters_sk[5][1] = 0.2
    for file in master_files:
        ari = get_ari_sk_r(file, parameters_sk, parameters_r)
        if ari == 0:
            continue
        ari_mean = ari
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 1", 
                        'Mean ARI' : ari_mean, 
                        'Min ARI' : 0}, ignore_index=True)
        
    parameters_r[2][1] = "auto"
    parameters_r[5][1] = "IF"
    parameters_sk[5][1] = "IF"
    for file in master_files:
        ari = get_ari_sk_r(file, parameters_sk, parameters_r)
        if ari == 0:
            continue
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 2",
                        'Mean ARI' : ari,
                        'Min ARI' : 0}, ignore_index=True)
    
    parameters_r[2][1] = "scale"
    parameters_r[0][1] = "polynomial"
    for file in master_files:
        ari = get_ari_sk_r(file, parameters_sk, parameters_r)
        if ari == 0:
            continue
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 3",
                        'Mean ARI' : ari,
                        'Min ARI' : 0}, ignore_index=True)

    parameters_r[5][1] = 0.2
    parameters_sk[5][1] = 0.2
    
    # parameters_sk[0][1] = "poly"
    for file in master_files:
        ari = get_ari_sk_r(file, parameters_sk, parameters_r)
        if ari == 0:
            continue
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 4", 
                        'Mean ARI' : ari,
                        'Min ARI' : 0}, ignore_index=True)
    
    
    fig = plt.Figure()
    axmean = sns.boxplot(x="Configuration", y="Mean ARI", data=df)
    axmean.set(xlabel=None)
    plt.title("One Class SVM - Scikit-learn VS R")
    plt.savefig("Fig/BoxPlot/OCSVM_SkR_MeanARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    df.to_csv("Stats/OCSVM_SvR.csv")
    
def run_sk_mat():
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()
    
    parameters_sk = []
    
    kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    degree = [3, 4, 5, 6] # Kernel poly only
    gamma = ['scale', 'auto'] # Kernel ‘rbf’, ‘poly’ and ‘sigmoid’
    coef0 = [0.0, 0.1, 0.2, 0.3, 0.4] # Kernel ‘poly’ and ‘sigmoid’
    tol = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    nu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    shrinking = [True, False]
    cache_size = [50, 100, 200, 400]
    max_iter = [50, 100, 150, 200, 250, 300, -1]
    
    parameters_sk.append(["kernel", 'rbf', kernel])
    parameters_sk.append(["degree", 3, degree])
    parameters_sk.append(["gamma", 'scale', gamma])
    parameters_sk.append(["coef0", 0.0, coef0])
    parameters_sk.append(["tol", 1e-3, tol])
    parameters_sk.append(["nu", 0.5, nu])
    parameters_sk.append(["shrinking", True, shrinking])
    parameters_sk.append(["cache_size", 200, cache_size])
    parameters_sk.append(["max_iter", -1, max_iter])
    
    
    parameters_mat = []
    
    ContaminationFraction = [0, 0.05, 0.1, 0.15, 0.2, 0.25, "LOF", "IF"];
    KernelScale = [1, "auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5];
    Lambda = ["auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5];
    NumExpansionDimensions = ["auto", 2^12, 2^15, 2^17, 2^19];
    StandardizeData = [0, 1];
    BetaTolerance = [1e-2, 1e-3, 1e-4, 1e-5];
    GradientTolerance = [1e-3, 1e-4, 1e-5, 1e-6];
    IterationLimit = [100, 200, 500, 1000, 2000];
    
    parameters_mat.append(["ContaminationFraction", 0, ContaminationFraction])
    parameters_mat.append(["KernelScale", 1, KernelScale])
    parameters_mat.append(["Lambda", 'auto', Lambda])
    parameters_mat.append(["NumExpansionDimensions", 'auto', NumExpansionDimensions])
    parameters_mat.append(["StandardizeData", 0, StandardizeData])
    parameters_mat.append(["BetaTolerance", 1e-4, BetaTolerance])
    parameters_mat.append(["GradientTolerance", 1e-4, GradientTolerance])
    parameters_mat.append(["IterationLimit", 1000, IterationLimit])
    
    
    df = pd.DataFrame(columns = ['Filename', 'Configuration', 'Mean ARI', 'Min ARI'])
    
    for file in master_files:
        ari = get_ari_sk_mat(file, parameters_sk, parameters_mat)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Default", 
                        'Mean ARI' : ari_mean, 
                        'Min ARI' : ari_min}, ignore_index=True)
    
    
    parameters_mat[0][1] = 0.2
    parameters_mat[1][1] = 1
    for file in master_files:
        ari = get_ari_sk_mat(file, parameters_sk, parameters_mat)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 1", 
                        'Mean ARI' : ari_mean, 
                        'Min ARI' : ari_min}, ignore_index=True)
        
    
    parameters_mat[0][1] = "IF"
    parameters_mat[1][1] = 'auto'
    parameters_sk[5][1] = "IF"
    for file in master_files:
        ari = get_ari_sk_mat(file, parameters_sk, parameters_mat)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 2", 
                        'Mean ARI' : ari_mean,
                        'Min ARI' : ari_min}, ignore_index=True)


    parameters_mat[2][1] = 0.2
    parameters_mat[0][1] = "LOF"
    parameters_sk[5][1] = "LOF"
    for file in master_files:
        ari = get_ari_sk_mat(file, parameters_sk, parameters_mat)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 3", 
                        'Mean ARI' : ari_mean,
                        'Min ARI' : ari_min}, ignore_index=True)
    
    
    parameters_mat[2][1] = 'auto'
    parameters_mat[0][1] = "LOF"
    parameters_sk[5][1] = "LOF"
    for file in master_files:
        ari = get_ari_sk_mat(file, parameters_sk, parameters_mat)
        if ari == 0:
            continue
        ari_mean = np.mean(ari)
        ari_min = np.min(ari)
        
        df = df.append({'Filename' : file,
                        'Configuration' : "Configure 4", 
                        'Mean ARI' : ari_mean,
                        'Min ARI' : ari_min}, ignore_index=True)
    
    fig = plt.Figure()
    axmean = sns.boxplot(x="Configuration", y="Mean ARI", data=df)
    axmean.set(xlabel=None)
    plt.title("One Class SVM - Scikit-learn VS Matlab")
    plt.savefig("Fig/BoxPlot/OCSVM_SkMat_MeanARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    df.to_csv("Stats/OCSVM_SvM.csv")
    fig = plt.Figure()
    axmin = sns.boxplot(x="Configuration", y="Min ARI", data=df)
    axmin.set(xlabel=None)
    plt.title("One Class SVM - Scikit-learn VS Matlab")
    plt.savefig("Fig/BoxPlot/OCSVM_SkMat_MinARI.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    

if __name__ == '__main__':
    run_sk_mat()
    run_sk_r()
    run_mat_r()
