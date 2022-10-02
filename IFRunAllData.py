from datetime import datetime
import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
import collections
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn import metrics
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics.cluster import adjusted_rand_score

datasetFolderDir = '/Users/muyeedahmed/Desktop/Research/Dataset/Dataset_Anomaly/'


def isolationforest(filename):
    print(filename)
    folderpath = datasetFolderDir
    # folderpath = 'Datasets/'
    # print(f"File name: {filename}", end=' ')
    # if os.path.getsize(folderpath+filename+".csv") > 100000: # 10MB
    #     print("Didn\'t run -> Too large")    
    #     return
    
    if os.path.exists(folderpath+filename+".mat") == 0:
        print("File doesn't exist")
        return
    
    try:
        df = loadmat(folderpath+filename+".mat")
    except NotImplementedError:
        df = mat73.loadmat(folderpath+filename+".mat")
    #n_cluster=X["target"].nunique()
    #print(type(df['X']))
    # target=df["y"]
    gt=df["y"]
    gt = gt.reshape((len(gt)))
    X=df['X']
    if np.isnan(X).any():
        print("Didn\'t run -> NaN")
        return
    
    n_estimators = [2, 4, 8, 16, 32, 64, 100, 128, 256, 512, 1024]
    max_samples = ['auto', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    contamination = ['auto'] 
    max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bootstrap = [True, False]
    n_jobs = [1, -1] 
    warm_start = [True, False]
    
    # for ne in n_estimators:
    #     for ms in max_samples:
    #         for cont in contamination:    
    #             for mf in max_features:
    #                 for bs in bootstrap:
    #                     for nj in n_jobs:
    #                         for ws in warm_start:
    #                             runIF(filename, X, gt, ne, ms, cont, mf, bs, nj, ws)
    
    for mf in max_features:
        runIF(filename, X, gt, i_max_features=mf)
    
    
    # for ws in warm_start:
    #     runIF(filename, X, gt, i_warm_start=ws)
    
    
def runIF(filename, X, gt, i_n_estimators=100, i_max_samples='auto', i_contamination='auto', 
          i_max_features=1.0, i_bootstrap=False, i_n_jobs=None, i_warm_start=False):
    labelFile = filename + "_" + str(i_n_estimators) + "_" + str(i_max_samples) + "_" + str(i_contamination) + "_" + str(i_max_features) + "_" + str(i_bootstrap) + "_" + str(i_n_jobs) + "_" + str(i_warm_start)

    for i in range(30):
        clustering = IsolationForest(n_estimators=i_n_estimators, max_samples=i_max_samples, 
                                     max_features=i_max_features, bootstrap=i_bootstrap, 
                                     n_jobs=i_n_jobs, warm_start=i_warm_start).fit(X)
    
        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        
        flabel=open("IF/Labels_"+labelFile+".csv", 'a')
        flabel.write(','.join(str(s) for s in l) + '\n')
        flabel.close()


def calculateF1Score_WS(allFiles):
    i_n_estimators=100
    i_max_samples='auto'
    i_contamination='auto' 
    i_max_features=1.0
    i_bootstrap=False
    i_n_jobs=None
    warm_start = [True, False]
    

    accuracy_range_all = []
    accuracy_med_all = []
    f1_range_all = []
    f1_med_all = []

    for filename in allFiles:
        print(filename)
        folderpath = datasetFolderDir
    
        if os.path.exists(folderpath+filename+".mat") == 0:
            print("File doesn't exist")
            return
        try:
            df = loadmat(folderpath+filename+".mat")
        except NotImplementedError:
            df = mat73.loadmat(folderpath+filename+".mat")

        gt=df["y"]
        gt = gt.reshape((len(gt)))
        X=df['X']
        if np.isnan(X).any():
            print("Didn\'t run -> NaN")
            return
        for i_warm_start in warm_start:
            labelFile = "IF/Labels_"+filename + "_" + str(i_n_estimators) + "_" + str(i_max_samples) + "_" + str(i_contamination) + "_" + str(i_max_features) + "_" + str(i_bootstrap) + "_" + str(i_n_jobs) + "_" + str(i_warm_start)
            labels = pd.read_csv(labelFile+'.csv', header=None).to_numpy()
    
            accuracy = [0] * 30
            f1 = [0] * 30
            ari = [0] * 30
            for run in range(30):
                l = labels[run]
                accuracy[run] = metrics.accuracy_score(gt, l)
                f1[run] = metrics.f1_score(gt, l)
                # ari[run] = adjusted_rand_score(gt, l)
                
            accDiff = np.percentile(accuracy, 75) - np.percentile(accuracy, 25)
            accuracy_range_all.append([i_warm_start, accDiff])
            accuracy_med_all.append([i_warm_start, np.percentile(accuracy, 50)])
            
            f1Diff = np.percentile(f1, 75) - np.percentile(f1, 25)
            f1_range_all.append([i_warm_start, f1Diff])
            f1_med_all.append([i_warm_start, np.percentile(f1, 50)])
        # if filename=='ionosphere':
        #     break
    df_acc_r = pd.DataFrame(accuracy_range_all, columns = ['warm_start', 'Accuracy_Range'])
    df_acc_m = pd.DataFrame(accuracy_med_all, columns = ['warm_start', 'Accuracy_Median'])
    df_f1_r = pd.DataFrame(f1_range_all, columns = ['warm_start', 'F1Score_Range'])
    df_f1_m = pd.DataFrame(f1_med_all, columns = ['warm_start', 'F1Score_Median'])
    
    
    fig = plt.Figure()
    axf = sns.boxplot(x="warm_start", y="Accuracy_Range", data=df_acc_r)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_warm_start_Accuracy_Range.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axf = sns.boxplot(x="warm_start", y="Accuracy_Median", data=df_acc_m)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_warm_start_Accuracy_Median.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axf = sns.boxplot(x="warm_start", y="F1Score_Range", data=df_f1_r)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_warm_start_F1Score_Range.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axf = sns.boxplot(x="warm_start", y="F1Score_Median", data=df_f1_m)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_warm_start_F1Score_Median.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()


def calculateF1Score_MF(allFiles):
    i_n_estimators=100
    i_max_samples='auto'
    i_contamination='auto' 
    max_features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    i_bootstrap=False
    i_n_jobs=None
    i_warm_start = False
    

    accuracy_range_all = []
    accuracy_med_all = []
    f1_range_all = []
    f1_med_all = []

    for filename in allFiles:
        print(filename)
        folderpath = datasetFolderDir
    
        if os.path.exists(folderpath+filename+".mat") == 0:
            print("File doesn't exist")
            return
        try:
            df = loadmat(folderpath+filename+".mat")
        except NotImplementedError:
            df = mat73.loadmat(folderpath+filename+".mat")

        gt=df["y"]
        gt = gt.reshape((len(gt)))
        X=df['X']
        if np.isnan(X).any():
            print("Didn\'t run -> NaN")
            return
        for i_max_features in max_features:
            labelFile = "IF/Labels_"+filename + "_" + str(i_n_estimators) + "_" + str(i_max_samples) + "_" + str(i_contamination) + "_" + str(i_max_features) + "_" + str(i_bootstrap) + "_" + str(i_n_jobs) + "_" + str(i_warm_start)
            labels = pd.read_csv(labelFile+'.csv', header=None).to_numpy()
    
            accuracy = [0] * 30
            f1 = [0] * 30
            ari = [0] * 30
            for run in range(30):
                l = labels[run]
                accuracy[run] = metrics.accuracy_score(gt, l)
                f1[run] = metrics.f1_score(gt, l)
                # ari[run] = adjusted_rand_score(gt, l)
                
            accDiff = np.percentile(accuracy, 75) - np.percentile(accuracy, 25)
            accuracy_range_all.append([i_max_features, accDiff])
            accuracy_med_all.append([i_max_features, np.percentile(accuracy, 50)])
            
            f1Diff = np.percentile(f1, 75) - np.percentile(f1, 25)
            f1_range_all.append([i_max_features, f1Diff])
            f1_med_all.append([i_max_features, np.percentile(f1, 50)])
        # if filename=='ionosphere':
        #     break
    df_acc_r = pd.DataFrame(accuracy_range_all, columns = ['max_features', 'Accuracy_Range'])
    df_acc_m = pd.DataFrame(accuracy_med_all, columns = ['max_features', 'Accuracy_Median'])
    df_f1_r = pd.DataFrame(f1_range_all, columns = ['max_features', 'F1Score_Range'])
    df_f1_m = pd.DataFrame(f1_med_all, columns = ['max_features', 'F1Score_Median'])
    
    
    fig = plt.Figure()
    axf = sns.boxplot(x="max_features", y="Accuracy_Range", data=df_acc_r)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_max_features_Accuracy_Range.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axf = sns.boxplot(x="max_features", y="Accuracy_Median", data=df_acc_m)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_max_features_Accuracy_Median.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axf = sns.boxplot(x="max_features", y="F1Score_Range", data=df_f1_r)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_max_features_F1Score_Range.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axf = sns.boxplot(x="max_features", y="F1Score_Median", data=df_f1_m)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_max_features_F1Score_Median.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()



if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files = glob.glob(folderpath+"*.mat")
    #labels_df = pd.DataFrame(columns = ['Dataset','labels_pred'])
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    #done_files = pd.read_csv("AP/donefiles.csv", header=None).to_numpy()
    #master_files = [i for i in master_files if i not in done_files]
    master_files.sort()
    
    # for FileNumber in range(len(master_files)):
    #     isolationforest(master_files[FileNumber])
    calculateF1Score_WS(master_files)
    # calculateF1Score_MF(master_files)
    
    
    
    