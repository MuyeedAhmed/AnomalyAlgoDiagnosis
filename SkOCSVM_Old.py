from sklearn.svm import OneClassSVM
import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
import collections
import numpy as np
from sklearn import metrics
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics.cluster import adjusted_rand_score

datasetFolderDir = '/home/neamtiu/Desktop/ma234/AnomalyDetection/Dataset/'


def ocsvm(filename):
    print(filename)
    folderpath = datasetFolderDir
    
    if os.path.exists(folderpath+filename+".mat") == 1:
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
        if os.path.getsize(folderpath+filename+".csv") > 1000000: # 10MB
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
    
    kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    degree = [3, 4, 5, 6] # Kernel poly only
    gamma = ['scale', 'auto'] # Kernel ‘rbf’, ‘poly’ and ‘sigmoid’
    coef0 = [0.0, 0.1, 0.2, 0.3, 0.4] # Kernel ‘poly’ and ‘sigmoid’
    tol = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    nu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    shrinking = [True, False]
    cache_size = [50, 100, 200, 400]
    max_iter = [50, 100, 150, 200, 250, 300, -1]
    
    for k in kernel:
        runIF(filename, X, gt, i_kernel=k)
    # for d in degree:
    #     runIF(filename, X, gt, =d)
    
    for g in gamma:
        runIF(filename, X, gt, i_gamma=g)
    
    # for c in coef0:
    #     runIF(filename, X, gt, =c)
    
    for t in tol:
        runIF(filename, X, gt, i_tol=t)
    
    for n in nu:
        runIF(filename, X, gt, i_nu=n)
        
    for s in shrinking:
        runIF(filename, X, gt, i_shrinking=s)
        
    for cs in cache_size:
        runIF(filename, X, gt, i_cache_size=cs)
        
    for mi in max_iter:
        runIF(filename, X, gt, i_max_iter=mi)
        
    
    
def runIF(filename, X, gt, i_kernel='rbf', i_gamma='scale', i_tol=1e-3, i_nu = 0.5, 
          i_shrinking = True, i_cache_size = 200, i_max_iter = -1):
    
    labelFile = filename + "_" + i_kernel + "_" + i_gamma + "_" + str(i_tol) + "_" + str(i_nu) + "_" + str(i_shrinking) + "_" + str(i_cache_size) + "_" + str(i_max_iter)
    
    if os.path.exists("OCSVM/Labels_"+labelFile+".csv") == 1:
        print("The Labels Already Exist")
        print("OCSVM/Labels_"+labelFile+".csv")
        return
    
    for i in range(30):
        try:
            clustering = OneClassSVM(kernel=i_kernel, gamma=i_gamma, tol=i_tol, nu=i_nu, 
                                     shrinking=i_shrinking, cache_size=i_cache_size, max_iter=i_max_iter).fit(X)
            l = clustering.predict(X)
            l = [0 if x == 1 else 1 for x in l]
            
            flabel=open("OCSVM/Labels_"+labelFile+".csv", 'a')
            flabel.write(','.join(str(s) for s in l) + '\n')
            flabel.close()
        except:
            print("Issue - ", filename)
            return

def calculate_draw_score(allFiles, parameter, parameter_values):
    i_kernel = 'rbf'
    # i_degree = 3 # Kernel poly only
    i_gamma = 'scale' # Kernel ‘rbf’, ‘poly’ and ‘sigmoid’
    # i_coef0 = 0.0 # Kernel ‘poly’ and ‘sigmoid’
    i_tol = 1e-3
    i_nu = 0.5
    i_shrinking = True
    i_cache_size = 200
    i_max_iter = -1
    
    
    
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
            continue
        for p in parameter_values:
            if parameter == 'kernel':
                i_kernel = p
            elif parameter == 'gamma':
                i_gamma = p
            elif parameter == 'tol':
                i_tol = p
            elif parameter == 'nu':
                i_nu = p
            elif parameter == 'shrinking':
                i_shrinking = p
            elif parameter == 'cache_size':
                i_cache_size = p
            elif parameter == 'max_iter':
                i_max_iter = p
            labelFile = "OCSVM/Labels_"+filename + "_" + i_kernel + "_" + i_gamma + "_" + str(i_tol) + "_" + str(i_nu) + "_" + str(i_shrinking) + "_" + str(i_cache_size) + "_" + str(i_max_iter)

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
            accuracy_range_all.append([p, accDiff])
            accuracy_med_all.append([p, np.percentile(accuracy, 50)])
            
            f1Diff = np.percentile(f1, 75) - np.percentile(f1, 25)
            f1_range_all.append([p, f1Diff])
            f1_med_all.append([p, np.percentile(f1, 50)])
        # if filename=='ionosphere':
        #     break
    df_acc_r = pd.DataFrame(accuracy_range_all, columns = [parameter, 'Accuracy_Range'])
    df_acc_m = pd.DataFrame(accuracy_med_all, columns = [parameter, 'Accuracy_Median'])
    df_f1_r = pd.DataFrame(f1_range_all, columns = [parameter, 'F1Score_Range'])
    df_f1_m = pd.DataFrame(f1_med_all, columns = [parameter, 'F1Score_Median'])
    
    
    fig = plt.Figure()
    axf = sns.boxplot(x=parameter, y="Accuracy_Range", data=df_acc_r)
    # axf.set(xlabel=None)
    plt.savefig("Fig/OCSVM_SK_"+parameter+"_Accuracy_Range.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axf = sns.boxplot(x=parameter, y="Accuracy_Median", data=df_acc_m)
    # axf.set(xlabel=None)
    plt.savefig("Fig/OCSVM_SK_"+parameter+"_Accuracy_Median.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axf = sns.boxplot(x=parameter, y="F1Score_Range", data=df_f1_r)
    # axf.set(xlabel=None)
    plt.savefig("Fig/OCSVM_SK_"+parameter+"_F1Score_Range.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axf = sns.boxplot(x=parameter, y="F1Score_Median", data=df_f1_m)
    # axf.set(xlabel=None)
    plt.savefig("Fig/OCSVM_SK_"+parameter+"_F1Score_Median.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    



if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    #done_files = pd.read_csv("AP/donefiles.csv", header=None).to_numpy()
    #master_files = [i for i in master_files if i not in done_files]
    master_files.sort()
    
    # print(len(master_files))
    for FileNumber in range(len(master_files)):
        ocsvm(master_files[FileNumber])
    
    kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    degree = [3, 4, 5, 6] # Kernel poly only
    gamma = ['scale', 'auto'] # Kernel ‘rbf’, ‘poly’ and ‘sigmoid’
    coef0 = [0.0, 0.1, 0.2, 0.3, 0.4] # Kernel ‘poly’ and ‘sigmoid’
    tol = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    nu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    shrinking = [True, False]
    cache_size = [50, 100, 200, 400]
    max_iter = [50, 100, 150, 200, 250, 300, -1]
    
    # calculate_draw_score(master_files, 'kernel', kernel)
    # calculate_draw_score(master_files, 'degree', degree)
    # calculate_draw_score(master_files, 'gamma', gamma)
    # calculate_draw_score(master_files, 'coef0', coef0)
    # calculate_draw_score(master_files, 'tol', tol)
    # calculate_draw_score(master_files, 'nu', nu)
    # calculate_draw_score(master_files, 'shrinking', shrinking)
    # calculate_draw_score(master_files, 'cache_size', cache_size)
    # calculate_draw_score(master_files, 'max_iter', max_iter)
    