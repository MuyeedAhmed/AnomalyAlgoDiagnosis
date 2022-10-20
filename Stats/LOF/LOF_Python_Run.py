# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 15:24:02 2022

@author: parth
"""

from datetime import datetime
import os
import glob
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import mat73
from scipy.io import loadmat
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

def localoutlierfactor(filename):
    folderpath = 'C:/Users/parth/School/Clustering/Dataset_Combined/'
    # if os.path.exists(folderpath+filename+".mat") == 0:
    #     print("File doesn't exist")
    #     return
    if filename.endswith(".mat"):
        try:
            df = loadmat(folderpath+filename)
        except NotImplementedError:
            df = mat73.loadmat(folderpath+filename)
        target=df["y"]
        X=df['X']
    elif filename.endswith(".csv"):
        df = pd.read_csv(folderpath+filename)
        target = df['target']
        X = df.drop(['target'],axis=1)
    labels_pred = LocalOutlierFactor(algorithm='kd_tree',n_neighbors=20).fit_predict(X)
    labels_pred_default = LocalOutlierFactor().fit_predict(X)
    
    #print(labels_pred_default)
    for i in range(len(labels_pred)):
        if labels_pred[i] == -1:
            labels_pred[i] = 1
        elif labels_pred[i] == 1:
            labels_pred[i] = 0
    for i in range(len(labels_pred_default)):
        if labels_pred_default[i] == -1:
            labels_pred_default[i] = 1
        elif labels_pred_default[i] == 1:
            labels_pred_default[i] = 0
    name = filename.split('.')[0]
    #print(name)
    labels_pred_df = pd.DataFrame(labels_pred)
    labels_pred_df.to_csv('C:/Users/parth/School/Clustering/LOF_Python_Modified_Labels/'+name+".csv",index=False)
    labels_pred_def_df = pd.DataFrame(labels_pred_default)
    labels_pred_def_df.to_csv('C:/Users/parth/School/Clustering/LOF_Python_Default_Labels/'+name+".csv",index=False)
    anomaly = ((labels_pred == 1).sum() / len(labels_pred))*100
    #f1score
    f1score = f1_score(target, labels_pred,average="micro")
    #f1score_default
    f1score_default = f1_score(target, labels_pred_default,average="micro")
    return anomaly,f1score,f1score_default
    
    



if __name__ == '__main__':
    folderpath = 'C:/Users/parth/School/Clustering/Dataset_Combined/'
    master_files_mat = glob.glob(folderpath+"*.mat")
    master_files_csv = glob.glob(folderpath+"*.csv")
    master_files = np.append(master_files_csv,master_files_mat).tolist()
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("\\")[-1]
    master_files.sort()
    anomaly_df = pd.DataFrame(columns=['Dataset','Anomaly_Python','F1Score_Python_Modified','F1Score_Python_Default'])
    for FileNumber in range(len(master_files)):
        
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        anomaly_perc,f1score,f1score_default= localoutlierfactor(master_files[FileNumber])
        anomaly_df = anomaly_df.append({'Dataset':master_files[FileNumber],'Anomaly_Python':anomaly_perc,'F1Score_Python_Modified':f1score,'F1Score_Python_Default':f1score_default},ignore_index=True)
        print()
    anomaly_df.to_csv(r'C:\Users\parth\School\Clustering\anomaly_python.csv',index=False)
