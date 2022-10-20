# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:30:15 2022

@author: parth
"""

from datetime import datetime
import os
import glob
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import mat73
from sklearn.metrics import f1_score
from scipy.io import loadmat

def calculatef1score(filename,threshold):
    print(filename)
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
    name = filename.split('.')[0]
    default = pd.read_csv('C:/Users/parth/School/Clustering/LOF_Default_MatLab_Labels/'+name+".csv",header=None)
    f1score_def = f1_score(target,default.iloc[:,0])
    if (threshold == 0):
        f1score = 0
        return (threshold,f1score,f1score_def)
    modified = pd.read_csv('C:/Users/parth/School/Clustering/LOF_Modified_MatLab_Labels/'+name+".csv",header=None)
    f1score = f1_score(target,modified.iloc[:,0])
    return (threshold,f1score,f1score_def)
                    
    

if __name__ == '__main__':
    matlab_df = pd.read_csv(r"C:\Users\parth\School\Clustering\anomaly_matlab_threshold.csv")
    master_files = matlab_df['Dataset']
    threshold = matlab_df['Anomaly_Matlab']
    anomaly_matlab = pd.DataFrame(columns=['Dataset','Anomaly_Matlab','F1Score_Matlab_Modified','F1Score_Matlab_Default'])
    for FileNumber in range(len(master_files)):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        thres,f1score,f1score_default= calculatef1score(master_files[FileNumber],threshold[FileNumber])
        anomaly_matlab = anomaly_matlab.append({'Dataset':master_files[FileNumber],'Anomaly_Matlab':thres,'F1Score_Matlab_Modified':f1score,'F1Score_Matlab_Default':f1score_default},ignore_index=True)
        print()
    anomaly_matlab.to_csv(r'C:\Users\parth\School\Clustering\anomaly_matlab.csv',index=False)