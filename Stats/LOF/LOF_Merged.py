# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 17:02:28 2022

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

def compute_ari(filename):
    print(filename)
    name = filename.split('.')[0]
    default_mat = pd.read_csv('C:/Users/parth/School/Clustering/LOF_Default_MatLab_Labels/'+name+".csv",header=None)
    default_r = pd.read_csv('C:/Users/parth/School/Clustering/LOF_R_Default_Labels/'+name+".csv")
    default_py = pd.read_csv('C:/Users/parth/School/Clustering/LOF_Python_Default_Labels/'+name+".csv")
    py_r_def_ari = adjusted_rand_score(default_py.iloc[:,0],default_r.iloc[:,0])
    py_mat_def_ari = adjusted_rand_score(default_py.iloc[:,0],default_mat.iloc[:,0])
    mat_r_def_ari = adjusted_rand_score(default_mat.iloc[:,0],default_r.iloc[:,0])
    try:
        modified_mat = pd.read_csv('C:/Users/parth/School/Clustering/LOF_Modified_MatLab_Labels/'+name+".csv",header=None)
    except :
        modified_mat = []
    try:
        modified_r = pd.read_csv('C:/Users/parth/School/Clustering/LOF_R_Modified_Labels/'+name+".csv")
        modified_r = pd.read_csv('C:/Users/parth/School/Clustering/LOF_R_Modified_Labels/'+name+".csv")
    except:
        modified_r = []
    try:
        modified_py = pd.read_csv('C:/Users/parth/School/Clustering/LOF_Python_Modified_Labels/'+name+".csv")
    except:
        modified_py = []
    try:
        py_r_mod_ari = adjusted_rand_score(modified_py.iloc[:,0],modified_r.iloc[:,0])
    except:
        py_r_mod_ari = 0
    try:
        py_mat_mod_ari = adjusted_rand_score(modified_py.iloc[:,0],modified_mat.iloc[:,0])
    except:
        py_mat_mod_ari = 0
    try:
        mat_r_mod_ari = adjusted_rand_score(modified_mat.iloc[:,0],modified_r.iloc[:,0])
    except:
        mat_r_mod_ari = 0
    return (py_r_def_ari,py_mat_def_ari,mat_r_def_ari,py_r_mod_ari,py_mat_mod_ari,mat_r_mod_ari)
    




if __name__ == '__main__':
    matlab_df = pd.read_csv(r"C:\Users\parth\School\Clustering\anomaly_matlab_threshold.csv")
    master_files = matlab_df['Dataset']
    ari_df = pd.DataFrame(columns=['Dataset','SKlearn_R_Modified_ARI','SKlearn_R_Default_ARI','SKlearn_Matlab_Modified_ARI','SKlearn_Matlab_Default_ARI','Matlab_R_Modified_ARI','Matlab_R_Default_ARI'])
    for FileNumber in range(len(master_files)):
        
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        py_r_def_ari,py_mat_def_ari,mat_r_def_ari,py_r_mod_ari,py_mat_mod_ari,mat_r_mod_ari = compute_ari(master_files[FileNumber])
        ari_df = ari_df.append({'Dataset':master_files[FileNumber],'SKlearn_R_Modified_ARI':py_r_mod_ari,'SKlearn_R_Default_ARI':py_r_def_ari,'SKlearn_Matlab_Modified_ARI':py_mat_mod_ari,'SKlearn_Matlab_Default_ARI':py_mat_def_ari,'Matlab_R_Modified_ARI':mat_r_mod_ari,'Matlab_R_Default_ARI':mat_r_def_ari},ignore_index=True)
        print()
    ari_df.to_csv(r'C:\Users\parth\School\Clustering\lof_ari_all.csv',index=False)
    df1 = pd.read_csv(r'C:\Users\parth\School\Clustering\anomaly_python.csv')
    df2 = pd.read_csv(r'C:\Users\parth\School\Clustering\anomaly_r.csv')
    df3= pd.read_csv(r'C:\Users\parth\School\Clustering\anomaly_matlab.csv')
    df4= pd.read_csv(r'C:\Users\parth\School\Clustering\lof_ari_all.csv')

    # Merge the two dataframes, using _ID column as key
    df5 = pd.concat([df1,df2,df3,df4],axis=1)

    # Write it to a new CSV file
    df5.to_csv(r'C:\Users\parth\School\Clustering\lof_merged.csv')