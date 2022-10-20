import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
import math


def calculate():
    df_f1 = pd.read_csv("Stats/ROCSVM_F1.csv")
    
    runs_r = []
    runs_ari = []
    
    for i in range(10):
        runs_r.append(('R'+str(i)))
    for i in range(45):
        runs_ari.append(('R'+str(i)))

    df_f1["F1_Median"] = 0
    
    for i in range(df_f1.shape[0]):
        run_values = df_f1.loc[i, runs_r].tolist()
        
        range_ = (np.percentile(run_values, 75) - np.percentile(run_values, 25))/(np.percentile(run_values, 75) + np.percentile(run_values, 25))
        if math.isnan(range_):
            range_ = 0
            
        df_f1.iloc[i, df_f1.columns.get_loc('F1_Median')] =  np.mean(run_values)
        df_f1.iloc[i, df_f1.columns.get_loc('F1_Range')] = range_
    df_f1 = df_f1.drop(columns=runs_r, axis=1)
        
    parameter_names = ["kernel","degree","gamma","coef0","tolerance","nu","shrinking","cachesize","epsilon"]
    median_df = df_f1.groupby(parameter_names)[["F1_Median"]].mean()
    median_df = median_df.reset_index()
    
    median_df.to_csv("Stats/ROCSVM_Grouped_Median.csv", index=False)

if __name__ == '__main__':
    calculate()
        
        
        