#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 21:00:58 2022

@author: muyeedahmed
"""

def calculate_draw_score(allFiles, parameter, parameter_values):
    i_n_estimators=100
    i_max_samples='auto'
    i_contamination='auto' 
    i_max_features=1.0
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
            continue
        for p in parameter_values:
            if parameter == 'n_estimators':
                i_n_estimators = p
            elif parameter == 'max_samples':
                i_max_samples = p
            elif parameter == 'max_features':
                i_max_features = p
            elif parameter == 'bootstrap':
                i_bootstrap = p
            elif parameter == 'n_jobs':
                i_n_jobs = p
            elif parameter == 'warm_start':
                i_warm_start = p
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
            accuracy_range_all.append([p, accDiff])
            accuracy_med_all.append([p, np.percentile(accuracy, 50)])
            
            f1Diff = np.percentile(f1, 75) - np.percentile(f1, 25)
            f1_range_all.append([p, f1Diff])
            f1_med_all.append([p, np.percentile(f1, 50)])

    df_acc_r = pd.DataFrame(accuracy_range_all, columns = [parameter, 'Accuracy_Range'])
    df_acc_m = pd.DataFrame(accuracy_med_all, columns = [parameter, 'Accuracy_Median'])
    df_f1_r = pd.DataFrame(f1_range_all, columns = [parameter, 'F1Score_Range'])
    df_f1_m = pd.DataFrame(f1_med_all, columns = [parameter, 'F1Score_Median'])
    
    
    fig = plt.Figure()
    axf = sns.boxplot(x=parameter, y="Accuracy_Range", data=df_acc_r)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_"+parameter+"_Accuracy_Range.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axf = sns.boxplot(x=parameter, y="Accuracy_Median", data=df_acc_m)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_"+parameter+"_Accuracy_Median.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axf = sns.boxplot(x=parameter, y="F1Score_Range", data=df_f1_r)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_"+parameter+"_F1Score_Range.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()
    
    fig = plt.Figure()
    axf = sns.boxplot(x=parameter, y="F1Score_Median", data=df_f1_m)
    # axf.set(xlabel=None)
    plt.savefig("Fig/IF_SK_"+parameter+"_F1Score_Median.pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()