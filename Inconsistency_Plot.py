import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
from sklearn.manifold import TSNE

import numpy as np
  
def calculateAccuracy(filename, Algo):
    folderpath = 'Dataset/'
    print(filename)
    
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
    # else:
    #     print("File doesn't exist")
    #     return
    
    # if os.path.exists(folderpath+filename+".mat") == 0:
    #     print("File doesn't exist")
    #     return
    
    # try:
    #     df = loadmat(folderpath+filename+".mat")
    # except NotImplementedError:
    #     df = mat73.loadmat(folderpath+filename+".mat")
    
    # gt=df["y"]
    # y = gt.reshape((len(gt)))
    # X=df['X']
    # if np.isnan(X).any():
    #     print("Didn\'t run -> NaN")
    #     return
    
    X_embedded = TSNE(n_components=2, learning_rate='auto',perplexity = 250,
                       init='random', random_state=(0)).fit_transform(X)
    
    X_embedded = pd.DataFrame(X_embedded)
    
    # drawPlot(filename, Algo, "Matlab", "R", X_embedded, gt)
    # drawPlot(filename, Algo, "Sklearn", "R", X_embedded, gt)
    drawPlot(filename, Algo, "Sklearn", "Matlab", X_embedded, gt)
    
    
def drawPlot(filename, Algo, tool1, tool2, x, y):
    labelsPath1 = "Labels//IF_Sk/Labels_Sk_IF_"+filename+"_100_auto_auto_1.0_False_None_False.csv"
    labelsPath2 = "Labels/IF_Matlab/Labels_Mat_IF_"+filename+"_0.1_100_auto.csv"
    
    # labelsPath1 = "../AnomalyAlgoDiagnosis_Labels/IF_Sk/Labels_Sk_IF_"+filename+"_512_auto_auto_1.0_False_None_False.csv"
    # labelsPath2 = "IF_Matlab/Labels_Mat_IF_"+filename+"_IF_512_auto.csv"
        
    if os.path.exists(labelsPath1) == 0 or os.path.exists(labelsPath2) == 0:
        print("File doesn't exist")
        return
    
    labels1 = pd.read_csv(labelsPath1, header=None).to_numpy()
    labels2 = pd.read_csv(labelsPath2, header=None).to_numpy()
    
    
    mari = 2
    min_i1 = 0
    min_i2 = 0
    for mi in range(10):
        for mj in range(10):
            ari=adjusted_rand_score(labels1[mi], labels2[mj])
            if ari < mari:
                mari = ari
                min_i1 = mi
                min_i2 = mj
    
    l1 = labels1[min_i1]   
    l2 = labels2[min_i2]
    
    # fig = plt.figure()
    # plt.rcParams['figure.figsize'] = [10, 10]
    
    
    # plt.title(filename, fontsize=25)
    
    # indicesToKeep = l1 == 0
    # plt0 = plt.scatter(x.loc[indicesToKeep,0]
    #  ,x.loc[indicesToKeep,1]
    #  ,s = 25, color='grey', rasterized=True)
    
    # indicesToKeep = l1 == 1
    # plt1 = plt.scatter(x.loc[indicesToKeep,0]
    #   ,x.loc[indicesToKeep,1]
    #   ,s = 50, color='red', rasterized=True)
    
    # plt.legend([plt0, plt1],["Normal", "Anomaly"], prop={'size': 25})
    # plt.grid(False)
    # plt.xticks(fontsize = 25)
    # plt.yticks(fontsize = 25)
    
    # plt.savefig('Fig_InterTool_tsne/'+Algo+'_'+filename+'_matlab_default.pdf', dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
    
    # fig = plt.figure()
    # plt.rcParams['figure.figsize'] = [10, 10]
    
    
    # plt.title(filename, fontsize=25)
    
    # indicesToKeep = l2 == 0xs
    # plt0 = plt.scatter(x.loc[indicesToKeep,0]
    #  ,x.loc[indicesToKeep,1]
    #  ,s = 25, color='grey', rasterized=True)
    
    # indicesToKeep = l2 == 1
    # plt1 = plt.scatter(x.loc[indicesToKeep,0]
    #   ,x.loc[indicesToKeep,1]
    #   ,s = 50, color='red', rasterized=True)
    
    # plt.legend([plt0, plt1],["Normal", "Anomaly"], prop={'size': 25})
    # plt.grid(False)
    # plt.xticks(fontsize = 25)
    # plt.yticks(fontsize = 25)
    
    # plt.savefig('Fig_InterTool_tsne/'+Algo+'_'+filename+'_sklearn_default.pdf', dpi=fig.dpi, bbox_inches="tight", pad_inches=0)

    cat = [0] * len(y)
    for i in range(len(y)):
        if l1[i] != l2[i]:
            if l1[i] == 0:
                cat[i] = 2
            if l2[i] == 0:
                cat[i] = 3
        else:
            # cat[i] = l1[i]
            cat[i] = 0
    cat = np.array(cat)
    
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = [10, 10]
    
    
    plt.title(filename, fontsize=15)
    
    indicesToKeep = cat == 0
    plt0 = plt.scatter(x.loc[indicesToKeep,0]
     ,x.loc[indicesToKeep,1]
     ,s = 25, color='grey')
    
    # indicesToKeep = cat == 1
    # plt1 = plt.scatter(x.loc[indicesToKeep,0]
    #  ,x.loc[indicesToKeep,1]
    #  ,s = 50, color='green')
    
    indicesToKeep = cat == 2
    plt2 = plt.scatter(x.loc[indicesToKeep,0]
     ,x.loc[indicesToKeep,1]
     ,s = 50, color='blue')
    
    indicesToKeep = cat == 3
    plt3 = plt.scatter(x.loc[indicesToKeep,0]
     ,x.loc[indicesToKeep,1]
     ,s = 50, color='red')
    
    
    
    plt.legend([plt0, plt2, plt3],["Both Toolkits Predicted Same Output", "Only "+tool2+" Predicted as Anomaly", "Only "+tool1+" Predicted as Anomaly"], prop={'size': 15})
    # plt.legend([plt0, plt1, plt2, plt3],["Both Toolkits Predicted as Normal", "Both Toolkits Predicted as Anomaly", "Only "+tool2+" Predicted as Anomaly", "Only "+tool1+" Predicted as Anomaly"], prop={'size': 15})
    plt.grid(False)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    
    plt.savefig('Fig_InterTool_tsne/'+tool1+'_'+tool2+"_"+Algo+'_'+filename+'_default.pdf', dpi=fig.dpi, bbox_inches="tight", pad_inches=0)

    



    




    labelsPath1 = "Labels/IF_Sk/Labels_Sk_IF_"+filename+"_512_auto_auto_1.0_False_None_False.csv"
    labelsPath2 = "Labels/IF_Matlab/Labels_Mat_IF_"+filename+"_IF_512_auto.csv"
        
    if os.path.exists(labelsPath1) == 0 or os.path.exists(labelsPath2) == 0:
        print("File doesn't exist")
        return
    
    labels1 = pd.read_csv(labelsPath1, header=None).to_numpy()
    labels2 = pd.read_csv(labelsPath2, header=None).to_numpy()
    
    
    mari = 2
    min_i1 = 0
    min_i2 = 0
    for mi in range(10):
        for mj in range(10):
            ari=adjusted_rand_score(labels1[mi], labels2[mj])
            if ari < mari:
                mari = ari
                min_i1 = mi
                min_i2 = mj
    
    l1 = labels1[min_i1]   
    l2 = labels2[min_i2]
    
    # fig = plt.figure()
    # plt.rcParams['figure.figsize'] = [10, 10]
    
    
    # plt.title(filename, fontsize=25)
    
    # indicesToKeep = l1 == 0
    # plt0 = plt.scatter(x.loc[indicesToKeep,0]
    #  ,x.loc[indicesToKeep,1]
    #  ,s = 25, color='grey', rasterized=True)
    
    # indicesToKeep = l1 == 1
    # plt1 = plt.scatter(x.loc[indicesToKeep,0]
    #   ,x.loc[indicesToKeep,1]
    #   ,s = 50, color='red', rasterized=True)
    
    # plt.legend([plt0, plt1],["Normal", "Anomaly"], prop={'size': 25})
    # plt.grid(False)
    # plt.xticks(fontsize = 25)
    # plt.yticks(fontsize = 25)
    
    # plt.savefig('Fig_InterTool_tsne/'+Algo+'_'+filename+'_matlab_Mod.pdf', dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
    
    # fig = plt.figure()
    # plt.rcParams['figure.figsize'] = [10, 10]
    
    
    # plt.title(filename, fontsize=25)
    
    # indicesToKeep = l2 == 0
    # plt0 = plt.scatter(x.loc[indicesToKeep,0]
    #  ,x.loc[indicesToKeep,1]
    #  ,s = 25, color='grey', rasterized=True)
    
    # indicesToKeep = l2 == 1
    # plt1 = plt.scatter(x.loc[indicesToKeep,0]
    #   ,x.loc[indicesToKeep,1]
    #   ,s = 50, color='red', rasterized=True)
    
    # plt.legend([plt0, plt1],["Normal", "Anomaly"], prop={'size': 25})
    # plt.grid(False)
    # plt.xticks(fontsize = 25)
    # plt.yticks(fontsize = 25)
    
    # plt.savefig('Fig_InterTool_tsne/'+Algo+'_'+filename+'_sklearn_Mod.pdf', dpi=fig.dpi, bbox_inches="tight", pad_inches=0)

    cat = [0] * len(y)
    for i in range(len(y)):
        if l1[i] != l2[i]:
            if l1[i] == 0:
                cat[i] = 2
            if l2[i] == 0:
                cat[i] = 3
        else:
            # cat[i] = l1[i]
            cat[i] = 0
    cat = np.array(cat)
    
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = [10, 10]
    
    
    plt.title(filename, fontsize=15)
    
    indicesToKeep = cat == 0
    plt0 = plt.scatter(x.loc[indicesToKeep,0]
     ,x.loc[indicesToKeep,1]
     ,s = 25, color='grey')
    
    # indicesToKeep = cat == 1
    # plt1 = plt.scatter(x.loc[indicesToKeep,0]
    #  ,x.loc[indicesToKeep,1]
    #  ,s = 50, color='green')
    
    indicesToKeep = cat == 2
    plt2 = plt.scatter(x.loc[indicesToKeep,0]
     ,x.loc[indicesToKeep,1]
     ,s = 50, color='blue')
    
    indicesToKeep = cat == 3
    plt3 = plt.scatter(x.loc[indicesToKeep,0]
     ,x.loc[indicesToKeep,1]
     ,s = 50, color='red')
    
    
    
    plt.legend([plt0, plt2, plt3],["Both Toolkits Predicted Same Output", "Only "+tool2+" Predicted as Anomaly", "Only "+tool1+" Predicted as Anomaly"], prop={'size': 15})
    # plt.legend([plt0, plt1, plt2, plt3],["Both Toolkits Predicted as Normal", "Both Toolkits Predicted as Anomaly", "Only "+tool2+" Predicted as Anomaly", "Only "+tool1+" Predicted as Anomaly"], prop={'size': 15})
    plt.grid(False)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    
    plt.savefig('Fig_InterTool_tsne/'+tool1+'_'+tool2+"_"+Algo+'_'+filename+'_Mod.pdf', dpi=fig.dpi, bbox_inches="tight", pad_inches=0)

if __name__ == '__main__':
    folderpath = 'Dataset/'
    master_files = glob.glob(folderpath+"*.mat")
    
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    master_files.sort()

    # Algos = ["EE", "IF", "LF", "OCSVM"]
    # for Algo in Algos:    
    #     for FileNumber in range(len(master_files)):
    #         calculateAccuracy(master_files[FileNumber], Algo)
    # calculateAccuracy("arsenic-male-lung", "IF")
        
    calculateAccuracy("breastw", "IF")
        



