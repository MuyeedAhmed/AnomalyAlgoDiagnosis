#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:23:17 2022

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



lof_merged = pd.read_csv("Stats/LOF/lof_merged.csv")


R_Performance = lof_merged[['F1Score_R_default', 'F1Score_R_Modified']]
R_Performance = R_Performance.rename(columns={"F1Score_R_default":"Default", "F1Score_R_Modified":"Modified"})

fig, ax = plt.subplots()
axmean = sns.boxplot(x="value", y="variable", data=pd.melt(R_Performance)).set(xlabel = 'F1 Score', ylabel=None)
plt.title("Local Outlier Factor - R")
plt.savefig("Fig/BoxPlot/LOF_R_Performance.pdf", bbox_inches="tight", pad_inches=0)
plt.clf()


Mat_Performance = lof_merged[['F1Score_Matlab_Default', 'F1Score_Matlab_Modified']]
Mat_Performance = Mat_Performance.rename(columns={"F1Score_Matlab_Default":"Default", "F1Score_Matlab_Modified":"Modified"})

fig, ax = plt.subplots()
axmean = sns.boxplot(x="value", y="variable", data=pd.melt(Mat_Performance)).set(xlabel = 'F1 Score', ylabel=None)
plt.title("Local Outlier Factor - Matlab")
plt.savefig("Fig/BoxPlot/LOF_Mat_Performance.pdf", bbox_inches="tight", pad_inches=0)
plt.clf()





# plt.title("Local Outlier Factor - Scikit-learn VS Matlab")



SkVsR = lof_merged[['SKlearn_R_Default_ARI', 'SKlearn_R_Modified_ARI']]
SkVsR = SkVsR.rename(columns={"SKlearn_R_Default_ARI":"Default", "SKlearn_R_Modified_ARI":"Modified"})

fig, ax = plt.subplots()
axmean = sns.boxplot(x="variable", y="value", data=pd.melt(SkVsR)).set(xlabel = 'F1 Score', ylabel=None)
plt.title("Local Outlier Factor - Scikit-learn VS R")
plt.savefig("Fig/BoxPlot/LOF_Sk_R_Performance.pdf", bbox_inches="tight", pad_inches=0)
plt.clf()


SkVsMat = lof_merged[['SKlearn_Matlab_Default_ARI', 'SKlearn_Matlab_Modified_ARI']]
SkVsMat = SkVsMat.rename(columns={"SKlearn_Matlab_Default_ARI":"Default", "SKlearn_Matlab_Modified_ARI":"Modified"})

fig, ax = plt.subplots()
axmean = sns.boxplot(x="variable", y="value", data=pd.melt(SkVsMat)).set(xlabel = 'F1 Score', ylabel=None)
plt.title("Local Outlier Factor - Scikit-learn VS Matlab")
plt.savefig("Fig/BoxPlot/LOF_Sk_Mat_Performance.pdf", bbox_inches="tight", pad_inches=0)
plt.clf()


RVsMat = lof_merged[['Matlab_R_Default_ARI', 'Matlab_R_Modified_ARI']]
RVsMat = RVsMat.rename(columns={"Matlab_R_Default_ARI":"Default", "Matlab_R_Modified_ARI":"Modified"})

fig, ax = plt.subplots()
axmean = sns.boxplot(x="variable", y="value", data=pd.melt(RVsMat)).set(xlabel = 'F1 Score', ylabel=None)
plt.title("Local Outlier Factor - R VS Matlab")
plt.savefig("Fig/BoxPlot/LOF_R_Mat_Performance.pdf", bbox_inches="tight", pad_inches=0)
plt.clf()

