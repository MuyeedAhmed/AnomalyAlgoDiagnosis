library(tidyverse)
library(rmatio)
library(dbscan)
library(rio)
library(raveio)

localoutlierfactor = function(filename,threshold){
  folderpath = 'C:/Users/parth/School/Clustering/Dataset_Combined/'
  if (endsWith(filename,'.mat')){
    df = read_mat(paste(folderpath,filename,sep = "")[1])
    target = df$y
    X = df$X
  }
  else if (endsWith(filename,".csv")){
    df = read.csv(paste(folderpath,filename,sep = "")[1])
    target = df$target
    X = subset(df, select=-c(target))
  }
  name = strsplit(filename,"\\.")[[1]][1]
  #default
  pred_def = lof(X)
  for (j in c(1:length(pred_def))){
    if(pred_def[j]> 1){
      pred_def[j] = 1
    }
    else{
      pred_def[j] = 0
    }
  }
  write.csv(pred_def, paste("C:/Users/parth/School/Clustering/LOF_R_Default_Labels/",name,".csv",sep = ""),row.names = FALSE)
  f1score_def = F1_Score(target,pred_def)
  #modified
  if (threshold == 0){
    f1score = 0
    return (list(f1score,f1score_def))
  }
  pred = lof(X, minPts=20)
  for (j in c(1:length(pred))){
    if(pred[j]> threshold){
      pred[j] = 1
    }
    else{
      pred[j] = 0
    }
  }
  
  write.csv(pred, paste("C:/Users/parth/School/Clustering/LOF_R_Modified_Labels/",name,".csv",sep = ""),row.names = FALSE)
  f1score = F1_Score(target,pred)
 
  return (list(f1score,f1score_def))
}
folderpath = 'C:/Users/parth/School/Clustering/Dataset_Combined/'
df_anomaly = read.csv(paste('C:/Users/parth/School/Clustering/anomaly_r_threshold.csv',sep="")[1])
master_files = df_anomaly$Dataset
threshold = df_anomaly$Threshold
f1score_list = list()
f1score_def_list = list()
for (i in c(1:length(master_files))){
  score_list=localoutlierfactor(master_files[i],threshold[i])
  f1score_list = append(f1score_list,score_list[1])
  f1score_def_list = append(f1score_def_list,score_list[2])
  print(master_files[i])
}
df_anomaly$F1Score_R_Modified = f1score_list
df_anomaly$F1Score_R_default = f1score_def_list
df_anomaly <- apply(df_anomaly,2,as.character)
write.csv(df_anomaly, paste("C:/Users/parth/School/Clustering/anomaly_r.csv",sep = ""),row.names = FALSE)