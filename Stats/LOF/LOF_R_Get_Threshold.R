library(tidyverse)
library(rmatio)
library(dbscan)
library(rio)
library(raveio)
library(MLmetrics)

localoutlierfactor = function(filename,anomaly_perc){
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
  pred = lof(X, minPts=20)
  name = strsplit(filename,"\\.")[[1]][1]
  #write.csv(pred, paste("C:/Users/parth/School/Clustering/LOF_R_Labels/",name,".csv",sep = ""))
  #binary search
  start = min(pred)
  end = 20
  iteration = 1
  while (TRUE) {
    iteration = iteration+1
    if(iteration>1000){
      print("Went above 1000")
      print(filename)
      return(0)
      break
    }
    middle = (start+end)/2
    pred_new = pred
    for (j in c(1:length(pred))){
      if(pred[j]> middle){
        pred_new[j] = 1
      }
      else{
        pred_new[j] = 0
      }
    }
    anomaly = (sum(pred_new == 1)/length(pred_new))*100
    if (abs(anomaly - anomaly_perc) < 0.000001){
      return(middle)
      break
    }
    else if (anomaly > anomaly_perc){
      start = middle
    }
    else{
      end = middle
    }
  }
}
folderpath = 'C:/Users/parth/School/Clustering/Dataset_Combined/'
df_anomaly = read.csv(paste('C:/Users/parth/School/Clustering/anomaly_python.csv',sep="")[1])
master_files = df_anomaly$Dataset
anomaly_perc = df_anomaly$Anomaly_Python
df_threshold = data.frame(matrix(ncol=2,nrow=0))
colnames(df_threshold) = c('Dataset','Threshold')
for (i in c(1:length(master_files))){
  #print(master_files[i])
  threshold= localoutlierfactor(master_files[i],anomaly_perc[i])
  df_threshold[nrow(df_threshold)+1,] = c(master_files[i],threshold)
}
write.csv(df_threshold, paste("C:/Users/parth/School/Clustering/anomaly_r_threshold.csv",sep = ""),row.names = FALSE)