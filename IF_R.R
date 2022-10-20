#importing libraries
library(tidyverse)
library(rmatio)
library(dbscan)
library(MLmetrics)
library(rio)
library(raveio)
library(rlang)
library(e1071)
library(pdfCluster)
library(comprehenr)
library(mclust)
library(isotree)

datasetFolderDir = 'Dataset/'
folderpath = datasetFolderDir
master_files_mat = Sys.glob(file.path(folderpath,"*.mat"))
master_files_csv = Sys.glob(file.path(folderpath,"*.csv"))
master_files = append(master_files_mat,master_files_csv)
for (i in c(1:length(master_files))){
  master_files[i] = strsplit(master_files[[i]],"//")[[1]][2]
  if (endsWith(master_files[i],".mat")){
    master_files[i] = str_split(master_files[i],".mat")[[1]][1]
  }
  else{
    master_files[i] = str_split(master_files[i],".csv")[[1]][1]
  }
  
}
master_files = sort(master_files)
parameters = list()
ntrees = list(2, 4, 8, 16, 32, 64, 100, 128, 256, 512)
standardize_data = list(TRUE,FALSE)
sample_size = list('auto',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,NULL)
ncols_per_tree = list('def',0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)

parameters = append(parameters,list(list("ntrees",512,ntrees)))
parameters = append(parameters,list(list("standardize_data",TRUE,standardize_data)))
parameters = append(parameters,list(list("sample_size",'auto',sample_size)))
parameters = append(parameters,list(list("ncols_per_tree",'def',ncols_per_tree)))



for (FileNumber in c(1:length(master_files))){
  print(FileNumber)
  isolationforest(master_files[FileNumber], parameters)
}
datasetFolderDir = 'Dataset/'
isolationforest = function(filename, parameters){
  print(filename)
  folderpath = 'Dataset/'
  if (file.exists(paste(folderpath,filename,".mat",sep = ""))){
    if (file.info(paste(folderpath,filename,".mat",sep = ""))$size > 200000){
      print("Too Large")
      return()
    }
    df = read_mat(paste(folderpath,filename,".mat",sep = "")[1])
    gt = df$y
    X = df$X
    if (any(is.na(X))){
      print("Didn't run - NaN")
      return()
    }
  }
  else if (file.exists(paste(folderpath,filename,".csv",sep = ""))){
    if (file.info(paste(folderpath,filename,".csv",sep = ""))$size > 200000){
      print("Too Large")
      return()
    }
    df = read.csv(paste(folderpath,filename,".csv",sep = "")[1])
    gt = df$target
    X = subset(df, select=-c(target))
    if (any(is.na(X))){
      print("Didn't run - NaN")
      return()
    }
  }
 
 
  for (p in c(1:length(parameters))) {
    passing_param <- duplicate(parameters, shallow = FALSE)
    cat(paste(parameters[[p]][[1]], ":",sep=""))
    for (pv in c(1:length(parameters[[p]][[3]]))){
      passing_param[[p]][[2]] = parameters[[p]][[3]][[pv]]
      runif(filename,X,gt,passing_param)
      cat(paste(parameters[[p]][[3]][[pv]],",",sep=""))
    }
    cat("\n")
  }
  
}

runif = function(filename,X,gt,params){
  labelfile = paste0(filename,"_",params[[1]][[2]],"_",params[[2]][[2]],"_",params[[3]][[2]],"_",params[[4]][[2]])
  if (file.exists(paste('IF_R/',labelfile,".csv",sep=""))){
    return()
  }
  labels = c()
  f1 = c()
  ari = c()
  if (params[[3]][[2]] == "auto"){
      p3 = min(nrow(X),10000L)
  }else{
      p3 = params[[3]][[2]]
  }
  if (params[[4]][[2]] == "def"){
      p4 = ncol(X)
  }else{
      p4 = params[[4]][[2]]
  }
  
  
  labels_df = data.frame()
  for (i in 1:c(10)){
    tryCatch({
      clustering = isolation.forest(X,ntrees = params[[1]][[2]],standardize_data = params[[2]][[2]], sample_size = p3, ncols_per_tree = p4, seed = sample(c(1:100),1))
    }, error = function(err){
      print(err)
      return()
    })
    l = predict(clustering,X)
    list_pred = to_vec(for(j in c(1:length(l))) if(l[j] > 0.5) l[j] = 1 else l[j] = 0)
    #    for (j in c(1:length(l))){
    #      if(l[j] == TRUE){
    #        l[j] = 1
    #      }
    #      else{
    #        l[j]  = 0
    #      }
    #    }
    labels = c(labels,list(list_pred))
    f1score = F1_Score(gt,list_pred)
    f1 = c(f1,f1score)
    labels_df = rbind(labels_df, data.frame(t(sapply(list_pred,c))))
  }
  write.csv(labels_df,paste('IF_R/',labelfile,".csv",sep=""))
  #ari
  for (i in c(1:length(labels))){
    for (j in c(i+1:length(labels))){
      ari_score = adjustedRandIndex(labels[i],labels[j])
      ari = c(ari,ari_score)
    }
  }
  
}


