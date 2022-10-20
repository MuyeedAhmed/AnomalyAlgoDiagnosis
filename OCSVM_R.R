#importing libraries
library(tidyverse)
library(rmatio)
library(dbscan)
library(MLmetrics)
library(rio)
#library(raveio)
library(rlang)
library(e1071)
library(pdfCluster)
library(comprehenr)
library(mclust)
datasetFolderDir = 'Dataset/'
folderpath = datasetFolderDir
master_files_mat = Sys.glob(file.path(folderpath,"*.mat"))
master_files_csv = Sys.glob(file.path(folderpath,"*.csv"))
master_files = append(master_files_mat,master_files_csv)
for (i in c(1:length(master_files))){
  master_files[i] = strsplit(master_files[[i]],"//")[[1]][2]
  if (grepl(".csv",master_files[i])){
    master_files[i] = str_split(master_files[i],".csv")[[1]][1]
  }
  else{
    master_files[i] = str_split(master_files[i],".mat")[[1]][1]
  }
}
master_files = sort(master_files)
print(master_files)
parameters = list()
kernel = list('linear', 'polynomial', 'radial', 'sigmoid')
degree = list(3, 4, 5, 6)
gamma = list('scale', 'auto')
coef0 = list(0.0, 0.1, 0.2, 0.3, 0.4)
tolerance = list(0.1, 0.01, 0.001, 0.0001, 0.00001)
nu = list(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
shrinking = list(TRUE, FALSE)
cachesize = list(50, 100, 200, 400)
epsilon = list(0.1, 0.2, 0.01, 0.05)
parameters = append(parameters,list(list("kernel", 'radial', kernel)))
parameters = append(parameters, list(list("degree", 3, degree)))
parameters = append(parameters,list(list("coef0", 0.0, coef0)))
parameters = append(parameters,list(list("tolerance", 0.001, tolerance)))
parameters = append(parameters,list(list("nu", 'IF', nu)))
parameters = append(parameters,list(list("shrinking", TRUE, shrinking)))
parameters = append(parameters,list(list("cachesize", 200, cachesize)))
parameters = append(parameters,list(list("epsilon",0.1,epsilon)))
parameters = append(parameters,list(list("gamma","scale",gamma)))

for (FileNumber in c(1:length(master_files))){
  print(FileNumber)
  ocsvm(master_files[FileNumber], parameters)
}

ocsvm = function(filename, parameters){
  print(filename)
  folderpath = 'Dataset/'
  
  if (file.exists(paste(folderpath,filename,".mat",sep = ""))){
    if (file.info(paste(folderpath,filename,".mat",sep = ""))$size > 200000){
      print("Large")
      return()
    }
    # else{
    #   return()
    # }
    df = read.mat(paste(folderpath,filename,".mat",sep = "")[1])
    gt = df$y
    X = df$X
    if (any(is.na(X))){
      print("Didn't run - NaN")
      return()
    }
  }
  else if (file.exists(paste(folderpath,filename,".csv",sep = ""))){
    if (file.info(paste(folderpath,filename,".csv",sep = ""))$size > 200000){
      print("Large")
      return()
    }
    # else{
    #   return()
    # }
    df = read.csv(paste(folderpath,filename,".csv",sep = "")[1])
    gt = df$target
    X = subset(df, select=-c(target))
    if (any(is.na(X))){
      print("Didn't run - NaN")
      return()
    }
  }
  else{
    print("File Not Found")
    return()
  }
  
  for (p in c(1:length(parameters))) {
    passing_param <- duplicate(parameters, shallow = FALSE)
    if (p == 2 & passing_param[[1]][[2]] != 'polynomial'){
      next
    }
    if (p == 3 & passing_param[[1]][[2]] != 'radial' & passing_param[[1]][[2]] != 'polynomial' & passing_param[[1]][[2]] != 'sigmoid'){
      next
    }
    if (p == 4 & passing_param[[1]][[2]] != 'polynomial' & passing_param[[1]][[2]] != 'sigmoid'){
      next
    }
    cat(paste(parameters[[p]][[1]], ":",sep=""))
    for (pv in c(1:length(parameters[[p]][[3]]))){
      passing_param[[p]][[2]] = parameters[[p]][[3]][[pv]]
      runocsvm(filename,X,gt,passing_param)
      cat(paste(parameters[[p]][[3]][[pv]],",",sep=""))
    }
    cat("\n")
  }
  
}

runocsvm = function(filename,X,gt,params){
  labelfile = paste0(filename,"_",params[[1]][[2]],"_",params[[2]][[2]],"_",params[[9]][[2]],"_",params[[3]][[2]],"_", params[[4]][[2]],"_",params[[5]][[2]],"_",params[[6]][[2]],"_",params[[7]][[2]],"_",params[[8]][[2]])
  if(file.exists(paste('OCSVM_R/',labelfile,".csv",sep = ""))){
    return()
  }
  labels = c()
  f1 = c()
  ari = c()
  labels_df = data.frame()
  if (params[[9]][[2]] == "auto"){
    g = 1 / dim(X)[2]
  }else{
    g = 1 / (dim(X)[2]*mean(var(X)))
  }
  nu_m = params[[5]][[2]]
  if (params[[5]][[2]] == "IF"){
    df_anomaly = read.csv(paste('Stats/SkPercentage.csv',sep="")[1])
    master_files = df_anomaly$Filename
    nu_m = df_anomaly[df_anomaly$Filename == filename, ]$IF
  }
  
  
  tryCatch({
    clustering = svm(X,kernel = params[[1]][[2]], degree = params[[2]][[2]], gamma = g,coef0 = params[[3]][[2]],tolerance = params[[4]][[2]],nu = nu_m,shrinking = params[[6]][[2]],cachesize = params[[7]][[2]],epsilon = params[[8]][[2]])
  }, error = function(err){
    print(err)
    return()
  })
  l = predict(clustering,X)
  list_pred = to_vec(for(j in c(1:length(l))) if(l[j] == TRUE) l[j] = 1 else l[j] = 0)

  labels = c(labels,list(list_pred))
  f1score = F1_Score(gt,list_pred)
  f1 = c(f1,f1score)
  labels_df = rbind(labels_df, data.frame(t(sapply(l,c))))

  write.csv(labels_df,paste('OCSVM_R/',labelfile,".csv",sep=""))

  
}


