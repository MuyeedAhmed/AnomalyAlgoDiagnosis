clear
clc
%% Init
main_function()
%% Main Function
function main_function()
T = readtable('anomaly_matlab_threshold.csv', 'ReadVariableNames', true);
    dataset = string(T.Dataset);
    threshold = T.Anomaly_Matlab;
    for i = 1:length(dataset)
        fprintf(dataset(i))
        LOF_labels(dataset(i),threshold(i));
    end
end
 %% Read File
function [X, y] = csvfileread(readfilename)
    T = readtable(readfilename, 'ReadVariableNames', true);
    ColIndex = find(strcmp(T.Properties.VariableNames, 'target'), 1);
    A = table2array(T);
    A(any(isnan(A), 2), :) = [];
    target=A(:, ColIndex);
    A(:, ColIndex)=[];
    X = A;
    y = target;
end
function [X, y] = matfileread(readfilename)
    A = load(readfilename);
    X = A.X;
    y = A.y;
end

%% LOF
function LOF_labels(filename,threshold)
    readfilename = sprintf('Dataset/%s', filename);
%     the_size=dir(readfilename).bytes;
%     if the_size > 1000000
%         return
%     end
    
    if contains(filename, '.csv') == true
        [X, y] = csvfileread(readfilename);
    end
    if contains(filename, '.mat') == true
        [X, y] = matfileread(readfilename);
    end
    if size(X, 1) < size(X,2)*2
        disp("Dimention Error")
        return
    end
    %default
    outliersSet = [];
    [suspicious_index lof] = LOF(X, 1);
    outlier=lof>=2;
    outliersSet = [outliersSet;outlier];

    writefilename = 'LOF_Default_MatLab_Labels/' + extractBetween(filename, 1, strlength(filename)-4) + '.csv';
    csvwrite(writefilename,outliersSet)
    %modified
    outliersSet1 = [];
    [suspicious_index lof] = LOF(X, 20);
    if threshold == 0
        return
    end
    outlier1=lof>=threshold;
    outliersSet1 = [outliersSet1;outlier1];
    writefilename = 'LOF_Modified_MatLab_Labels/' + extractBetween(filename, 1, strlength(filename)-4) + '.csv';
    csvwrite(writefilename,outliersSet1)

end