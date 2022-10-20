clear
clc
%% Init
main_function()
%% Main Function
function main_function()
    T = readtable('anomaly_python.csv', 'ReadVariableNames', true);
    dataset = string(T.Dataset);
    anomaly = T.Anomaly_Python;
    middle_list = [];
    for i = 1:length(dataset)
        mid = LOF_bin(dataset(i),anomaly(i));
        mid
        middle_list = [middle_list;mid];
    end
    anomaly_matlab_df = table(dataset(:), middle_list(:), 'VariableNames', {'Dataset','Anomaly_Matlab'})
    writetable(anomaly_matlab_df,'anomaly_matlab_threshold.csv');
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
function mid  = LOF_bin(filename,anomaly)
    readfilename = sprintf('Dataset/%s',filename);
    fprintf(readfilename)
    the_size=dir(readfilename).bytes;
    mid = 0;
%     if the_size > 1000000
%         disp("\n")
%         disp("Large File")
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
    k = 20;
    [suspicious_index lof] = LOF(X, k);
    start = min(lof);
    last = 20;
    iteration = 1;
    while true 
        iteration = iteration+1;
        if iteration>10000
            %fprintf(sprintf("Went above 1000%s",filename));
            return
        end
        middle = (start+last)/2;
        pred_new = lof;
        for  j = 1:length(lof)
            if lof(j)> middle
                pred_new(j) = 1;
            else
                pred_new(j) = 0;
      
            end
        end
        anomaly_bin = (sum(pred_new == 1)/length(pred_new))*100;
        if (abs(anomaly_bin - anomaly) < 0.000001)
            mid = middle;
        elseif anomaly_bin > anomaly
            start = middle;
        else
            last = middle;
        end
    end
    end

