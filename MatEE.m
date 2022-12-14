clear
clc
%% Init
main_function()
%% Main Function
function main_function()
    files_csv = dir ('Dataset/*.csv');
    filecount_csv = height(struct2table(files_csv));
    filenames = [];
    for i= 1:filecount_csv
        filenames = [filenames, string(files_csv(i).name)];
    end

    files_mat = dir ('Dataset/*.mat');
    filecount_mat = height(struct2table(files_mat));

    for i= 1:filecount_mat
        filenames = [filenames, string(files_mat(i).name)];
    end

    parameters = [];

    Method.name = "Method";
    Method.default = "fmcd";
    Method.values = ["fmcd", "ogk", "olivehawkins"];
    OutlierFraction.name = "OutlierFraction";% fcmd, olivehawkins
    OutlierFraction.default = 0.5;
    OutlierFraction.values = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5];
    NumTrials.name = "NumTrials"; % 500 if fcmd, 2 if olivehawkins
    NumTrials.default = 500;
    NumTrials.values = [2, 5, 10, 25, 50, 100, 250, 500, 750, 1000];
    BiasCorrection.name = "BiasCorrection"; %fcmd
    BiasCorrection.default = 1;
    BiasCorrection.values = [1, 0];
    NumOGKIterations.name = "NumOGKIterations"; % ogk
    NumOGKIterations.default = 2;
    NumOGKIterations.values = [1, 2, 3];
    UnivariateEstimator.name = "UnivariateEstimator"; % ogk
    UnivariateEstimator.default = "tauscale";
    UnivariateEstimator.values = ["tauscale", "qn"];
    ReweightingMethod.name = "ReweightingMethod";%olivehawkins
    ReweightingMethod.default = "rfch";
    ReweightingMethod.values = ["rfch", "rmvn"];
    NumConcentrationSteps.name = "NumConcentrationSteps";%olivehawkins
    NumConcentrationSteps.default = 10;
    NumConcentrationSteps.values = [2, 5, 10, 15, 20];
    StartMethod.name = "StartMethod";%olivehawkins
    StartMethod.default = "classical";
    StartMethod.values = ["classical", "medianball", "elemental"];
    
    parameters = [Method, OutlierFraction, NumTrials, BiasCorrection, NumOGKIterations, UnivariateEstimator, ReweightingMethod, NumConcentrationSteps, StartMethod];

    filenamesize = size(filenames);
    
    for i = 1:filenamesize(2)
        filename = filenames(i);
        fprintf("%d %s\n",i, filename)
        EE(filename, parameters);
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

%% OCSVM
function EE(filename, parameters)
    readfilename = sprintf('Dataset/%s', filename);
    the_size=dir(readfilename).bytes;
    if the_size > 200000
        return
    end
    
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
    for p = 1:length(parameters)
        passing_param = parameters;
        fprintf("  %s: ", passing_param(p).name)
        if strcmp(passing_param(1).default, "fmcd") == 1 && p ~= 1
            if p == 5 || p == 6 || p == 7 || p == 8 || p == 9
                fprintf("\n")
                continue
            end
        end
        if strcmp(passing_param(1).default, "ogk") == 1 && p ~= 1
            if p == 2 || p == 3 || p == 4 || p==7 || p == 8 || p == 9
                fprintf("\n")
                continue
            end
        end
        if strcmp(passing_param(1).default, "olivehawkins") == 1 && p ~= 1
            if p == 4 || p == 5 || p == 6
                fprintf("\n")
                continue
            end
        end
        for pv=1:length(parameters(p).values)
            if isa(parameters(p).values,'double')
                passing_param(p).default = parameters(p).values(pv);
                fprintf("%d, ", parameters(p).values(pv))
            elseif isa(parameters(p).values,'string')
                if isempty(str2num(parameters(p).values(pv)))
                    passing_param(p).default = parameters(p).values(pv);
                else
                    passing_param(p).default = str2num(parameters(p).values(pv));
                end
                fprintf("%s, ", parameters(p).values(pv))
            end
            runEE(filename, X, y, passing_param);
            
        end
        fprintf("\n")
    end
end

%% Run EE
function runEE(filename_with_extension, X, y, params)
    filename_char = convertStringsToChars(filename_with_extension);
    filename = filename_char(1:end-4);
    labelFile = "EE_Matlab/Labels_Mat_EE_"+filename + "_" + params(1).default + "_" + params(2).default + "_" + params(3).default + "_" + params(4).default + "_" + params(5).default + "_" + params(6).default + "_" + params(7).default + "_" + params(8).default + "_" + params(9).default + ".csv";
    if isfile(labelFile)
       return
    end
    p1 = params(1).default;
    p2 = params(2).default;
    p3 = params(3).default;
    p4 = params(4).default;
    p5 = params(5).default;
    p6 = params(6).default;
    p7 = params(7).default;
    p8 = params(8).default;
    p9 = params(9).default;
    
    outliersSet = [];
    try
        for z = 1:10
            if strcmp(p1, "fmcd") == 1
                [sig,mu,mah,outliers] = robustcov(X, Method=p1, OutlierFraction=p2, NumTrials=p3, BiasCorrection=p4);
        
            elseif strcmp(p1, "ogk") == 1
                [sig,mu,mah,outliers] = robustcov(X, Method=p1, NumOGKIterations=p5, UnivariateEstimator=p6);
            elseif strcmp(p1, "olivehawkins") == 1
                [sig,mu,mah,outliers] = robustcov(X, Method=p1, OutlierFraction=p2, ...
                    ReweightingMethod=p7, NumConcentrationSteps=p8, StartMethod=p9);
            end
            outliersSet = [outliersSet;outliers'];
        end
        csvwrite(labelFile,outliersSet); 
    catch
        fprintf("-Failed")
    end
end