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
    ContaminationFraction.name = "ContaminationFraction";
    ContaminationFraction.default = "LOF";
    ContaminationFraction.values = ["LOF"];
%     ContaminationFraction.default = 0;
%     ContaminationFraction.values = [0, 0.05, 0.1, 0.15, 0.2, 0.25, "LOF", "IF"];
    KernelScale.name = "KernelScale";
    KernelScale.default = "auto";
    KernelScale.values = ["auto"];
%     KernelScale.default = 1;
%     KernelScale.values = [1, "auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5];
    Lambda.name = "Lambda";
    Lambda.default = "auto";
    Lambda.values = ["auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5];
    NumExpansionDimensions.name = "NumExpansionDimensions";
    NumExpansionDimensions.default = "auto";
    NumExpansionDimensions.values = ["auto", 2^12, 2^15, 2^17, 2^19];
    StandardizeData.name = "StandardizeData";
    StandardizeData.default = 0;
    StandardizeData.values = [0, 1];
    BetaTolerance.name = "BetaTolerance";
    BetaTolerance.default = 1e-4;
    BetaTolerance.values = [1e-2, 1e-3, 1e-4, 1e-5];
    GradientTolerance.name = "GradientTolerance";
    GradientTolerance.default = 1e-4;
    GradientTolerance.values = [1e-3, 1e-4, 1e-5, 1e-6];
    IterationLimit.name = "IterationLimit";
    IterationLimit.default = 1000;
    IterationLimit.values = [100, 200, 500, 1000, 2000];
    
    parameters = [ContaminationFraction, KernelScale, Lambda, NumExpansionDimensions, StandardizeData, BetaTolerance, GradientTolerance, IterationLimit];

    filenamesize = size(filenames);
    
    for i = 1:filenamesize(2)
        filename = filenames(i);
        fprintf("%d %s\n",i, filename)
        OCSVM(filename, parameters);
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
function OCSVM(filename, parameters)
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
            runOCSVM(filename, X, y, passing_param);
            
        end
        fprintf("\n")
    end
end
%% Run OCSVM
function runOCSVM(filename_with_extension, X, y, params)
    filename_char = convertStringsToChars(filename_with_extension);
    filename = filename_char(1:end-4);
    labelFile = "OCSVM_Matlab/Labels_Mat_OCSVM_"+filename + "_" + params(1).default + "_" + params(2).default + "_" + params(3).default + "_" + params(4).default + "_" + params(5).default + "_" + params(6).default + "_" + params(7).default + "_" + params(8).default + ".csv";
    if isfile(labelFile)
       return
    end
%     labelFile
    p1 = params(1).default;
    p2 = params(2).default;
    p3 = params(3).default;
    p4 = params(4).default;
    p5 = params(5).default;
    p6 = params(6).default;
    p7 = params(7).default;
    p8 = params(8).default;
    if string(p1) == "LOF"
        percentage_table = readtable("Stats/SkPercentage.csv");
        percentage_table_file = percentage_table(string(percentage_table.Filename)==filename, :);
        p1 = percentage_table_file.LOF;
    elseif string(p1) == "IF"
        percentage_table = readtable("Stats/SkPercentage.csv");
        percentage_table_file = percentage_table(string(percentage_table.Filename)==filename, :);
        p1 = percentage_table_file.IF;        
    end
    sX = string([1:size(X, 2)]);
    
    outliersSet = [];
    try
        for z = 1:10
            [Mdl, tf] = ocsvm(X, PredictorNames=sX,ContaminationFraction=p1, KernelScale=p2, Lambda=p3, NumExpansionDimensions=p4, ...
                StandardizeData=p5, BetaTolerance=p6, ...
                GradientTolerance=p7, IterationLimit=p8);
            outliersSet = [outliersSet;tf'];
        end
%         writefilename = sprintf('../AnomalyAlgoDiagnosis_Labels_Matlab/%s', labelFile);
        csvwrite(labelFile,outliersSet); 
    catch
        fprintf("-Failed")
    end
end