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
    ContaminationFraction.default = 0;
    ContaminationFraction.values = [0, 0.05, 0.1, 0.15, 0.2, 0.25, "LOF", "IF"];
    NumLearners.name = "NumLearners";
    NumLearners.default = 100;
    NumLearners.values = [1, 2, 4, 8, 16, 32, 64, 100, 128, 256, 512];
    NumObservationsPerLearner.name = "NumObservationsPerLearner";
    NumObservationsPerLearner.default = "auto";
    NumObservationsPerLearner.values = ["auto", 0.05, 0.1, 0.2, 0.5, 1];
    
    parameters = [ContaminationFraction, NumLearners, NumObservationsPerLearner];

    filenamesize = size(filenames);
    
    for i = 1:filenamesize(2)
        filename = filenames(i);
        fprintf("%d %s\n",i, filename)
        IF(filename, parameters);
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

%% IF
function IF(filename, parameters)
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
            runIF(filename, X, y, passing_param);
            
        end
        fprintf("\n")
    end
end
%% Run IF
function runIF(filename, X, y, params)
    labelFile = "IF_Matlab/Labels_Mat_IF_"+filename + "_" + params(1).default + "_" + params(2).default + "_" + params(3).default + ".csv";
    if isfile(labelFile)
       return
    end

    p1 = params(1).default;
    p2 = params(2).default;
    p3 = params(3).default;
    if string(p1) == "LOF"
        percentage_table = readtable("Stats/SkPercentage.csv");
        filename_char = convertStringsToChars(filename);
        percentage_table_file = percentage_table(string(percentage_table.Filename)==filename_char(1:end-4), :);
        p1 = percentage_table_file.LOF;
    elseif string(p1) == "IF"
        percentage_table = readtable("Stats/SkPercentage.csv");
        filename_char = convertStringsToChars(filename);
        percentage_table_file = percentage_table(string(percentage_table.Filename)==filename_char(1:end-4), :);
        p1 = percentage_table_file.IF;        
    end
    if string(p3) == "auto"
        p3 = min(size(X,1), 256);
    else
        p3 = floor(p3*size(X,1));
    end
    outliersSet = [];
    try
        for z = 1:10
            [forest, tf, score] = iforest(X, ContaminationFraction=p1, NumLearners=p2, NumObservationsPerLearner=p3);
            outliersSet = [outliersSet;tf'];

        end
        csvwrite(labelFile,outliersSet); 
    catch
        fprintf("-Failed")
    end
end