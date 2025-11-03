clc; clear
rng(0); 

% Retrieve the full path of the currently active MATLAB script
thisFile = matlab.desktop.editor.getActiveFilename;
rootPath = fileparts(thisFile);
addpath(genpath(rootPath));  

% Load the incomplete multi-view dataset
dataPath = fullfile(rootPath, '/Data/Yale_MissingRatio0.5.mat');
load(dataPath); 

% Configure hyperparameters for the ATSL-IMUFS algorithm
options = []; 
options.beta = 0.001;  
options.eta = 0.001; 
options.r = 2;  

% Execute the proposed ATSL-IMUFS method
[W, X] = ATSL_IMUFS(X_fill_mean, X_fill_zero, Y, E_v, options);

% Evaluate the clustering performance using k-means 
kmeans_iter = 30;              
fea_rate = 0.5;               
[ACC_mean, NMI_mean] = ClusteringPerformance(X , Y, W, kmeans_iter, fea_rate);

% Display the average clustering accuracy and normalized mutual information
fprintf('ACC(mean) = %.2f, NMI(mean) = %.2f\n', ACC_mean, NMI_mean);