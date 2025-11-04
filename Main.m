clc; clear
rng(0); 

% Retrieve the full path of the currently active MATLAB script
thisFile = matlab.desktop.editor.getActiveFilename;
rootPath = fileparts(thisFile);
addpath(genpath(rootPath));  

% Load the incomplete multi-view dataset "Yale_MissingRatio0.5"
dataPath = fullfile(rootPath, '/Data/Yale_MissingRatio0.5.mat');
load(dataPath); 

% Configure the hyperparameters for the ATSL-IMUFS algorithm
% For the "Yale_MissingRatio0.5" dataset, the following parameter values yield the best performance.
% For other datasets, the optimal parameters can be identified via grid search, 
% where beta and eta are chosen from {10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 10^{2}, 10^{3}}, 
% and r is selected from {2, 3, 4, 5, 6}.
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