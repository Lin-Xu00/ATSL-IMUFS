function [W_v, X_hat] = ATSL_IMUFS(X_fill_mean, X_fill_zero, Y, E_v, options)
% ATSL_IMUFS performs incomplete multi-view unsupervised feature selection
% via Adaptive Topological Similarity Learning.
%
% This algorithm jointly learns feature selection matrices, adaptive graph
% structures, and completes missing views by iteratively updating model parameters.
%
% Input:
%   X_fill_mean - 1-by-v cell array of multi-view data matrices with missing entries imputed by mean values.
%                 Each X{i} is a [d_i x n] matrix (d_i: feature dimension, n: number of samples).
%   X_fill_zero - 1-by-v cell array of multi-view data matrices with missing entries filled by zeros.
%   Y           - Ground truth labels.
%   E_v         - Cell array indicating the indices of missing samples for each view.
%   options     - Struct containing parameters:
%                   .beta : regularization for consensus similarity matrix S. 
%                   .eta  : regularization for feature selection.
%                   .r    : power exponent for view weight updating.
%
% Output:
%   W_v   - 1-by-v cell array containing learned feature selection matrices for each view.
%   X_hat - Completed multi-view data after adaptive filling.

%% Parameter initialization
X  = X_fill_mean;
X0 = X_fill_zero;

delta = 1;         % Penalty coefficient for augmented Lagrangian
rho   = 1.1;       % Update rate for delta
epson = 1e-3;      % Convergence threshold
max_iter = 50;     % Maximum iterations
K = 15;            % Number of neighbors for graph construction. 
% We set k = 15 in our method, following the common practice in prior work:
% H. Wang, Y. Yang, and B. Liu, "GMC: Graph-based multi-view clustering," IEEE Trans. Knowl. Data Eng., vol. 32, no. 6, pp. 1116–1129, 2019.
beta = options.beta;
eta  = options.eta;
r    = options.r;

v = length(X);                     % Number of views
d = cellfun(@(x) size(x,1), X);    % Number of features in each view
n = size(X{1}, 2);                 % Number of samples
c = length(unique(Y));             % Number of clusters
out_matrix = [];

%% Initialize graph structures and parameters
G_v = cell(1, v);       % Cell array to store similarity graphs for each view
D_W = cell(1, v);       % Cell array to store diagonal weight matrices for each view
L_v = cell(1, v);       % Cell array to store Laplacian matrices for each view
alpha_v = cell(1, v);   % Cell array to store weights of each view

for i = 1:v
    % Construct the initial similarity graph
    G_v{i} = InitializeSIGs(X{i}, K, 0);
    
    % Compute the Laplacian matrix of the similarity graph
    L_v{i} = diag(sum((G_v{i} + G_v{i}')/2)) - (G_v{i} + G_v{i}')/2;
    
    % Initialize the diagonal weight matrix as an identity matrix
    D_W{i} = eye(d(i));
    
    % Initialize the weights of all views to be equal
    alpha_v{i} = 1 / v;
end

% Initialize the consensus graph S as the average of the similarity graphs across all views
S = cellsum(G_v) / v;

% Normalize each row of the consensus graph S
for j = 1:n
    S(j, :) = S(j, :) / (sum(S(j, :)) + eps); 
end

P = S;              % Initialize the Lagrange multiplier P as the consensus graph S
Z = zeros(n, n);    % % Initialize the auxiliary matrix Z as a zero matrix

% Compute the Laplacian matrix of the consensus graph S
L = diag(sum((S + S')/2)) - (S + S')/2;

% Compute the initial cluster indicator matrix
[F, ~, ~] = eig1(L, c, 0);

%% Iterative optimization
Isconverg = zeros(1,v);
iter = 0;

while sum(Isconverg) == 0
    %% Update W_v 
    for i = 1:v
        alpha_r = alpha_v{i}^r;
        temp_W = 2 * alpha_r * X{i} * L_v{i} * X{i}' + eta * D_W{i};
        [W_v{i}, ~, ~] = eig1(temp_W, min(c, d(i)), 0);
        D_W{i} = diag(0.5 ./ sqrt(sum(W_v{i}.^2, 2) + eps));
    end

    %% Update S 
    alpha_Gv = cellfun(@(a, g) a * g, alpha_v, G_v, 'UniformOutput', false);
    A = (1 + beta) * eye(n) - cellsum(alpha_Gv);
    B = 0.5 * L2_distance_1(F', F');
    P = S - 1/delta * (S*A' - Z);

    S = zeros(n);
    for i = 1:n
        q = P(i,:) - 1/delta * (Z(i,:) + P(i,:)*A + B(i,:));
        S(i,:) = SloutionToP19(q, 1);
    end

    Z = Z + delta * (S - P);
    delta = delta * rho;

    L = diag(sum((S+S')/2)) - (S+S')/2;

    %% Update G_v 
    E_Gv = cell(1, v);
    lambda_Gv = cell(1, v);
    S_diff_sq = zeros(n, n);
    for k = 1:n
        S_diff_sq = S_diff_sq + 0.5 * (S(k,:)' - S(k,:)).^2;
    end

    for i = 1:v
        Wi = W_v{i};
        Xi = X{i};
        alpha_r = alpha_v{i}^r;
        L2dis = L2_distance_1(Wi' * Xi, Wi' * Xi);
        E_Gv{i} = alpha_r * (S_diff_sq + L2dis);
        [G_v{i}, lambda_Gv{i}] = topK_square(E_Gv{i}, K);
        L_v{i} = diag(sum((G_v{i}+G_v{i}')/2)) - (G_v{i}+G_v{i}')/2;
    end

    %% Update F
    [F, ~, ~] = eig1(L, c, 0);

    %% Update alpha_v 
    Hv = cell(1, v);
    sum_Hall = 0;
    one_over_1_minus_r = 1 / (1 - r);
    for i = 1:v
        Wi = W_v{i};
        Xi = X{i};
        Gi = G_v{i};
        norm_sq = L2_distance_1(Wi' * Xi, Wi' * Xi);
        sumH = sum(sum(Gi .* (norm_sq + S_diff_sq)));
        Hv{i} = (sumH + eps)^one_over_1_minus_r;
        sum_Hall = sum_Hall + Hv{i};
    end
    for l = 1:v
        alpha_v{l} = Hv{l} / (sum_Hall + eps);
    end

    %% Update X 
    for i = 1:v
        B_m = E_v{i} * L_v{i} * E_v{i}';
        C_m = -X0{i} * L_v{i}' * E_v{i}';
        M_v = C_m * pinv(B_m + eps);
        X{i} = X0{i} + M_v * E_v{i};
    end

    %% Convergence check
    WXG = 0; G_v_fro = 0; W_L21 = 0; 
    for l = 1:v
        WXG = WXG + alpha_v{l}^r * sum(sum(G_v{l} .* (L2_distance_1(W_v{l}' * X{l}, W_v{l}' * X{l}) + S_diff_sq)));
        G_v_fro = G_v_fro + lambda_Gv{l} * norm(G_v{l}, 'fro')^2;
        W_L21 = W_L21 + norm_l21(W_v{l});
    end

    trF = trace(F' * L * F);
    F1 = WXG + trF + beta * norm(S, 'fro')^2 + eta * W_L21 + G_v_fro;
    out_matrix = [out_matrix; F1];

    if iter > 2
        diff = abs(out_matrix(end) - out_matrix(end-1));
    else
        diff = 1;
    end

    if (iter > max_iter) || (diff < epson)
        Isconverg = ones(1,v);
    end
    iter = iter + 1;
end

X_hat = data_norm(X);

end


function [S,xiv] = InitializeSIGs(X, k, issymmetric)
% InitializeSIGs computes the similarity graph for a given dataset.
%
% Input:
% - X: d×n data matrix, where each column represents a sample
% - k: number of the nearest neighbors
% - issymmetric: if issymmetric = 1, enforce symmetry by setting S = (S + S') / 2
%
% Output:
% - S: n×n similarity matrix
% - xiv: auxiliary variable
%
% Reference:
% F. Nie, X. Wang, M. I. Jordan, and H. Huang, "The constrained Laplacian
% rank algorithm for graph-based clustering," AAAI, 2016.

if nargin < 3
    issymmetric = 1;
end
if nargin < 2
    k = 5;
end

[~, n] = size(X);
D = L2_distance_1(X, X);
[~, idx] = sort(D, 2); % sort each row

S = zeros(n);
xiv = 0;
for i = 1:n  
    id = idx(i,2:k+2);
    di = D(i, id);
    fenzi = di(k+1)-di;
    fenmu = k*di(k+1)-sum(di(1:k));
    if fenmu < eps
        fenmu = eps;
    end
    S(i,id) = fenzi / fenmu;
       
    xiv = xiv + 0.5 * k * di(k+1) - 0.5 * sum(di(1:k));
end
xiv = xiv/n;

if issymmetric == 1
    S = (S+S')/2;
end
end


function [x, ft] = SloutionToP19(q0, m)
%  min  1/2 sum_v|| s - qv||^2
%  s.t. s>=0, 1's=1
max_iter = 100;
if nargin < 2
    m = 1;
end
ft=1;
n = length(q0);
p0 = sum(q0,1)/m-mean(sum(q0,1))/m + 1/n;
vmin = min(p0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = lambda_m-p0;
        posidx = v1>0;
        npos = sum(posidx);
        g = npos/n-1;
        if 0 == g
            g = eps;
        end
        f = sum(v1(posidx))/n - lambda_m;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > max_iter
            x = max(-v1,0);
            break;
        end
    end
    x = max(-v1,0);
else
    x = p0;
end
end


function d = L2_distance_1(a, b)
% Computes the squared Euclidean distance matrix between two sets of column vectors.
% 
% Input:
%   a, b: Matrices of size [d x n], where each column represents a d-dimensional data point.
%
% Output:
%   d: Squared Euclidean distance matrix of size [n_a x n_b], where d(i,j) = ||a(:,i) - b(:,j)||^2

% Ensure input has at least two rows to avoid dimension mismatch
if size(a,1) == 1
    a = [a; zeros(1, size(a,2))]; 
    b = [b; zeros(1, size(b,2))]; 
end

% Compute squared norms and cross-term
aa = sum(a.^2);          % Row vector of squared norms of columns in a
bb = sum(b.^2);          % Row vector of squared norms of columns in b
ab = a' * b;             % Inner products between columns of a and b

% Compute pairwise squared Euclidean distances
d = repmat(aa', 1, length(bb)) + repmat(bb, length(aa), 1) - 2 * ab;

% Ensure numerical stability
d = real(d);             % Eliminate possible imaginary parts due to numerical errors
d = max(d, 0);           % Ensure all distances are non-negative
d = d - diag(diag(d));   % Set diagonal to zero explicitly
end

function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)
% eig1 computes the top or bottom $c$ eigenvectors and eigenvalues of matrix $A$.
%
% Input:
%   A     - Input square matrix.
%   c     - Number of eigenvectors to return.
%   isMax - Boolean (1: largest $c$, 0: smallest $c$).
%   isSym - Boolean (1: force $A$ to be symmetric).
%
% Output:
%   eigvec      - The $c$ selected eigenvectors.
%   eigval      - The $c$ selected eigenvalues.
%   eigval_full - All sorted eigenvalues.

    if nargin < 2
        c = size(A,1);
        isMax = 1;
        isSym = 1;
    elseif c > size(A,1)
        c = size(A,1);
    end

    if nargin < 3
        isMax = 1;
        isSym = 1;
    end

    if nargin < 4
        isSym = 1;
    end

    if isSym == 1
        A = max(A,A'); % Enforce symmetry
    end
    
    [v, d] = eig(A); 
    d = diag(d);
    
    if isMax == 0
        [d1, idx] = sort(d); % Smallest first
    else
        [d1, idx] = sort(d,'descend'); % Largest first
    end

    idx1 = idx(1:c);
    eigval = d(idx1); 
    eigvec = v(:,idx1); 

    eigval_full = d(idx); 
end


function X_norm = data_norm(data)

% Perform feature-wise Min-Max normalization on multi-view data
%
% Input:
% - data: 1×v cell array, each cell data{i} contains a d×n data matrix
%         corresponding to the i-th view (d: number of features, n: number of samples)
%
% Output:
% - X_norm: 1×v cell array containing the normalized data for each view
%
% Description:
% This function applies Min-Max normalization independently to each feature
% (row) in all data matrices. Specifically:
% - Features with varying values are scaled to the range [0, 1]
% - Features with all zero values remain unchanged
% - Features with identical nonzero values are set to 1

v = length(data);            % Number of views
d = cell(1, v);              

for i = 1:v
    d{i} = size(data{i}, 1); % Number of features in each view
end

X_norm = cell(1, v);         % Initialize normalized data cell array

for i = 1:v
    for j = 1:d{i}
        max_ = max(data{i}(j, :));    % Maximum of the j-th feature
        min_ = min(data{i}(j, :));    % Minimum of the j-th feature
        extremum = max_ - min_;
        
        if max_ > min_
            % Normal case: apply Min-Max normalization
            X_norm{i}(j, :) = (data{i}(j, :) - min_) / extremum;
        elseif max_ == 0 && min_ == 0
            % If max = min = 0, retain original values
            X_norm{i}(j, :) = data{i}(j, :);
        else
            % If max = min ≠ 0, set all values to 1
            X_norm{i}(j, :) = data{i}(j, :) ./ data{i}(j, :);
        end
        clear min_ max_
    end
end
end

function sum_mat = cellsum(Gv)
% Computes the element-wise sum of all matrices in a cell array.
%
% Input:
%   Gv - A 1-by-V cell array containing matrices of the same size [n x n].
%
% Output:
%   sum_mat - An [n x n] matrix representing the element-wise sum over all cells.

n = size(Gv{1},1);
mat = zeros(n,n);
for v = 1:size(Gv,2)
    mat = mat + Gv{v};
end
sum_mat = mat;
end

function [A, lambda_A] = topK_square(M, K)
% Constructs a sparse affinity matrix by selecting K nearest neighbors based on distances.
%
% Input:
%   M - A pairwise distance matrix of size [n x n].
%   K - The number of neighbors to consider for each sample.
%
% Output:
%   A        - A [n x n] affinity matrix where each row retains weights for K nearest neighbors.
%   lambda_A - The average regularization parameter across all rows.

n = size(M, 1);

% Sort each row to find the nearest neighbors
[~, idxx] = sort(M, 2);

% Initialize output matrix
A = zeros(n);
lambda_A_list = zeros(1, n);

for i = 1:n
    % Exclude self-distance; select K+1 neighbors (including one for margin)
    id = idxx(i, 2:K+2);
    di = M(i, id);

    % Compute weight numerator and denominator
    numerator = di(K+1) - di;
    denominator = K * di(K+1) - sum(di(1:K));

    % Assign non-negative weights
    A(i, id) = max(numerator / (denominator + eps), 0);

    % Store regularization term
    lambda_A_list(i) = 0.5 * (denominator + eps);
end

lambda_A = mean(lambda_A_list);
end

function n21 = norm_l21(x, g_d,g_t, w2,w1)
%   Usage:  n21 = norm_l21(x);
%           n21 = norm_l21(x, g_d,g_t);
%           n21 = norm_l21(x, g_d,g_t, w2,w1);
%
%   Input parameters:
%         x     : Input data 
%         g_d   : group vector 1
%         g_t   : group vector 2
%         w2    : weights for the two norm (default 1)
%         w1    : weights for the one norm (default 1)
%   Output parameters:
%         y     : Norm
%
%   NORM_L21(x, g_d,g_t, w2,w1) returns the norm L21 of x. If x is a
%   matrix the 2 norm will be computed as follow:
%
%       n21 = || x ||_21 = sum_j ( sum_i |x(i,j)|^2 )^(1/2) 
%
%   In this case, all other argument are not necessary.
%
%   'norm_l21(x)' with x a row vector is equivalent to norm(x,1) and
%   'norm_l21(x)' with x a line vector is equivalent to norm(x)
%
%   For fancy group, please provide the groups vectors.
%
%   g_d, g_t are the group vectors. g_d contain the indices of the
%   element to be group and g_t the size of different groups.
%       
%   Example: 
%                x=[x1 x2 x3 x4 x5 x6] 
%                Group 1: [x1 x2 x4 x5] 
%                Group 2: [x3 x6]
%
%   Leads to 
%           
%               => g_d=[1 2 4 5 3 6] and g_t=[4 2]
%               Or this is also possible
%               => g_d=[4 5 3 6 1 2] and g_t=[2 4]   
%
%   This function works also for overlapping groups.
%
%   See also: norm_linf1 norm_tv
%
%   Url: https://lts2.epfl.ch/unlocbox/doc/utils/norm_l21.php

% Copyright (C) 2012-2016 Nathanael Perraudin.
% This file is part of UNLOCBOX version 1.7.3
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

% Author: Nathanael Perraudin
% Date: October 2011
% Testing: test_mixed_sparsity


% Optional input arguments
if nargin<2, g_d = 1:numel(x); end
if nargin<3 
    if numel(x) == size(x,1)*size(x,2) % matrix case
        g_t = size(x,2)*ones(1,size(x,1)); 
    else
        g_t = ones(1,numel(x)); 
    end
end
if nargin<5, w1=ones(size(g_t,2),size(g_t,1)); end
if nargin<4, w2=ones(numel(x),size(g_t,1)); end


% overlapping groups
if size(g_d,1)>1
    n21=0;
    for ii=1:size(g_d,1)
        n21 = n21 + norm_l21(x,g_d(ii,:),g_t(ii,:), ...
            w2(:,ii),w1(:,ii));
    end
else % non overlapping groups
    l=length(g_t);

    % Compute the norm
    if max(g_t)==min(g_t) % group of the same size
        X = transpose(x);
        X = X(g_d);
        X = transpose(reshape(X,numel(x)/l,l));

        W2 = transpose(reshape(w2(g_d),numel(x)/l,l));
        normX2 = sqrt(sum((abs(W2.*X)).^2,2));
        n21 = sum(w1.*normX2);
    else % group of different size

        n21 = 0;
        indice = 0;
        X = x(:);
        X = X(g_d);
        W2 = w2;
        W2 = W2(g_d);
        for i=1:l

            n21 = n21 + w1(i) * norm(W2(indice+1:indice+g_t(i)) ...
                         .*X(indice+1:indice+g_t(i)));
            indice = indice+g_t(i);
        end

    end
    
end
end
