% Train an LDA model by estimating the Dx(K-1) projection matrix
% Input:
%   trainData        - 10x1 cell array containing training data. trainData{1}
%                      contains a number of '0', each is stored in a row.
% Output:
%   W                - D x (K-1) LDA projection matrix
%   lambda           - Eigenvalues corresponding to the K-1 eigenvectors
% Example:
%   load '../data/noisy_train_digits.mat'; 
%   [W, lambda] = train_lda_model(trainData);
%
% Author: M.W. Mak (Oct. 2015)

function [W,lambda] = train_lda_model(trainData)
X = cell2mat(trainData);
D = size(X,2);                              % Dim of input vectors
Sw = zeros(D,D);                            % Within-class cov matrix
Sb = zeros(D,D);                            % Between-class cov matrix
mu = mean(X,1);                             % Glogal mean of input vectors
nClasses = length(trainData);
for k = 1:nClasses
    X_k = trainData{k};
    N_k = size(X_k,1);
    Sw = Sw + N_k*cov(X_k,1); 
    mu_k = mean(X_k,1);                     % Class mean
    Sb = Sb + N_k*(mu_k-mu)'*(mu_k-mu);     % Note that mu_k and mu are row vectors           
end

% Find the K-1 largest eigenvectors and eigenvalues of AU=Lambda B, i.e.,
% find the eigenvectors of inv(Sw)*Sb
[W,Lambda] = eigs(Sb,Sw,nClasses-1);
lambda = diag(Lambda);
end    


