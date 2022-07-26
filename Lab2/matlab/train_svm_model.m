% Train an SVM classifier
% Input:
%   trainData        - 10x1 cell array containing training data. trainData{1}
%                      contains a number of '0', each is stored in a row.
%   kerType          - Type of SVM kernel, can be 'rbf', 'poly', or 'linear'
%   kerPara          - Kernel parameter, sigma for 'rbf', degree for 'poly'
%   C                - Panelty factor for training SVM
% Output:
%   SVMmodel         - Cell array containing SVM structures
%                      SVMmodel(1) contains the SVM structure for digit '0'
% Example:
%   load '../data/clean_train_digits.mat'; trainData = extract_data(trainData, 100);
%   SVMmodel = train_svm_model(trainData, 'poly', 2, 100);
%
% Author: M.W. Mak (Oct. 2015)

function SVMmodel = train_svm_model(trainData, kerType, kerPara, C)
addpath '../bioinfo/biolearning';
addpath '../bioinfo/bioinfo';
nClasses = length(trainData);
SVMmodel = cell(nClasses,1);
X = cell2mat(trainData);                % Convert training data from cell to matrix
startIdx = zeros(nClasses+1,1);
startIdx(1) = 1;
for k = 2:nClasses+1,
    startIdx(k) = startIdx(k-1)+size(trainData{k-1},1);
end
for k = 1:nClasses,
    fprintf('Training digit %d\n', k-1);
    y = -1*ones(size(X,1),1);
    y(startIdx(k):startIdx(k+1)-1) = 1;
    switch (kerType)
        case 'rbf'
            net = svmtrain(X,y,'Kernel_Function','rbf','RBF_Sigma',kerPara,'BoxConstraint',C,...
                           'Method','SMO','Autoscale',false);
        case 'poly'
            net = svmtrain(X,y,'Kernel_Function','polynomial','Polyorder',kerPara,'BoxConstraint',C,...
                           'Method','SMO','Autoscale',false);
        case 'linear'
            net = svmtrain(X,y,'Kernel_Function','linear','BoxConstraint',C,...
                           'Method','SMO','Autoscale',false);
            net.normalw = (net.SupportVectors'*(abs(net.Alpha).*y(net.SupportVectorIndices)))';
        case 'custom'
            error('Custom kernel not support');
    end
    % Make the struct net compatible with that created by Auton's svm
    % trainer
    net.svind = net.SupportVectorIndices;
    net.alpha = zeros(length(y),1);
    net.alpha(net.svind) = abs(net.Alpha);
    net.sv = net.SupportVectors;
    net.svcoeff = abs(net.Alpha);
    net.bias = net.Bias;
    SVMmodel{k} = net;
end    


