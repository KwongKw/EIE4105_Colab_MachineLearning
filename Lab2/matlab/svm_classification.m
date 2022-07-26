% Predict the digit of a test image based on an SVM classifier
% Input:
%   SVMmodel         - A cell array containing the SVM structure
%                      See train_svm_model.m for details
%   x                - A test image in D-dim row vector
% Output:
%   label            - Predicted class label of x, '0' - '9'
%   scores           - K-dim array containing the SVM score for each class 
%  
% Example:
%   load '../data/clean_test_digits.mat';
%   [label, scores] = svm_classification(SVMmodel, testData{1}(1,:));
%
% Author: M.W. Mak (Sept. 2015)

function [label, scores] = svm_classification(SVMmodel, x)
nClasses = length(SVMmodel);
scores = zeros(1,nClasses);     % SVM scores, f(x)

% Compute SVM output for each class given an input vector x 
for k = 1:nClasses,
    [~,scores(k)] = NEWsvmclassify(SVMmodel{k}, x);
end

% Find the predicted class (implement the argmax operator)
[~, label] = max(scores);
label = label - 1;              % Adjust for offset (index in Matlab starts from 1, but our
                                % digits start from 0)