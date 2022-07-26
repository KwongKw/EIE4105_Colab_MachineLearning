% Digit recognition evaluation (LDA projection + Gaussian classifier)
%
% Author: M.W. Mak (Oct. 2015)

clear; close all;

dataType = 'noisy';                         % Type of data, can be 'clean' or 'noisy'         
covType = 'diag';                       % Type of covariance matrix, 'full' or 'diagonal'

% Load training and test data into memory
trnfile = sprintf('../data/%s_train_digits.mat',dataType);
tstfile = sprintf('../data/%s_test_digits.mat',dataType);
load(trnfile);                              % Load data structure trainData
load(tstfile);                              % Load data structure testData

% Extract 785 from each class in trainData{}
trainData = extract_data(trainData, 785);

% Extract 100 from each class in testData{} to reduce computation time
testData = extract_data(testData, 100);

% Train an LDA projection matrix
W = train_lda_model(trainData);

% Project data to K-1 dim space
nClasses = length(trainData);
Xtrn = cell(nClasses,1);
Xtst = cell(nClasses,1);
for k=1:nClasses,
    Xtrn{k} = trainData{k}*W;
    Xtst{k} = testData{k}*W;
end

% Train a Gaussian classify using the projected vectors as input
fprintf('Start evaluating %s digit data using Gaussian classifier with %s cov matrix\n',dataType,covType);

% Train a Gaussian density function for each digit and store the PDF parameters
% in the structure array GModel
GModel = train_gauss_model(Xtrn, covType);

% For each test pattern (testData{k}(t,:)), present it to the classifier to find
% the most likely class (label). Then, compare the the true label to see if
% the classification decision is correct. Sum all the correct classification counts
% to estimate the overall accuracy.
totalTest = 0;
nCorrect = 0;
for k = 1:length(Xtst),
    nTest = size(Xtst{k},1);
    fprintf('Evaluating %d samples of digit %d\n', nTest, k-1);
    totalTest = totalTest + nTest;
    label = zeros(1,nTest);
    for t = 1:nTest,
        label(t) = gauss_classification(GModel, Xtst{k}(t,:));
    end
    nCorrect = nCorrect + length(find(label==k-1));
end

acc = 100*nCorrect/totalTest;
fprintf('Accuracy = %.2f\n',acc);