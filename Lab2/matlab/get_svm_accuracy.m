% Digit recognition evaluation using SVM classifier
%
% Author: M.W. Mak (Oct. 2015)

clear;
close all;

kerType = 'poly';                            % SVM kernel type, can be 'rbf','poly','linear'
kerPara = 2;                                % Kernel parameter, sigma for 'rbf', polynomial degree for 'poly'                                 
C = 100;                                     % Penalty factor for training SVM
dataType = 'clean';                         % Type of data, can be 'clean' or 'noisy'         

% Load training and test data into memory
trnfile = sprintf('../data/%s_train_digits.mat',dataType);
tstfile = sprintf('../data/%s_test_digits.mat',dataType);
load(trnfile);                              % Load data structure trainData
load(tstfile);                              % Load data structure testData

% Extract 100 from each class in trainData{}
trainData = extract_data(trainData, 100);

% Extract 100 test samples from each class to reduce scoring time. Do not
% use this when reporting results.
testData = extract_data(testData, 100);


fprintf('Start evaluating %s digit data using %s-svm\n',dataType,kerType);

% Train one one-vs-rest SVM for each class
SVMmodel = train_svm_model(trainData, kerType, kerPara, C);

% For each test pattern (testData{k}(t,:)), present it to the classifier to find
% the most likely class (label). Then, compare the the true label to see if
% the classification decision is correct. Sum all the correct classification counts
% to estimate the overall accuracy.
totalTest = 0;
nCorrect = 0;
for k = 1:length(testData),
    nTest = size(testData{k},1);
    fprintf('Evaluating %d samples of digit %d\n', nTest, k-1);
    totalTest = totalTest + nTest;
    label = zeros(1,nTest);
    for t = 1:nTest,
        label(t) = svm_classification(SVMmodel, testData{k}(t,:));
    end
    nCorrect = nCorrect + length(find(label==k-1));
end

acc = 100*nCorrect/totalTest;
fprintf('Accuracy = %.2f%%\n',acc);