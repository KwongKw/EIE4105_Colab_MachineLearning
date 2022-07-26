% Find the principal components of a digit using PCA. Note that in practice,
% PCA is unsupervised and therefore we should not use the labels (or do
% not have the labels). However, if we want to see the digit patterns in
% the eigenvectors, we may find the PC componenet using the image of a
% single digit only.
%
% Author: M.W. Mak (Oct. 2015)

clear; close all;

dataType = 'noisy';                         % Type of data, can be 'clean' or 'noisy'         
digit = 8;
nPC = [9 50 100 200];

% Load training and test data into memory
trnfile = sprintf('../data/%s_train_digits.mat',dataType);
tstfile = sprintf('../data/%s_test_digits.mat',dataType);
load(trnfile);                              % Load data structure trainData
load(tstfile);                              % Load data structure testData

% Find the PC of a digit
%X = trainData{digit+1};                    % Digit-wise PCA
X = cell2mat(trainData);                    % PCA on all digits
Sigma = cov(X);
[U, Lambda] = eigs(Sigma,max(nPC));         % Google "Matlab eigs" to study this function

% Plot the first 5 eigenvectors
for i=1:5,
    subplot(1,5,i), imagesc(reshape(U(:,i),28,28)'); colormap(1-gray);
end

% Plot the last 5 eigenvectors
D = size(U,2);
figure;
for i = 5:-1:1,
    j = D-i+1;
    subplot(1,5,5-i+1), imagesc(reshape(U(:,j),28,28)'); colormap(1-gray);
end

% Project test vectors on the first 3 axes defined by the 3 eigenvectors with 
% the largest eigenvalue
V = U(:,1:3);
figure;
markers = {'+','o','*','.','x','s','d','^','v','>','<','p','h'};
colors = {'b','g','r','c','m','y','k'};
for k = 1:length(testData),
    Y = trainData{k}(500:600,:);
    Yprj = (Y-repmat(mean(X,1),size(Y,1),1)) * V;
    m = mod(k,numel(markers))+1;
    c = mod(k,numel(colors))+1;
    plot3(Yprj(:,1), Yprj(:,2), Yprj(:,3),...
        'LineStyle', 'None', 'Marker',markers{m},'Color',colors{c}); hold on;
end
grid on;
xlabel('u_1'); ylabel('u_2'); zlabel('u_3'); %axis equal;

% Extract one test image and display it
y = testData{digit+1}(2,:);
figure; subplot(1,length(nPC)+1,1), imagesc(reshape(y,28,28)'); colormap(1-gray);

% Project the test image and reconstruct it. Display the reconstructed image
for i = 1:length(nPC),
    V = U(:,1:nPC(i));                          % Use the first nPC principal components only

    % Project the test images
    y_prj = V'*(y' - mean(X,1)');               % Remove mean before projection

    % Reconstruct the test images from low-dim subspace to the original space
    y_rec = V*y_prj;
    y_rec = y_rec' + mean(X,1);                 % Convert to row vector and add back offset

    % Display original and reconstructed images
    subplot(1,length(nPC)+1,i+1), imagesc(reshape(y_rec,28,28)'); colormap(1-gray);
end
