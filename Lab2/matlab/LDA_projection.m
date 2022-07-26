% Find LDA projection matrix and use it to reconstruct the digit images. 
%
% Author: M.W. Mak (Oct. 2015)

clear; close all;

dataType = 'noisy';                         % Type of data, can be 'clean' or 'noisy'         
digit = 8;
n_ev = [5 9];                               % Dimension of subspace

% Load training and test data into memory
trnfile = sprintf('../data/%s_train_digits.mat',dataType);
tstfile = sprintf('../data/%s_test_digits.mat',dataType);
load(trnfile);                              % Load data structure trainData
load(tstfile);                              % Load data structure testData

% Compute the between-class and within-class covariance matrices
X = cell2mat(trainData);                    % X contains all digits (row vectors)
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
    Sb = Sb + N_k*(mu_k-mu)'*(mu_k-mu);         % Note that mu_k and mu are row vectors           
end

% Find the K-1 largest eigenvectors and eigenvalues of AU=Lambda B, i.e.,
% find the eigenvectors of inv(Sw)*Sb
[U,Lambda] = eigs(Sb,Sw,nClasses-1);

% Normalize V to make unit norm
for i=1:size(U,2),
    U(:,i) = U(:,i)/norm(U(:,i));
end

% Plot the first 5 eigenvectors
for i=1:5,
    subplot(1,5,i), imagesc(reshape(U(:,i),28,28)'); colormap(1-gray);
end

% Project test vectors on the first 3 axes defined by the 3 eigenvectors with 
% the largest eigenvalue
W = U(:,1:3);
figure;
markers = {'+','o','*','.','x','s','d','^','v','>','<','p','h'};
colors = {'b','g','r','c','m','y','k'};
for k = 1:nClasses,
    Y = testData{k}(1:100,:);
    Yprj = (Y-repmat(mu,size(Y,1),1)) * W;                           % Y contains row vectors
    m = mod(k,numel(markers))+1;
    c = mod(k,numel(colors))+1;
    plot3(Yprj(:,1), Yprj(:,2), Yprj(:,3),...
        'LineStyle', 'None', 'Marker',markers{m},'Color',colors{c}); hold on;
end
grid on;
xlabel('e_1'); ylabel('e_2'); zlabel('e_3'); axis equal;

% Extract one test image and display it
y = testData{digit+1}(2,:);
figure; subplot(1,length(n_ev)+1,1), imagesc(reshape(y,28,28)'); colormap(1-gray);

% Project the test image and reconstruct it. Display the reconstructed image
for i = 1:length(n_ev),
    V = U(:,1:n_ev(i));                         % Use the first n_ev(i) eigenvectors only

    % Project the test images
    y_prj = V'*(y-mu)';
    
    % Reconstruct the test images from low-dim subspace to the original space
    y_rec = V*y_prj;
    y_rec = y_rec'+mu;                             % Convert to row vector and add back offset

    % Display original and reconstructed images
    subplot(1,length(n_ev)+1,i+1), imagesc(reshape(y_rec,28,28)'); colormap(1-gray);
end
