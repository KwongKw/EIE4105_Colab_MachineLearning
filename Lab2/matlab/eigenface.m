% Find the eigenvectors and eigenvalues of 400 facial images
% Perform PCA projection to project the high-dim facial images to low-dim 
% vectors

clear; close all;

% Load facial images to obtain the matrix faces(:,:) and the width and height
% of the images. Each row in faces(:,:) contains a face in row vector form.
load('../data/faces.mat');

% Divided into training and test faces
trnFaces = faces(1:350,:);
tstFaces = faces(351:end,:);
clear faces;                            % Free up memory

% Maximum no. of principal components is no. of training samples-1
maxnPC = size(trnFaces,1)-1;

% Define no. of principal components
nPC = [1 20 50 100 200 maxnPC];

% Plot the mean training face
mf = mean(trnFaces,1);
figure; imagesc(reshape(mf,width,height)'); colormap(gray)

%=============================================================================
% Solve the eigen problem that involve a big covariance matrix
% See http://en.wikipedia.org/wiki/Eigenface. This technique allows us
% to compute the covariane of NxN matrix rather than a DxD matrix where
% N is the number of training samples and D is the feature dimension.
% To make it easier for you to follow, I use the same symbols as in the Wiki page
%=============================================================================
% Step 1: Subtract the mean face and make T to contain column vectors of mean-subtracted faces
T = trnFaces;
T = (T - repmat(mf,size(T,1),1))';

% Step 2: Compute the eigenvectors of T'T to obtain u_i
[U,Lambda] = eigs(T'*T, maxnPC);

% Step 3: Noting that the eigenvectors (v_i) of TT' is Tu_i
V = T*U;

% Step 4: Normalize V to make unit norm
for i=1:size(V,2),
    V(:,i) = V(:,i)/norm(V(:,i));
end

%=============================================================================
% Plot the first 10 Eigenfaces
%=============================================================================
fHandle = figure;
set(fHandle, 'Position', [100, 300, 2000, 200]);
for i=1:10,
    subplot(1,10,i), imagesc(reshape(V(:,i),width,height)'); colormap(gray);
end

%=============================================================================
% Plot the last 5 Eigenfaces
%=============================================================================
D = size(U,2);
fHandle = figure;
set(fHandle, 'Position', [100, 400, 1000, 200]);
for i = 5:-1:1,
    j = D-i+1;
    subplot(1,5,5-i+1), imagesc(reshape(V(:,j),width,height)'); colormap(gray);
end

%=============================================================================
% Project 10 faces to nPC-dim space and then reconstruct them in the original space
%=============================================================================
% Project 10 faces and reconstruct them. Display the original on the first row
% and the reconstructed images on the 2nd row
nfaces = 10;
W = V(:,1:max(nPC));                        % Use the first nPC principal components only
y = cell(nfaces,1);
y_prj = cell(nfaces,1);
y_rec = cell(nfaces,1);
P = randperm(size(tstFaces,1));
for i = 1:nfaces,
    y{i} = tstFaces(P(i),:)';               % Get a random face
    y_prj{i} = W'*(y{i}-mf');               % Project the mean-subtracted face 
    y_rec{i} = W*y_prj{i}+mf';              % Reconstruct the face   
end

fHandle = figure;
set(fHandle, 'Position', [100, 500, 2000, 500]); 

% Display original and reconstructed images
for i=1:nfaces,
    subplot(2,10,i), imagesc(reshape(y{i},width,height)'); colormap(gray);
end
for i=1:nfaces,
    subplot(2,10,nfaces+i), imagesc(reshape(y_rec{i},width,height)'); colormap(gray);
end

%=============================================================================
% Project 1 face to nPC-dim space and then reconstruct them in the original space
%=============================================================================
fHandle = figure;
set(fHandle, 'Position', [100, 600, 1500, 250]); 
z = tstFaces(P(1),:)';                         % Randomly select one image
subplot(1,length(nPC)+1,1), imagesc(reshape(z,width,height)'); colormap(1-gray);
for i = 1:length(nPC),
    W = V(:,1:nPC(i));
    z_prj = W'*(z-mf');                     % Project the mean-subtracted face 
    z_rec = W*z_prj+mf';                    % Reconstruct the face
    subplot(1,length(nPC)+1,i+1), imagesc(reshape(z_rec,width,height)'); colormap(gray);
end

