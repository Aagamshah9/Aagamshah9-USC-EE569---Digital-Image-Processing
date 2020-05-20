% clear all;
% clc;
L = [1 4 6 4 1];
E = [-1 -2 0 2 1];
S = [-1 0 2 0 -1];
W = [-1 2 0 -2 1];
R = [1 -4 6 -4 1];
Input_image = readraw('Comp.raw',600,450);
row = 450;
col = 600;
%% Boundary Extension %%
mid_image = double(Input_image);
I = mid_image;
I = double(mid_image - (sum(sum(mid_image))/(450*600)));
J = zeros(row+4,col+4);
J(3:row+2,3:col+2)=I;        % Center of an image
J(2,3:col+2)=I(2,:);         % 2nd Upper Row
J(1,3:col+2)=I(3,:);         % 1st Upper Row %
J(row+3,3:col+2)=I(row-1,:); % 2nd Lower Row
J(row+4,3:col+2)=I(row-2,:); % 1st Lower Row %
J(:,2)=J(:,4);               % 2nd Leftmost Column
J(:,1)=J(:,5);               % 1st Leftmost Column %
J(:,col+3)=J(:,col+1);       % 2nd Rightmost Column 
J(:,col+4)=J(:,col);         % 1st Rightmost Column %
%% Average Energy %%
r_map1=average_energy_1(L'*L,J,row,col);

o1=average_energy_1(L'*E,J,row,col);
o2=average_energy_1(L'*S,J,row,col);
o3=average_energy_1(L'*W,J,row,col);
o4=average_energy_1(L'*R,J,row,col);
o5=average_energy_1(E'*L,J,row,col);

r_map4=average_energy_1(E'*E,J,row,col);

o6=average_energy_1(E'*S,J,row,col);
o7=average_energy_1(E'*W,J,row,col);
o8=average_energy_1(E'*R,J,row,col);
o9=average_energy_1(S'*L,J,row,col);
o10=average_energy_1(S'*E,J,row,col);

r_map7=average_energy_1(S'*S,J,row,col);

o11=average_energy_1(S'*W,J,row,col);
o12=average_energy_1(S'*R,J,row,col);
o13=average_energy_1(W'*L,J,row,col);
o14=average_energy_1(W'*E,J,row,col);
o15=average_energy_1(W'*S,J,row,col);

r_map10=average_energy_1(W'*W,J,row,col);

o16=average_energy_1(W'*R,J,row,col);
o17=average_energy_1(R'*L,J,row,col);
o18=average_energy_1(R'*E,J,row,col);
o19=average_energy_1(R'*S,J,row,col);
o20=average_energy_1(R'*W,J,row,col);

r_map13=average_energy_1(R'*R,J,row,col);

r_map2 = ((o1+o5)/2);
r_map3 = ((o6+o10)/2);
r_map5 = ((o2+o9)/2);
r_map6 = ((o7+o14)/2);
r_map8 = ((o3+o13)/2);
r_map9 = ((o8+o18)/2);
r_map11 = ((o4+o17)/2);
r_map12 = ((o11+o15)/2);
r_map14 = ((o16+o20)/2);
r_map15 = ((o12+o19)/2);

%% Feature Window %%
 
n=15;
E1 = energy_feature_kernel(r_map1,row,col,n);
E2 = energy_feature_kernel(r_map2,row,col,n);
E3 = energy_feature_kernel(r_map3,row,col,n);
E4 = energy_feature_kernel(r_map4,row,col,n);
E5 = energy_feature_kernel(r_map5,row,col,n);
E6 = energy_feature_kernel(r_map6,row,col,n);
E7 = energy_feature_kernel(r_map7,row,col,n);
E8 = energy_feature_kernel(r_map8,row,col,n);
E9 = energy_feature_kernel(r_map9,row,col,n);
E10 = energy_feature_kernel(r_map10,row,col,n);
E11 = energy_feature_kernel(r_map11,row,col,n);
E12 = energy_feature_kernel(r_map12,row,col,n);
E13 = energy_feature_kernel(r_map13,row,col,n);
E14 = energy_feature_kernel(r_map14,row,col,n);
E15 = energy_feature_kernel(r_map15,row,col,n);

%% Energy Feature Normalization %%

count=1;
for i=1:row
    for j=1:col
        e=[E2(i,j) E3(i,j) E4(i,j) E5(i,j) E6(i,j) E7(i,j) E8(i,j) E9(i,j) E10(i,j) E11(i,j) E12(i,j) E13(i,j) E14(i,j) E15(i,j)];  
        normalised_energy=e./E1(i,j);
        matrix(count,:)=normalised_energy; 
        count=count+1;
    end
end

%% Advanced Segmentation %%
% Number of textures = 6
% PCA %
% [coeff,score] = pca(matrix);
segments=6;
% X=score*coeff';
%% Principal Component Analysis %%

% Step 1: Feature Matrix X with n(number of features) = 15 and m(number of data points) = 36.
X = zscore(matrix);
% Step 2: Calculate the covariance matrix
S = (1/36)*(X')*(X);
% Step 3: Computer SVD of Y
[U,S,V]=svd(S);
% Step 4: Dimensionality Reduction
Ur = U(:,1:6);
% Step 5: Transformation Step
Yr = Ur'*X';
Yt = Yr';
%% K-means Clustering

Z = Yt(:,1:segments);
idx=kmeans(Z,6);
variations=[0,51,102,153,204,255];
k=1;
for i=1:row
    for j=1:col
       segmented_image(i,j)=variations(1,idx(k,1));  
       k=k+1;
    end
end
Final_image = imfill(segmented_image);
figure(2);
imshow(uint8(Final_image));
title('Advanced Texture Segmented Image using PCA');