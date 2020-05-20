clear all;
clc;
L = [1 4 6 4 1];
E = [-1 -2 0 2 1];
S = [-1 0 2 0 -1];
W = [-1 2 0 -2 1];
R = [1 -4 6 -4 1];
N = 128;

%% For Test Dataset %%
I1 = readraw('1.raw',128,128);
figure(1)
imshow(I1);
% Boundary Extension %
Test1 = extended_boundary_image(I1,N);
E1 = updated_feature_vector_function(L,E,S,W,R,Test1,N);
I2 = readraw('2.raw',128,128);
figure(2)
imshow(I2);
% Boundary Extension %
Test2 = extended_boundary_image(I2,N);
E2 = updated_feature_vector_function(L,E,S,W,R,Test2,N);
I3 = readraw('3.raw',128,128);
figure(3)
imshow(I3);
% Boundary Extension %
Test3 = extended_boundary_image(I3,N);
E3 = updated_feature_vector_function(L,E,S,W,R,Test3,N);
I4 = readraw('4.raw',128,128);
figure(4)
imshow(I4);
% Boundary Extension %
Test4 = extended_boundary_image(I4,N);
E4 = updated_feature_vector_function(L,E,S,W,R,Test4,N);
I5 = readraw('5.raw',128,128);
figure(5)
imshow(I5);
% Boundary Extension %
Test5 = extended_boundary_image(I5,N);
E5 = updated_feature_vector_function(L,E,S,W,R,Test5,N);
I6 = readraw('6.raw',128,128);
figure(6)
imshow(I6);
% Boundary Extension %
Test6 = extended_boundary_image(I6,N);
E6 = updated_feature_vector_function(L,E,S,W,R,Test6,N);
I7 = readraw('7.raw',128,128);
figure(7)
imshow(I7);
% Boundary Extension %
Test7 = extended_boundary_image(I7,N);
E7 = updated_feature_vector_function(L,E,S,W,R,Test7,N);
I8 = readraw('8.raw',128,128);
figure(8)
imshow(I8);
% Boundary Extension %
Test8 = extended_boundary_image(I8,N);
E8 = updated_feature_vector_function(L,E,S,W,R,Test8,N);
I9 = readraw('9.raw',128,128);
figure(9)
imshow(I9);
% Boundary Extension %
Test9 = extended_boundary_image(I9,N);
E9 = updated_feature_vector_function(L,E,S,W,R,Test9,N);
I10 = readraw('10.raw',128,128);
figure(10)
imshow(I10);
% Boundary Extension %
Test10 = extended_boundary_image(I10,N);
E10 = updated_feature_vector_function(L,E,S,W,R,Test10,N);
I11 = readraw('11.raw',128,128);
figure(11)
imshow(I11);
% Boundary Extension %
Test11 = extended_boundary_image(I11,N);
E11 = updated_feature_vector_function(L,E,S,W,R,Test11,N);
I12 = readraw('12.raw',128,128);
figure(12)
imshow(I12);
% Boundary Extension %
Test12 = extended_boundary_image(I12,N);
E12 = updated_feature_vector_function(L,E,S,W,R,Test12,N);

E_Final = [E1;E2;E3;E4;E5;E6;E7;E8;E9;E10;E11;E12];

%% Principal Component Analysis %%

% Step 1: Feature Matrix X with n(number of features) = 15 and m(number of data points) = 12.
X = zscore(E_Final);
% Step 2: Calculate the covariance matrix
Q = (1/12)*(X')*(X);
% Step 3: Computer SVD of Y
[U,S,V]=svd(Q);
% Step 4: Dimensionality Reduction
Ur = U(:,1:3);
% Step 5: Transformation Step
Yr = Ur'*X';
Yt = Yr';

% labels={'1','2','3','4','5','6','7','8','9','10','11','12'};
% for i=1:12
% figure(2);
% plot3(Yt(i,1),Yt(i,2),Yt(i,3),'*');
% hold on;
% grid on;
% text(Yt(i,1),Yt(i,2),Yt(i,3),labels(i));
% end 

x = Yr(1,:)';
y = Yr(2,:)';
z = Yr(3,:)';
scatter3(x,y,z,'r*');
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
zlabel('3rd Principal Component');
title('PCA-Reduced 3D Feature Space');