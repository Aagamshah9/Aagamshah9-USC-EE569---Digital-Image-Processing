clear all;
clc;
L = [1 4 6 4 1];
E = [-1 -2 0 2 1];
S = [-1 0 2 0 -1];
W = [-1 2 0 -2 1];
R = [1 -4 6 -4 1];
N = 128;

%% For Blanket Train Dataset %%
I1 = readraw('blanket1.raw',128,128);
% figure(1)
% imshow(I1);
% Boundary Extension %
Train1 = extended_boundary_image(I1,N);
E1 = updated_feature_vector_function(L,E,S,W,R,Train1,N);
I2 = readraw('blanket2.raw',128,128);
% figure(2)
% imshow(I2);
% Boundary Extension %
Train2 = extended_boundary_image(I2,N);
E2 = updated_feature_vector_function(L,E,S,W,R,Train2,N);
I3 = readraw('blanket3.raw',128,128);
% figure(3)
% imshow(I3);
% Boundary Extension %
Train3 = extended_boundary_image(I3,N);
E3 = updated_feature_vector_function(L,E,S,W,R,Train3,N);
I4 = readraw('blanket4.raw',128,128);
% figure(4)
% imshow(I4);
% Boundary Extension %
Train4 = extended_boundary_image(I4,N);
E4 = updated_feature_vector_function(L,E,S,W,R,Train4,N);
I5 = readraw('blanket5.raw',128,128);
% figure(5)
% imshow(I5);
% Boundary Extension %
Train5 = extended_boundary_image(I5,N);
E5 = updated_feature_vector_function(L,E,S,W,R,Train5,N);
I6 = readraw('blanket6.raw',128,128);
% figure(6)
% imshow(I6);
% Boundary Extension %
Train6 = extended_boundary_image(I6,N);
E6 = updated_feature_vector_function(L,E,S,W,R,Train6,N);
I7 = readraw('blanket7.raw',128,128);
% figure(7)
% imshow(I7);
% Boundary Extension %
Train7 = extended_boundary_image(I7,N);
E7 = updated_feature_vector_function(L,E,S,W,R,Train7,N);
I8 = readraw('blanket8.raw',128,128);
% figure(8)
% imshow(I8);
% Boundary Extension %
Train8 = extended_boundary_image(I8,N);
E8 = updated_feature_vector_function(L,E,S,W,R,Train8,N);
I9 = readraw('blanket9.raw',128,128);
% figure(9)
% imshow(I9);
% Boundary Extension %
Train9 = extended_boundary_image(I9,N);
E9 = updated_feature_vector_function(L,E,S,W,R,Train9,N);

%% For Brick Train Dataset %%
I10 = readraw('brick1.raw',128,128);
% figure(10)
% imshow(I10);
% Boundary Extension %
Train10 = extended_boundary_image(I10,N);
E10 = updated_feature_vector_function(L,E,S,W,R,Train10,N);
I11 = readraw('brick2.raw',128,128);
% figure(11)
% imshow(I11);
% Boundary Extension %
Train11 = extended_boundary_image(I11,N);
E11 = updated_feature_vector_function(L,E,S,W,R,Train11,N);
I12 = readraw('brick3.raw',128,128);
% figure(12)
% imshow(I12);
% Boundary Extension %
Train12 = extended_boundary_image(I12,N);
E12 = updated_feature_vector_function(L,E,S,W,R,Train12,N);
I13 = readraw('brick4.raw',128,128);
% figure(13)
% imshow(I13);
% Boundary Extension %
Train13 = extended_boundary_image(I13,N);
E13 = updated_feature_vector_function(L,E,S,W,R,Train13,N);
I14 = readraw('brick5.raw',128,128);
% figure(14)
% imshow(I14);
% Boundary Extension %
Train14 = extended_boundary_image(I14,N);
E14 = updated_feature_vector_function(L,E,S,W,R,Train14,N);
I15 = readraw('brick6.raw',128,128);
% figure(15)
% imshow(I15);
% Boundary Extension %
Train15 = extended_boundary_image(I15,N);
E15 = updated_feature_vector_function(L,E,S,W,R,Train15,N);
I16 = readraw('brick7.raw',128,128);
% figure(16)
% imshow(I16);
% Boundary Extension %
Train16 = extended_boundary_image(I16,N);
E16 = updated_feature_vector_function(L,E,S,W,R,Train16,N);
I17 = readraw('brick8.raw',128,128);
% figure(17)
% imshow(I17);
% Boundary Extension %
Train17 = extended_boundary_image(I17,N);
E17 = updated_feature_vector_function(L,E,S,W,R,Train17,N);
I18 = readraw('brick9.raw',128,128);
% figure(18)
% imshow(I18);
% Boundary Extension %
Train18 = extended_boundary_image(I18,N);
E18 = updated_feature_vector_function(L,E,S,W,R,Train18,N);

%% For Grass Train Dataset %%
I19 = readraw('grass1.raw',128,128);
% figure(19)
% imshow(I19);
% Boundary Extension %
Train19 = extended_boundary_image(I19,N);
E19 = updated_feature_vector_function(L,E,S,W,R,Train19,N);
I20 = readraw('grass2.raw',128,128);
% figure(20)
% imshow(I20);
% Boundary Extension %
Train20 = extended_boundary_image(I20,N);
E20 = updated_feature_vector_function(L,E,S,W,R,Train20,N);
I21 = readraw('grass3.raw',128,128);
% figure(21)
% imshow(I21);
% Boundary Extension %
Train21 = extended_boundary_image(I21,N);
E21 = updated_feature_vector_function(L,E,S,W,R,Train21,N);
I22 = readraw('grass4.raw',128,128);
% figure(22)
% imshow(I22);
% Boundary Extension %
Train22 = extended_boundary_image(I22,N);
E22 = updated_feature_vector_function(L,E,S,W,R,Train22,N);
I23 = readraw('grass5.raw',128,128);
% figure(23)
% imshow(I23);
% Boundary Extension %
Train23 = extended_boundary_image(I23,N);
E23 = updated_feature_vector_function(L,E,S,W,R,Train23,N);
I24 = readraw('grass6.raw',128,128);
% figure(24)
% imshow(I24);
% Boundary Extension %
Train24 = extended_boundary_image(I24,N);
E24 = updated_feature_vector_function(L,E,S,W,R,Train24,N);
I25 = readraw('grass7.raw',128,128);
% figure(25)
% imshow(I25);
% Boundary Extension %
Train25 = extended_boundary_image(I25,N);
E25 = updated_feature_vector_function(L,E,S,W,R,Train25,N);
I26 = readraw('grass8.raw',128,128);
% figure(26)
% imshow(I26);
% Boundary Extension %
Train26 = extended_boundary_image(I26,N);
E26 = updated_feature_vector_function(L,E,S,W,R,Train26,N);
I27 = readraw('grass9.raw',128,128);
% figure(27)
% imshow(I27);
% Boundary Extension %
Train27 = extended_boundary_image(I27,N);
E27 = updated_feature_vector_function(L,E,S,W,R,Train27,N);

%% For Rice Train Dataset %%
I28 = readraw('rice1.raw',128,128);
% figure(28)
% imshow(I28);
% Boundary Extension %
Train28 = extended_boundary_image(I28,N);
E28 = updated_feature_vector_function(L,E,S,W,R,Train28,N);
I29 = readraw('rice2.raw',128,128);
% figure(29)
% imshow(I29);
% Boundary Extension %
Train29 = extended_boundary_image(I29,N);
E29 = updated_feature_vector_function(L,E,S,W,R,Train29,N);
I30 = readraw('rice3.raw',128,128);
% figure(30)
% imshow(I30);
% Boundary Extension %
Train30 = extended_boundary_image(I30,N);
E30 = updated_feature_vector_function(L,E,S,W,R,Train30,N);
I31 = readraw('rice4.raw',128,128);
% figure(31)
% imshow(I31);
% Boundary Extension %
Train31 = extended_boundary_image(I31,N);
E31 = updated_feature_vector_function(L,E,S,W,R,Train31,N);
I32 = readraw('rice5.raw',128,128);
% figure(32)
% imshow(I32);
% Boundary Extension %
Train32 = extended_boundary_image(I32,N);
E32 = updated_feature_vector_function(L,E,S,W,R,Train32,N);
I33 = readraw('rice6.raw',128,128);
% figure(33)
% imshow(I33);
% Boundary Extension %
Train33 = extended_boundary_image(I33,N);
E33 = updated_feature_vector_function(L,E,S,W,R,Train33,N);
I34 = readraw('rice7.raw',128,128);
% figure(34)
% imshow(I34);
% Boundary Extension %
Train34 = extended_boundary_image(I34,N);
E34 = updated_feature_vector_function(L,E,S,W,R,Train34,N);
I35 = readraw('rice8.raw',128,128);
% figure(35)
% imshow(I35);
% Boundary Extension %
Train35 = extended_boundary_image(I35,N);
E35 = updated_feature_vector_function(L,E,S,W,R,Train35,N);
I36 = readraw('rice9.raw',128,128);
% figure(36)
% imshow(I36);
% Boundary Extension %
Train36 = extended_boundary_image(I36,N);
E36 = updated_feature_vector_function(L,E,S,W,R,Train36,N);

E_Final = [E1;E2;E3;E4;E5;E6;E7;E8;E9;E10;E11;E12;E13;E14;E15;E16;E17;E18;E19;E20;E21;E22;E23;E24;E25;E26;E27;E28;E29;E30;E31;E32;E33;E34;E35;E36];

%% Principal Component Analysis %%

% Step 1: Feature Matrix X with n(number of features) = 15 and m(number of data points) = 36.
X = zscore(E_Final);
% Step 2: Calculate the covariance matrix
S = (1/36)*(X')*(X);
% Step 3: Computer SVD of Y
[U,S,V]=svd(S);
% Step 4: Dimensionality Reduction
Ur = U(:,1:3);
% Step 5: Transformation Step
Yr = Ur'*X';
Y_t = Yr'; 

% labels={'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','35'};
% for i=1:36
% figure(2);
% plot3(Y_t(i,1),Y_t(i,2),Y_t(i,3),'o');
% hold on;
% grid on;
% text(Y_t(i,1),Y_t(i,2),Y_t(i,3),labels(i));
% end 
%% Plotting algorithm %%
x = Yr(1,:)';
y = Yr(2,:)';
z = Yr(3,:)';

x1 = x(1:9,:);
y1 = y(1:9,:);
z1 = z(1:9,:);

x2 = x(9:18,:);
y2 = y(9:18,:);
z2 = z(9:18,:);

x3 = x(18:27,:);
y3 = y(18:27,:);
z3 = z(18:27,:);

x4 = x(27:36,:);
y4 = y(27:36,:);
z4 = z(27:36,:);

scatter3(x1,y1,z1,'r');
hold on;
scatter3(x2,y2,z2,'g');
hold on;
scatter3(x3,y3,z3,'b');
hold on;
scatter3(x4,y4,z4,'m');
hold off;
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
zlabel('3rd Principal Component');
title('PCA-Reduced 3D Feature Space');