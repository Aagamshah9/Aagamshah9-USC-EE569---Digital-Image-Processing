A = readraw('Geartooth.raw',250,250);
figure(1);
imshow(A);
title('Geartooth');
load('S_conditional.mat');
load('T_conditional.mat');
load('K_conditional.mat');
I = zeros(250,250);
for i = 1:250
    for j = 1:250
        if A(i,j)>127
            I(i,j)=255;
        else
            I(i,j)=0;
        end
    end
end
I = I/255;

% Boundary Extension %

N = 250;
I_new(2:N+1,2:N+1)=I;      % Center of an image
I_new(1,2:N+1)=I(2,:);     % Upper Row
I_new(N+2,2:N+1)=I(N-1,:); % Lower Row
I_new(2:N+1,1)=I(:,2);     % Leftmost Column
I_new(2:N+1,N+2)=I(:,N-1); % Rightmost Column 
F = I_new;

for i = 1:126
    for j = 1:252
        F_half(i,j) = F(i,j);
    end
end
figure(2);
imshow(F_half);
J = imrotate(F_half,180);
figure(3);
imshow(J);
C = vertcat(F_half,J);
figure(4);
imshow(C);
for i = 1:252
    for j = 1:126
        C_half(i,j) = C(i,j);
    end
end
figure(5);
imshow(C_half);
K = flip(C_half,2);
figure(6);
imshow(K);
L = flip(K,1);
figure(7);
imshow(L);
M = imfuse(K,L);
figure(8);
imshow(M);
N = flip(M,2);
figure(9);
imshow(N);
O = horzcat(N,M);
figure(10);
imshow(O);
P = rgb2gray(O);
figure(11);
imshow(P);
Q = zeros(252,252);
for i = 1:252
    for j = 1:252
        if P(i,j)>100
            Q(i,j) = 255;
        else
            Q(i,j) = 0;
        end
    end
end
figure(12);
imshow(Q);
F = imfill(F,'holes');
figure(13);
imshow((F));
X = uint8(F.*255);
Q = imfill(Q,'holes');
figure(14);
imshow((Q));
Y = uint8(Q);
Final = imsubtract(Y,X);
figure(15);
imshow(Final);