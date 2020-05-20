I = readraw('stars.raw',640,480);
figure(1);
imshow(I);
title('Stars');
load('S_conditional.mat');
I = I/255;

% Boundary Extension %

R = 480;
C = 640;
I_new(2:R+1,2:C+1)=I;      % Center of an image
I_new(1,2:C+1)=I(2,:);     % Upper Row
I_new(R+2,2:C+1)=I(R-1,:); % Lower Row
I_new(2:R+1,1)=I(:,2);     % Leftmost Column
I_new(2:R+1,C+2)=I(:,C-1); % Rightmost Column 
F = I_new;
M = zeros(482,642);
flag = true;
G = zeros(482,642);

% Conditional Logic %

a = 1;
while(flag)
G_old = G;
for i = 2:R+1
    for j = 2:C+1
        if (F(i,j)==1)
            array = [F(i,j+1),F(i-1,j+1),F(i-1,j),F(i-1,j-1),F(i,j-1),F(i+1,j-1),F(i+1,j),F(i+1,j+1)];
            m=0;
            for n = 1:length(S_conditional)
                if(isequal(array,S_conditional(n,:)))
                    m=1;
                end
            end
            M(i,j)=m;
        end
    end
end
figure(2);
imshow(M);
title('Stars Conditional');

% Unconditional Logic %

for i = 2:R+1
    for j = 2:C+1
        G(i,j) = S_T_unconditional(F(i,j),M(i,j),M(i,j+1),M(i-1,j+1),M(i-1,j),M(i-1,j-1),M(i,j-1),M(i+1,j-1),M(i+1,j),M(i+1,j+1));
    end
end
figure(3);
imshow(G);
title('Stars Unconditional');
a = a + 1;
if(isequal(G_old,G))
    flag = false;
end
F=G;
M=zeros(482,642);
end
figure(4);
imshow(G);
title('Morphed image');
% To find the total number of stars
counter = 0;
for i = 1:482
    for j = 1:642
        if G(i,j)==1
            counter = counter + 1;
        end
    end
end
disp(counter);
% sum(G(:) == 1)