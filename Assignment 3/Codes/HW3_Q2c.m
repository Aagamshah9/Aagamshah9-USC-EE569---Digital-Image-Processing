A = readraw1('PCB.raw',239,124);
figure(1);
imshow(A);
title('PCB');
R = A(:,:,1);
G = A(:,:,2);
B = A(:,:,3);
C = ((0.2989*R)+(0.5870*G)+(0.1140*B));
figure(2)
imshow(C);
title('Grayscale image of PCB');
I = zeros(239,124);
for i = 1:239
    for j = 1:124
        if C(i,j)>80
            I(i,j)=255;
        else
            I(i,j)=0;
        end
    end
end
figure(3)
imshow(I);
title('Binary image of PCB');
load('S_conditional.mat');
load('T_conditional.mat');
load('K_conditional.mat');
I = I/255;

% Boundary Extension %

R = 239;
C = 124;
I_new(2:R+1,2:C+1)=I;      % Center of an image
I_new(1,2:C+1)=I(2,:);     % Upper Row
I_new(R+2,2:C+1)=I(R-1,:); % Lower Row
I_new(2:R+1,1)=I(:,2);     % Leftmost Column
I_new(2:R+1,C+2)=I(:,C-1); % Rightmost Column 
F = I_new;
M = zeros(241,126);
flag = true;
G = zeros(241,126);

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
title('PCB Conditional');

% Unconditional Logic %

for i = 2:R+1
    for j = 2:C+1
        G(i,j) = S_T_unconditional(F(i,j),M(i,j),M(i,j+1),M(i-1,j+1),M(i-1,j),M(i-1,j-1),M(i,j-1),M(i+1,j-1),M(i+1,j),M(i+1,j+1));
    end
end
figure(3);
imshow(G);
title('PCB Unconditional');
a = a + 1;
if(isequal(G_old,G))
    flag = false;
end
F=G;
M=zeros(241,126);
end
figure(4);
imshow(G);
title('Morphed image');

% To find the total number of holes in PCB
counter = 0;
for i = 1:241
    for j = 1:126
        if G(i,j)==1 && G(i,j+1)==0 && G(i-1,j+1)==0 && G(i-1,j)==0 && G(i-1,j-1)==0 && G(i,j-1)==0 && G(i+1,j-1)==0 && G(i+1,j)==0 && G(i+1,j+1)==0 
            counter = counter + 1;
        end
    end
end
disp(counter);