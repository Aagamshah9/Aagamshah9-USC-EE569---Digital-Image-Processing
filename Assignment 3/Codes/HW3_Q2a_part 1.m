I = readraw('fan.raw',558,558);
figure(1);
imshow(I);
title('Fan');
load('S_conditional.mat');
load('T_conditional.mat');
load('K_conditional.mat');
I = I/255;

% Boundary Extension %

N = 558;
I_new(2:N+1,2:N+1)=I;      % Center of an image
I_new(1,2:N+1)=I(2,:);     % Upper Row
I_new(N+2,2:N+1)=I(N-1,:); % Lower Row
I_new(2:N+1,1)=I(:,2);     % Leftmost Column
I_new(2:N+1,N+2)=I(:,N-1); % Rightmost Column 
F = I_new;
M = zeros(560,560);
flag = true;
G = zeros(560,560);

% Conditional Logic %

a = 1;
while(flag)
G_old = G;
for i = 2:N+1
    for j = 2:N+1
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
title('Fan Conditional');

% Unconditional Logic %

for i = 2:N+1
    for j = 2:N+1
        G(i,j) = S_T_unconditional(F(i,j),M(i,j),M(i,j+1),M(i-1,j+1),M(i-1,j),M(i-1,j-1),M(i,j-1),M(i+1,j-1),M(i+1,j),M(i+1,j+1));
    end
end
figure(3);
imshow(G);
title('Fan Unconditional');
a = a + 1;
if(isequal(G_old,G))
    flag = false;
end
F=G;
M=zeros(560,560);
end
figure(4);
imshow(G);
title('Morphed image');