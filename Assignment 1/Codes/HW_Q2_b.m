H = readraw('Corn_noisy.raw',320,320);
I = zeros(320,320);
X = readraw('Corn_gray.raw',320,320);
%W = zeros(5,5);
%disp(W);

%   Gaussian Weight Function   %
%------------------------------%

Height = 320;
Width = 320;
r = Height-2;
c = Width-2;

% Creating a Kernel %
% Mask Height = MH = 5
% Mask Width = MW = 5

for i = 3:r
    for j = 3:c
        submatrix = H(i-2:i+2,j-2:j+2);
        %disp(submatrix);
        deno=0;
        num=0;
        for k = i-2:i+2
            for l = j-2:j+2
                K = ((i-k)^2);
                %disp(K);
                L = ((j-l)^2);
                %disp(L);
                J=((K+L)/(2*1));
                %disp(J);
                a=double(H(i,j));
                b=double(H(k,l));
                M=(((a-b)^2)/(2*1));
                %disp(M);
                Weight=(exp((-J-M)));
                %disp(Weight);
                deno=(deno+Weight);
                prod=(H(k,l)*Weight);
                %disp(prod);
                num=num+prod;
            end
        end
        I(i,j)=round(num/deno);
    end
end

subplot(1,2,1)
imshow(H);
subplot(1,2,2)
imshow(uint8(I));
count = writeraw(I,'Corn_Bilateral.raw');