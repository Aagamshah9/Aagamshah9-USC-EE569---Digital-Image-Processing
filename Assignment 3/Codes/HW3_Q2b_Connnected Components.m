I = readraw('stars.raw',640,480);
figure(1);
imshow(I);
title('Stars');

N = 480;
M = 640;
X = zeros(480,640);
I = uint8(I);

for i=1:1:N     
    for j=1:1:M
        if I(i,j) >=150
           I(i,j) = 255;
        else
           I(i,j) = 0;
        end
    end
end
I = imbinarize(I);
figure(1);
imshow(I);
for i=1:1:N    
    for j=1:1:M
        if (I(i,j) == 1)
           X(i,j) = -1;
        end
    end
end
label = 0;
for i=2:1:N    
    for j=2:1:M-1
        if (X(i,j) == -1)
            S = [X(i-1,j-1) X(i-1,j) X(i-1,j+1) X(i,j-1)];
            if ((S(1,1) == 0) && (S(1,2) == 0) && (S(1,3) == 0) && (S(1,4) == 0))
                label = label + 1;
                X(i,j) = label;
            else
                Min = min(S(S > 0));
                X(i,j) = Min;
                if (S(1,1) > 0)
                    X(i-1,j-1) = Min;
                end
                if (S(1,2) > 0)
                    X(i-1,j) = Min;
                end
                if (S(1,3) > 0)
                    X(i-1,j+1) = Min;
                end
                if (S(1,4) > 0)
                    X(i,j-1) = Min;
                end
            end
        end
    end
end
A = X(:);
B = sort(A); 
size_no = 0;
hist = 1;
count = 0;
m = 0;
C = zeros(114,1);
for k = 2:1:(M*N)
    if (B(k-1,1) ~= B(k,1))
        count = count + 1;  
        m = m + 1;
        hist = 1;
    end
    if((B(k-1,1) == B(k,1)) && (B(k,1) > 0))
        hist = hist + 1;   
        C(m,1) = hist;      
    end
end
D = sort(C);
for l = 2:1:(size(D))
    if (D(l-1,1) ~= D(l,1))
        size_no = size_no + 1;   
    end
end
disp(count);