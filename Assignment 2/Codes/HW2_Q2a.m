%Part 1

F = readraw('LightHouse.raw',750,500);
figure(1)
imshow(F);
title('Original Light House Image');
G = double(F);
for i = 1:500
    for j = 1:750
        if F(i,j)>=0 && F(i,j)<70
            G(i,j)=0;
        elseif F(i,j)>=70 && F(i,j)<256
            G(i,j)=255;
        end
    end
end
figure(2)
imshow(G);
title('Fixed Thresholding Light House Image');

%Part 2

H = double(F);
r = zeros(500,750);
for i = 1:500
    for j = 1:750
        r(i,j) = randi([0,255]);
        if F(i,j)>=0 && F(i,j)<r(i,j)
            H(i,j)=0;
        elseif F(i,j)>=r(i,j) && F(i,j)<256
            H(i,j)=255;
        end
    end
end
figure(3)
imshow(H);
title('Random Thresholding Light House Image');

%Part 3

J = double(F);
I2 = [1 2;3 0];
I4 = [((4*I2)+1) ((4*I2)+2);((4*I2)+3) ((4*I2))];
I8 = [((4*I4)+1) ((4*I4)+2);((4*I4)+3) ((4*I4))];
I16 = [((4*I8)+1) ((4*I8)+2);((4*I8)+3) ((4*I8))];
I32 = [((4*I16)+1) ((4*I16)+2);((4*I16)+3) ((4*I16))];

N = 2;
T = zeros(N,N);
for x = 1:N
    for y = 1:N
        T(x,y) = ((I2(x,y)+0.5)/(N^2))*255;
    end
end

I = double(F);
for i = 1:500
    for j = 1:750
        if F(i,j)<T(mod(i,N)+1,mod(j,N)+1)
            I(i,j)=0;
        else
            I(i,j)=255;
        end
    end
end
figure(4)
imshow(I);
title('Dithering Matrix Light House Image');