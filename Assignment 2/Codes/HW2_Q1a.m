C = readraw1('Gallery.raw',321,481);
figure(1)
imshow(C);
title('RGB Image');
R = C(:,:,1);
G = C(:,:,2);
B = C(:,:,3);
Y = ((0.2989*R)+(0.5870*G)+(0.1140*B));
figure(2)
imshow(Y);
title('Grayscale image');
X=double(Y);
Gx = zeros(481,321);
Gy = zeros(481,321);
for i = 2:320
    for j = 2:480
        %Sobel mask for x-gradient
        Gx=(((X(i-1,j+1))+((2)*X(i,j+1))+(X(i+1,j+1)))-((X(i-1,j-1))+((2)*X(i,j-1))+(X(i+1,j-1))));
        %Sobel mask for y-gradient
        Gy=(((X(i-1,j-1))+((2)*X(i-1,j))+(X(i-1,j+1)))-((X(i+1,j-1))+((2)*X(i+1,j))+(X(i+1,j+1))));
        %Gradient of final image
        Y(i,j)=(sqrt((Gx.^2)+(Gy.^2)));
        %Normalised output image
        Y_norm(i,j)=255*((Y(i,j)-min(Y(:)))/(max(Y(:))-min(Y(:))));
        %Thresholding step
        if Y_norm(i,j)>127
            Y_norm(i,j)=255;
        else
            Y_norm(i,j)=0;
        end
    end
end
%x-Gradient of final image
x_norm=((Gx-min(min(Gx)))./(max(max(Gx))-min(min(Gx)))).*255;
%y-Gradient of final image
y_norm=((Gy-min(min(Gx)))./(max(max(Gy))-min(min(Gy)))).*255;
figure(3)
imshow(uint8(x_norm));
title('Normalised x-gradient');
% figure(4)
imshow(uint8(y_norm));
title('Normalised y-gradient');
% figure(5)
imshow(Y);
title('Sobel Edge Detection');
figure(6)
imshow(Y_norm);
title('Sobel Edge Detection final');

for i=2:320
    for j=2:480
       if(Y_norm(i,j) > 127)
           edge_map(i,j)=0; 
       else
           edge_map(i,j)=255;
       end
    end
end

figure(7);
imshow(uint8(edge_map));
title('Threshold = 85%');
