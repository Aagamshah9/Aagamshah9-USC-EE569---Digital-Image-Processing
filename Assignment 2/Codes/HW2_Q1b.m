C = readraw1('Dogs.raw',321,481);
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
X=uint8(Y);
BW = edge(X,'canny',[0.15 0.2]);
figure(3)
imshow(BW);
title('Canny Edge Detection');