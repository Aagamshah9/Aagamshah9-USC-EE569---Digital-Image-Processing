clc;
clear all;
noisy = readraw('Corn_noisy.raw',320,320);
output = readraw('Corn_gray.raw',320,320);
[PSNR, y_est] = BM3D(output, noisy, 30, 'np', 1);
%count = writeraw(y_est,'Corn_BM3D.raw');