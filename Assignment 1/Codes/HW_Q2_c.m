img_n = readraw('Corn_noisy.raw',320,320);
img_f = nlm(img_n, 3, 6, 16, (0.3*((10)^2)), 10, 0);
imshow(uint8(img_f));
count = writeraw(img_f,'Corn_NLM.raw');