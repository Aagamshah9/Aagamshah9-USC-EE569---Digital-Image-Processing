I = readraw1('Hedwig.raw',512,512);
figure(1)
imshow(I);
title('RGB Image of Hedwig');

for i = 1:1:512
    for j = 1:1:512
        R(i,j) = I(i,j,1);
        G(i,j) = I(i,j,2);
        B(i,j) = I(i,j,3);
        x = (i-256)/256; % to bring it in range of -1 to 1
        y = (j-256)/256; % to bring it in range of -1 to 1
        u = x*sqrt(1-((y)^2)/2); % Elliptical Grid Mapping
        v = y*sqrt(1-((x)^2)/2); % Elliptical Grid Mapping
        u_new = round((u*256)+256);
        v_new = round((v*256)+256);
        O(u_new,v_new,1) = R(i,j);
        O(u_new,v_new,2) = G(i,j);
        O(u_new,v_new,3) = B(i,j);
    end
end

figure(2)
imshow(uint8(O));
title('Warped Image of Hedwig');


% for u_new = 1:1:512
%     for v_new = 1:1:512
%         if ((((u_new)^2)+((v_new)^2))<=1)
%         R1(u_new,v_new) = O(u_new,v_new,1);
%         G1(u_new,v_new) = O(u_new,v_new,2);
%         B1(u_new,v_new) = O(u_new,v_new,3);
%         u = (u_new-256)/256;
%         v = (v_new-256)/256;
%         u2 = double(u*u);
%         v2 = double(v*v);
%         twosqrt2 = double(2*sqrt(2));
%         subterm_x = double(2+u2-v2);
%         subterm_y = double(2-u2+v2);
%         term_x1 = double(subterm_x + (u*twosqrt2));
%         term_x2 = double(subterm_x - (u*twosqrt2));
%         term_y1 = double(subterm_y + (v*twosqrt2));
%         term_y2 = double(subterm_y - (v*twosqrt2));
%         x = (0.5*sqrt(term_x1))-(0.5*sqrt(term_x2));
%         y = (0.5*sqrt(term_y1))-(0.5*sqrt(term_y2));
%         x_1 = round((x*256)+256);
%         y_1 = round((y*256)+256);
%         S(x_1,y_1,1) = R1(u_new,v_new);
%         S(x_1,y_1,2) = G1(u_new,v_new);
%         S(x_1,y_1,3) = B1(u_new,v_new);
%         end
%     end
% end
% 
% figure(3)
% imshow(uint8(S));
% title('Reconstructed Image of Hedwig');

I = readraw1('Raccoon.raw',512,512);
figure(4)
imshow(I);
title('RGB Image of Raccoon');

for i = 1:1:512
    for j = 1:1:512
        R(i,j) = I(i,j,1);
        G(i,j) = I(i,j,2);
        B(i,j) = I(i,j,3);
        x = (i-256)/256;
        y = (j-256)/256;
        u = x*sqrt(1-((y)^2)/2); % Elliptical Grid Mapping
        v = y*sqrt(1-((x)^2)/2); % Elliptical Grid Mapping
        u_new = round((u*256)+256);
        v_new = round((v*256)+256);
        O(u_new,v_new,1) = R(i,j);
        O(u_new,v_new,2) = G(i,j);
        O(u_new,v_new,3) = B(i,j);
    end
end

figure(5)
imshow(uint8(O));
title('Warped Image of Raccoon');


% for u_new = 1:1:512
%     for v_new = 1:1:512
%         if ((((u_new)^2)+((v_new)^2))<=1)
%         R1(u_new,v_new) = O(u_new,v_new,1);
%         G1(u_new,v_new) = O(u_new,v_new,2);
%         B1(u_new,v_new) = O(u_new,v_new,3);
%         u = (u_new-256)/256;
%         v = (v_new-256)/256;
%         u2 = double(u*u);
%         v2 = double(v*v);
%         twosqrt2 = double(2*sqrt(2));
%         subterm_x = double(2+u2-v2);
%         subterm_y = double(2-u2+v2);
%         term_x1 = double(subterm_x + (u*twosqrt2));
%         term_x2 = double(subterm_x - (u*twosqrt2));
%         term_y1 = double(subterm_y + (v*twosqrt2));
%         term_y2 = double(subterm_y - (v*twosqrt2));
%         x = (0.5*sqrt(term_x1))-(0.5*sqrt(term_x2));
%         y = (0.5*sqrt(term_y1))-(0.5*sqrt(term_y2));
%         x_1 = round((x*256)+256);
%         y_1 = round((y*256)+256);
%         S(x_1,y_1,1) = R1(u_new,v_new);
%         S(x_1,y_1,2) = G1(u_new,v_new);
%         S(x_1,y_1,3) = B1(u_new,v_new);
%         end
%     end
% end
% 
% figure(6)
% imshow(uint8(S));
% title('Reconstructed Image of Raccoon');

I = readraw1('bb8.raw',512,512);
figure(7)
imshow(I);
title('RGB Image of bb8');

for i = 1:1:512
    for j = 1:1:512
        R(i,j) = I(i,j,1);
        G(i,j) = I(i,j,2);
        B(i,j) = I(i,j,3);
        x = (i-256)/256;
        y = (j-256)/256;
        u = x*sqrt(1-((y)^2)/2); % Elliptical Grid Mapping
        v = y*sqrt(1-((x)^2)/2); % Elliptical Grid Mapping
        u_new = round((u*256)+256);
        v_new = round((v*256)+256);
        O(u_new,v_new,1) = R(i,j);
        O(u_new,v_new,2) = G(i,j);
        O(u_new,v_new,3) = B(i,j);
    end
end

figure(8)
imshow(uint8(O));
title('Warped Image of bb8');


% for u_new = 1:1:512
%     for v_new = 1:1:512
%         if ((((u_new)^2)+((v_new)^2))<=1)
%         R1(u_new,v_new) = O(u_new,v_new,1);
%         G1(u_new,v_new) = O(u_new,v_new,2);
%         B1(u_new,v_new) = O(u_new,v_new,3);
%         u = (u_new-256)/256;
%         v = (v_new-256)/256;
%         u2 = double(u*u);
%         v2 = double(v*v);
%         twosqrt2 = double(2*sqrt(2));
%         subterm_x = double(2+u2-v2);
%         subterm_y = double(2-u2+v2);
%         term_x1 = double(subterm_x + (u*twosqrt2));
%         term_x2 = double(subterm_x - (u*twosqrt2));
%         term_y1 = double(subterm_y + (v*twosqrt2));
%         term_y2 = double(subterm_y - (v*twosqrt2));
%         x = (0.5*sqrt(term_x1))-(0.5*sqrt(term_x2));
%         y = (0.5*sqrt(term_y1))-(0.5*sqrt(term_y2));
%         x_1 = round((x*256)+256);
%         y_1 = round((y*256)+256);
%         S(x_1,y_1,1) = R1(u_new,v_new);
%         S(x_1,y_1,2) = G1(u_new,v_new);
%         S(x_1,y_1,3) = B1(u_new,v_new);
%         end
%     end
% end
% 
% figure(9)
% imshow(uint8(S));
% title('Reconstructed Image of bb8');