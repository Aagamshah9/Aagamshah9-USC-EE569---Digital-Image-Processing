% Part 1 %
% Separable Error Diffusion %

F = readraw1('Rose.raw',480,640);
R=F(:,:,1);
G=F(:,:,2);
B=F(:,:,3);
figure(1)
imshow(F);
title('Original Rose Image');

C = 255*ones(480,640) - (double(R));
M = 255*ones(480,640) - (double(G));
Y = 255*ones(480,640) - (double(B));

[rows,col]=size(C);
% For Cyan
b=zeros(rows,col);
H=C;
for i=1:rows    
    if mod(i,2)~=0
     if i~=rows
       for j=1:col
           if H(i,j)>127
               b(i,j)=255;
           else
               b(i,j)=0;
           end
           e = H(i,j)-b(i,j);
           if j==1 
               H(i,j+1)=H(i,j+1)+((7/16)*e);
               H(i+1,j)=H(i+1,j)+((5/16)*e);
               H(i+1,j+1)=H(i+1,j+1)+((1/16)*e);
           elseif j==col              
               H(i+1,j-1)=H(i+1,j-1)+((3/16)*e);
               H(i+1,j)=H(i+1,j)+((5/16)*e);
           else 
               H(i,j+1)=H(i,j+1)+((7/16)*e);
               H(i+1,j-1)=H(i+1,j-1)+((3/16)*e);
               H(i+1,j)=H(i+1,j)+((5/16)*e);
               H(i+1,j+1)=H(i+1,j+1)+((1/16)*e);
           end       
       end
     else
         for j=1:col
               if H(i,j)>127
                    b(i,j)=255;
               else
                    b(i,j)=0;
               end
               e = H(i,j)-b(i,j);
               if j~=col
                   H(i,j+1)=H(i,j+1)+((7/16)*e);
               end     
         end
     end
    else
        if i~=rows
            for j=col:-1:1
                if H(i,j)>127
                    b(i,j)=255;
                else
                    b(i,j)=0;
                end
                e = H(i,j)-b(i,j);
               if j==col
                   H(i,j-1)=H(i,j-1)+((7/16)*e);
                   H(i+1,j-1)=H(i+1,j-1)+((1/16)*e);
                   H(i+1,j)=H(i+1,j)+((5/16)*e);
               elseif j==1
                   H(i+1,j)=H(i+1,j)+((5/16)*e);
                   H(i+1,j+1)=H(i+1,j+1)+((3/16)*e);
               else
                   H(i,j-1)=H(i,j-1)+((7/16)*e);
                   H(i+1,j-1)=H(i+1,j-1)+((1/16)*e);
                   H(i+1,j)=H(i+1,j)+((5/16)*e);
                   H(i+1,j+1)=H(i+1,j+1)+((3/16)*e);
               end     
            end
        else
            for j=col:-1:1
               if H(i,j)>127
                    b(i,j)=255;
               else
                    b(i,j)=0;
               end
               e = H(i,j)-b(i,j);
               if j~=1
                   H(i,j-1)=H(i,j-1)+((7/16)*e);
               end     
            end
        end
    end
end
C_halftoned=b;

%For Magenta
b=zeros(rows,col);
H=M;
for i=1:rows    
    if mod(i,2)~=0
      if i~=rows
       for j=1:col
           if H(i,j)>127
               b(i,j)=255;
           else
               b(i,j)=0;
           end
           e = H(i,j)-b(i,j);
           if j==1 
               H(i,j+1)=H(i,j+1)+((7/16)*e);
               H(i+1,j)=H(i+1,j)+((5/16)*e);
               H(i+1,j+1)=H(i+1,j+1)+((1/16)*e);
           elseif j==col              
               H(i+1,j-1)=H(i+1,j-1)+((3/16)*e);
               H(i+1,j)=H(i+1,j)+((5/16)*e);
           else 
               H(i,j+1)=H(i,j+1)+((7/16)*e);
               H(i+1,j-1)=H(i+1,j-1)+((3/16)*e);
               H(i+1,j)=H(i+1,j)+((5/16)*e);
               H(i+1,j+1)=H(i+1,j+1)+((1/16)*e);
           end       
       end
      else
         for j=1:col
               if H(i,j)>127
                    b(i,j)=255;
               else
                    b(i,j)=0;
               end
               e = H(i,j)-b(i,j);
               if j~=col
                   H(i,j+1)=H(i,j+1)+((7/16)*e);
               end     
         end
     end
    else
        if i~=rows
            for j=col:-1:1
                if H(i,j)>127
                    b(i,j)=255;
                else
                    b(i,j)=0;
                end
                e = H(i,j)-b(i,j);
               if j==col
                   H(i,j-1)=H(i,j-1)+((7/16)*e);
                   H(i+1,j-1)=H(i+1,j-1)+((1/16)*e);
                   H(i+1,j)=H(i+1,j)+((5/16)*e);
               elseif j==1
                   H(i+1,j)=H(i+1,j)+((5/16)*e);
                   H(i+1,j+1)=H(i+1,j+1)+((3/16)*e);
               else
                   H(i,j-1)=H(i,j-1)+((7/16)*e);
                   H(i+1,j-1)=H(i+1,j-1)+((1/16)*e);
                   H(i+1,j)=H(i+1,j)+((5/16)*e);
                   H(i+1,j+1)=H(i+1,j+1)+((3/16)*e);
               end     
            end
        else
            for j=col:-1:1
               if H(i,j)>127
                    b(i,j)=255;
               else
                    b(i,j)=0;
               end
               e = H(i,j)-b(i,j);
               if j~=1
                   H(i,j-1)=H(i,j-1)+((7/16)*e);
               end     
            end
        end
    end
end
M_halftoned=b;

% For Yellow
b=zeros(rows,col);
H=Y;
for i=1:rows    
    if mod(i,2)~=0
      if i~=rows
       for j=1:col
           if H(i,j)>127
               b(i,j)=255;
           else
               b(i,j)=0;
           end
           e = H(i,j)-b(i,j);
           if j==1 
               H(i,j+1)=H(i,j+1)+((7/16)*e);
               H(i+1,j)=H(i+1,j)+((5/16)*e);
               H(i+1,j+1)=H(i+1,j+1)+((1/16)*e);
           elseif j==col              
               H(i+1,j-1)=H(i+1,j-1)+((3/16)*e);
               H(i+1,j)=H(i+1,j)+((5/16)*e);
           else 
               H(i,j+1)=H(i,j+1)+((7/16)*e);
               H(i+1,j-1)=H(i+1,j-1)+((3/16)*e);
               H(i+1,j)=H(i+1,j)+((5/16)*e);
               H(i+1,j+1)=H(i+1,j+1)+((1/16)*e);
           end       
       end
       else
         for j=1:col
               if H(i,j)>127
                    b(i,j)=255;
               else
                    b(i,j)=0;
               end
               e = H(i,j)-b(i,j);
               if j~=col
                   H(i,j+1)=H(i,j+1)+((7/16)*e);
               end     
         end
     end
    else
        if i~=rows
            for j=col:-1:1
                if H(i,j)>127
                    b(i,j)=255;
                else
                    b(i,j)=0;
                end
                e = H(i,j)-b(i,j);
               if j==col
                   H(i,j-1)=H(i,j-1)+((7/16)*e);
                   H(i+1,j-1)=H(i+1,j-1)+((1/16)*e);
                   H(i+1,j)=H(i+1,j)+((5/16)*e);
               elseif j==1
                   H(i+1,j)=H(i+1,j)+((5/16)*e);
                   H(i+1,j+1)=H(i+1,j+1)+((3/16)*e);
               else
                   H(i,j-1)=H(i,j-1)+((7/16)*e);
                   H(i+1,j-1)=H(i+1,j-1)+((1/16)*e);
                   H(i+1,j)=H(i+1,j)+((5/16)*e);
                   H(i+1,j+1)=H(i+1,j+1)+((3/16)*e);
               end     
            end
        else
            for j=col:-1:1
               if H(i,j)>127
                    b(i,j)=255;
               else
                    b(i,j)=0;
               end
               e = H(i,j)-b(i,j);
               if j~=1
                   H(i,j-1)=H(i,j-1)+((7/16)*e);
               end     
            end
        end
    end
end
Y_halftoned=b;

RGB2CMY_halftoned(:,:,1)=C_halftoned;
RGB2CMY_halftoned(:,:,2)=M_halftoned;
RGB2CMY_halftoned(:,:,3)=Y_halftoned;

figure(2);
imshow(RGB2CMY_halftoned);
title('Seperable Error Diffusion - CMY');

RGB2CMY_halftoned1(:,:,1)=255*ones(480,640) - (double(C_halftoned));
figure(3);
imshow(RGB2CMY_halftoned1(:,:,1));
title('Seperable Error Diffusion - Red plane');

RGB2CMY_halftoned1(:,:,2)=255*ones(480,640) - (double(M_halftoned));
figure(4);
imshow(RGB2CMY_halftoned1(:,:,2));
title('Seperable Error Diffusion - Green plane');

RGB2CMY_halftoned1(:,:,3)=255*ones(480,640) - (double(Y_halftoned));
figure(5);
imshow(RGB2CMY_halftoned1(:,:,3));
title('Seperable Error Diffusion - Blue plane');

figure(6)
imshow(RGB2CMY_halftoned1);
title('Seperable Error Diffusion');

% Part 2 %
% MBVQ %

v=zeros(375,500,3);
[rows, col]=size(R);
H_Red=double(R);
H_Green=double(G);
H_Blue=double(B);
for i=1:rows    
    if mod(i,2)~=0
      if i~=rows
       for j=1:col              
           [mbvq,v]=MBVQ(double(R(i,j)),double(G(i,j)),double(B(i,j)),H_Red(i,j),H_Green(i,j),H_Blue(i,j));
            b_Red(i,j)=v(1,1,1);
            b_Green(i,j)=v(1,1,2);
            b_Blue(i,j)=v(1,1,3);
            e_Red=H_Red(i,j)-b_Red(i,j);
            e_Green=H_Green(i,j)-b_Green(i,j);
            e_Blue=H_Blue(i,j)-b_Blue(i,j);
           if j==1 
               H_Red(i,j+1)=H_Red(i,j+1)+((7/16)*e_Red);
               H_Red(i+1,j)=H_Red(i+1,j)+((5/16)*e);
               H_Red(i+1,j+1)=H_Red(i+1,j+1)+((1/16)*e_Red);
               H_Green(i,j+1)=H_Green(i,j+1)+((7/16)*e_Green);
               H_Green(i+1,j)=H_Green(i+1,j)+((5/16)*e_Green);
               H_Green(i+1,j+1)=H_Green(i+1,j+1)+((1/16)*e_Green);
               H_Blue(i,j+1)=H_Blue(i,j+1)+((7/16)*e_Blue);
               H_Blue(i+1,j)=H_Blue(i+1,j)+((5/16)*e_Blue);
               H_Blue(i+1,j+1)=H_Blue(i+1,j+1)+((1/16)*e_Blue);               
           elseif j==col              
               H_Red(i+1,j-1)=H_Red(i+1,j-1)+((3/16)*e_Red);
               H_Red(i+1,j)=H_Red(i+1,j)+((5/16)*e_Red);
               H_Green(i+1,j-1)=H_Green(i+1,j-1)+((3/16)*e_Green);
               H_Green(i+1,j)=H_Green(i+1,j)+((5/16)*e_Green);
               H_Blue(i+1,j-1)=H_Blue(i+1,j-1)+((3/16)*e_Blue);
               H_Blue(i+1,j)=H_Blue(i+1,j)+((5/16)*e_Blue);
           else 
               H_Red(i,j+1)=H_Red(i,j+1)+((7/16)*e_Red);
               H_Red(i+1,j-1)=H_Red(i+1,j-1)+((3/16)*e_Red);
               H_Red(i+1,j)=H_Red(i+1,j)+((5/16)*e_Red);
               H_Red(i+1,j+1)=H_Red(i+1,j+1)+((1/16)*e_Red);
               H_Green(i,j+1)=H_Green(i,j+1)+((7/16)*e_Green);
               H_Green(i+1,j-1)=H_Green(i+1,j-1)+((3/16)*e_Green);
               H_Green(i+1,j)=H_Green(i+1,j)+((5/16)*e_Green);
               H_Green(i+1,j+1)=H_Green(i+1,j+1)+((1/16)*e_Green);
               H_Blue(i,j+1)=H_Blue(i,j+1)+((7/16)*e_Blue);
               H_Blue(i+1,j-1)=H_Blue(i+1,j-1)+((3/16)*e_Blue);
               H_Blue(i+1,j)=H_Blue(i+1,j)+((5/16)*e_Blue);
               H_Blue(i+1,j+1)=H_Blue(i+1,j+1)+((1/16)*e_Blue) ;
           end 
       end
      else
         for j=1:col
           [mbvq,v]=MBVQ(double(R(i,j)),double(G(i,j)),double(B(i,j)),H_Red(i,j),H_Green(i,j),H_Blue(i,j));
            b_Red(i,j)=v(1,1,1);
            b_Green(i,j)=v(1,1,2);
            b_Blue(i,j)=v(1,1,3);
            e_Red=H_Red(i,j)-b_Red(i,j);
            e_Green=H_Green(i,j)-b_Green(i,j);
            e_Blue=H_Blue(i,j)-b_Blue(i,j);
               if j~=col
                   H_Red(i,j+1)=H_Red(i,j+1)+((7/16)*e_Red);
                   H_Green(i,j+1)=H_Green(i,j+1)+((7/16)*e_Green);
                   H_Blue(i,j+1)=H_Blue(i,j+1)+((7/16)*e_Blue);
               end
         end
      end
    else
        if i~=rows
            for j=col:-1:1
                  [mbvq,v]=MBVQ(double(R(i,j)),double(G(i,j)),double(B(i,j)),H_Red(i,j),H_Green(i,j),H_Blue(i,j)); 
                   b_Red(i,j)=v(1,1,1);
                   b_Green(i,j)=v(1,1,2);
                   b_Blue(i,j)=v(1,1,3);
                   e_Red=H_Red(i,j)-b_Red(i,j);
                   e_Green=H_Green(i,j)-b_Green(i,j);
                   e_Blue=H_Blue(i,j)-b_Blue(i,j);
               if j==col
                   H_Red(i,j-1)=H_Red(i,j-1)+((7/16)*e_Red);
                   H_Red(i+1,j-1)=H_Red(i+1,j-1)+((1/16)*e_Red);
                   H_Red(i+1,j)=H_Red(i+1,j)+((5/16)*e_Red);
                   H_Green(i,j-1)=H_Green(i,j-1)+((7/16)*e_Green);
                   H_Green(i+1,j-1)=H_Green(i+1,j-1)+((1/16)*e_Green);
                   H_Green(i+1,j)=H_Green(i+1,j)+((5/16)*e_Green);
                   H_Blue(i,j-1)=H_Blue(i,j-1)+((7/16)*e_Blue);
                   H_Blue(i+1,j-1)=H_Blue(i+1,j-1)+((1/16)*e_Blue);
                   H_Blue(i+1,j)=H_Blue(i+1,j)+((5/16)*e_Blue);
               elseif j==1
                   H_Red(i+1,j)=H_Red(i+1,j)+((5/16)*e_Red);
                   H_Red(i+1,j+1)=H_Red(i+1,j+1)+((3/16)*e_Red);
                   H_Green(i+1,j)=H_Green(i+1,j)+((5/16)*e_Green);
                   H_Green(i+1,j+1)=H_Green(i+1,j+1)+((3/16)*e_Green);
                   H_Blue(i+1,j)=H_Blue(i+1,j)+((5/16)*e_Blue);
                   H_Blue(i+1,j+1)=H_Blue(i+1,j+1)+((3/16)*e_Blue);
               else
                   H_Red(i,j-1)=H_Red(i,j-1)+((7/16)*e_Red);
                   H_Red(i+1,j-1)=H_Red(i+1,j-1)+((1/16)*e_Red);
                   H_Red(i+1,j)=H_Red(i+1,j)+((5/16)*e_Red);
                   H_Red(i+1,j+1)=H_Red(i+1,j+1)+((3/16)*e_Red);
                   H_Green(i,j-1)=H_Green(i,j-1)+((7/16)*e_Green);
                   H_Green(i+1,j-1)=H_Green(i+1,j-1)+((1/16)*e_Green);
                   H_Green(i+1,j)=H_Green(i+1,j)+((5/16)*e_Green);
                   H_Green(i+1,j+1)=H_Green(i+1,j+1)+((3/16)*e_Green);
                   H_Blue(i,j-1)=H_Blue(i,j-1)+((7/16)*e_Blue);
                   H_Blue(i+1,j-1)=H_Blue(i+1,j-1)+((1/16)*e_Blue);
                   H_Blue(i+1,j)=H_Blue(i+1,j)+((5/16)*e_Blue);
                   H_Blue(i+1,j+1)=H_Blue(i+1,j+1)+((3/16)*e_Blue);
               end  
            end
        else
            for j=col:-1:1
                [mbvq,v]=MBVQ(double(R(i,j)),double(G(i,j)),double(B(i,j)),H_Red(i,j),H_Green(i,j),H_Blue(i,j));
                b_Red(i,j)=v(1,1,1);
                b_Green(i,j)=v(1,1,2);
                b_Blue(i,j)=v(1,1,3);
                e_Red=H_Red(i,j)-b_Red(i,j);
                e_Green=H_Green(i,j)-b_Green(i,j);
                e_Blue=H_Blue(i,j)-b_Blue(i,j); 
               if j~=1
                   H_Red(i,j-1)=H_Red(i,j-1)+((7/16)*e_Red);
                   H_Green(i,j-1)=H_Green(i,j-1)+((7/16)*e_Green);
                   H_Blue(i,j-1)=H_Blue(i,j-1)+((7/16)*e_Blue);
               end  
            end
        end
    end
end

mbvq_rgb(:,:,1)=b_Red;
figure(7);
imshow(mbvq_rgb(:,:,1));
title('MBVQ - Red plane');

mbvq_rgb(:,:,2)=b_Green;
figure(8);
imshow(mbvq_rgb(:,:,2));
title('MBVQ - Green plane');

mbvq_rgb(:,:,3)=b_Blue;
figure(9);
imshow(mbvq_rgb(:,:,3));
title('MBVQ - Blue plane');

figure(10);
imshow(mbvq_rgb);
title('MBVQ');
