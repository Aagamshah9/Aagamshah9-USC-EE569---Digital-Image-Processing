F = readraw('Corn_noisy.raw',320,320);
subplot(1,2,1)
X = readraw('Corn_gray.raw',320,320);
G=zeros(320,320);

%    Uniform Weight Function    %
%-------------------------------%
%                               %
%       1    1  1  1 ...        %
%      --- * 1  1  1 ...        %
%       N    1  1  1 ...        %                               
%            .  .  .            %
%            .  .  .            %
%                               %
%-------------------------------%

Height=320;
Width=320;
i=Height-1;
j=Width-1;

for r = 2:i
    for c = 2:j
        G(r,c)=((1/9)*(double(F(r-1,c-1))+double(F(r-1,c))+double(F(r-1,c+1))+double(F(r,c-1))+double(F(r,c))+double(F(r,c+1))+double(F(r+1,c-1))+double(F(r+1,c))+double(F(r+1,c+1))));
        %G(r,c)=((1/25)*(double(F(r-2,c-2))+double(F(r-2,c-1))+double(F(r-2,c))+double(F(r-2,c+1))+double(F(r-2,c+2))+double(F(r-1,c-2))+double(F(r-1,c-1))+double(F(r-1,c))+double(F(r-1,c+1))+double(F(r-1,c+2))+double(F(r,c-2))+double(F(r,c-1))+double(F(r,c))+double(F(r,c+1))+double(F(r,c+2))+double(F(r+1,c-2))+double(F(r+1,c-1))+double(F(r+1,c))+double(F(r+1,c+1))+double(F(r+1,c+2))+double(F(r+2,c-2))+double(F(r+2,c-1))+double(F(r+2,c))+double(F(r+2,c+1))+double(F(r+2,c+2))));    
    end
end
subplot(1,2,2)
imshow(uint8(G));
count = writeraw(G,'Corn_Uniform.raw');

