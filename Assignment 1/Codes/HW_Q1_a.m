C = readraw('Dog.raw',600,532); 
[X,Y,Z]=size(C);
D=zeros(532,600,3);

% Bilinear interpolation %
%------------------------%

% Bayer CFA Pattern considered %
%------------------------------%
% G R G R G R G R G R G R G........
% B G B G B G B G B G B G B........
% G R G R G R G R G R G R G........
% B G B G B G B G B G B G B........
% G R G R G R G R G R G R G........
% . . . . . . . . . . . . .
% . . . . . . . . . . . . .
% . . . . . . . . . . . . . 
%-------------------------%
% Sensor Alignment %
%------------------%
%       G R        %
%       B G        %
%------------------%
%    Pattern 1     %    Pattern 2   %   Pattern 3   %   Pattern 4   %     
%                  %                %               %               %
%      G R G       %      R G R     %     B G B     %     G B G     %
%      B G B       %      G B G     %     G R G     %     R G R     %
%      G R G       %      R G R     %     B G B     %     G B G     %

Height=532;
Width=600;
i = Height-1;
j = Width-1;

for r = 2:i
    for c = 2:j
        if (mod(r,2)==0 && mod(c,2)==0)     % Pattern 1 
            G1=C(r,c);
            %disp(G1);
            Red11=C(r-1,c);
            %disp(Red11);
            Red12=C(r+1,c);
            %disp(Red12);
            R1=0.5*(double(Red11)+double(Red12));
            %disp(R1);
            Blue11=C(r,c-1);
            %disp(Blue11);
            Blue12=C(r,c+1);
            %disp(Blue12);
            B1=0.5*(double(Blue11)+double(Blue12));
            %disp(B1);  
            D(r,c,1)=R1;
            D(r,c,2)=G1;
            D(r,c,3)=B1;
        elseif (mod(r,2)==0 && mod(c,2)==1) % Pattern 2
            B2=C(r,c);
            %disp(B2);
            Red21=C(r-1,c-1);
            %disp(Red21);
            Red22=C(r-1,c+1);
            %disp(Red22);
            Red23=C(r+1,c+1);
            %disp(Red23);
            Red24=C(r+1,c-1);
            %disp(Red24);
            R2=0.25*(double(Red21)+double(Red22)+double(Red23)+double(Red24));
            %disp(R2);
            Green21=C(r,c+1);
            %disp(Green21);
            Green22=C(r,c-1);
            %disp(Green22);
            Green23=C(r-1,c);
            %disp(Green23);
            Green24=C(r+1,c);
            %disp(Green24);
            G2=0.25*(double(Green21)+double(Green22)+double(Green23)+double(Green24));
            %disp(G2);
            D(r,c,1)=R2;
            D(r,c,2)=G2;
            D(r,c,3)=B2;
        elseif (mod(r,2)==1 && mod(c,2)==0) % Pattern 3
            R3=C(r,c);
            %disp(R3);
            Blue31=C(r-1,c-1);
            %disp(Blue31);
            Blue32=C(r-1,c+1);
            %disp(Blue32);
            Blue33=C(r+1,c+1);
            %disp(Blue33);
            Blue34=C(r+1,c-1);
            %disp(Blue34);
            B3=0.25*(double(Blue31)+double(Blue32)+double(Blue33)+double(Blue34));
            %disp(B3);
            Green31=C(r,c+1);
            %disp(Green31);
            Green32=C(r,c-1);
            %disp(Green32);
            Green33=C(r-1,c);
            %disp(Green33);
            Green34=C(r+1,c);
            %disp(Green34);
            G3=0.25*(double(Green31)+double(Green32)+double(Green33)+double(Green34));
            %disp(G3);
            D(r,c,1)=R3;
            D(r,c,2)=G3;
            D(r,c,3)=B3;
        else (mod(r,2)==1 && mod(c,2)==1);  % Pattern 4
            G4=C(r,c);
            %disp(G4);
            Blue41=C(r-1,c);
            %disp(Blue41);
            Blue42=C(r+1,c);
            %disp(Blue42);
            B4=0.5*(double(Blue41)+double(Blue42));
            %disp(B4);
            Red41=C(r,c-1);
            %disp(Red41);
            Red42=C(r,c+1);
            %disp(Red42);
            R4=0.5*(double(Red41)+double(Red42));
            %disp(R4);
            D(r,c,1)=R4;
            D(r,c,2)=G4;
            D(r,c,3)=B4;
        end
    end
end
subplot(1,2,1)
imshow(C);
subplot(1,2,2)
imshow(uint8(D));
count = writeraw(D,'Dog_Bilinear.raw');