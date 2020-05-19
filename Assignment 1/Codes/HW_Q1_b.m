C = readraw('Dog.raw',600,532); 
[X,Y,Z]=size(C);
E=zeros(532,600,3);

% Malvar-He-Cutler algorithm %
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
%        G         %        B       %       R       %       G       %
%      G R G       %      R G R     %     B G B     %     G B G     %
%    G B G B G     %    B G B G B   %   R G R G R   %   G R G R G   %
%      G R G       %      R G R     %     B G B     %     G B G     %
%        G         %        B       %       R       %       G       %

Height=532;
Width=600;
i = Height-2;
j = Width-2;

for r = 3:i
    for c = 3:j
        if (mod(r,2)==0 && mod(c,2)==0)     % Pattern 1 
            G1=C(r,c);
            %disp(G1);
            Red11=C(r-1,c);
            %disp(Red11);
            Red12=C(r+1,c);
            %disp(Red12);
            Rbl1=0.5*(double(Red11)+double(Red12));
            %disp(Rbl1);
            Green11=C(r-2,c);
            %disp(Green11);
            Green12=C(r-1,c+1);
            %disp(Green12);
            Green13=C(r,c+2);
            %disp(Green13);
            Green14=C(r+1,c+1);
            %disp(Green14);
            Green15=C(r+2,c);
            %disp(Green15);
            Green16=C(r+1,c-1);
            %disp(Green16);
            Green17=C(r,c-2);
            %disp(Green17);
            Green18=C(r-1,c-1);
            %disp(Green18);
            Gdel1=G1-(0.125*(double(Green11)+double(Green12)+double(Green13)+double(Green14)+double(Green15)+double(Green16)+double(Green17)+double(Green18)));
            %disp(Gdel1);
            R1=(Rbl1+((5/8)*Gdel1));
            %disp(R1);
            Blue11=C(r,c-1);
            %disp(Blue11);
            Blue12=C(r,c+1);
            %disp(Blue12);
            Bbl1=0.5*(double(Blue11)+double(Blue12));
            %disp(Bbl1);
            B1=(Bbl1+((5/8)*Gdel1));
            %disp(B1);
            E(r,c,1)=R1;
            E(r,c,2)=G1;
            E(r,c,3)=B1;
        elseif (mod(r,2)==0 && mod(c,2)==1) % Pattern 2
            B2=C(r,c);
            %disp(B2);
            Red21=C(r-1,c-1);
            %disp(Red21);
            Red22=C(r-1,c+1);
            %disp(Red22);
            Red23=C(r+1,c-1);
            %disp(Red23);
            Red24=C(r+1,c+1);
            %disp(Red24);
            Rbl2=0.25*(double(Red21)+double(Red22)+double(Red23)+double(Red24));
            %disp(Rbl2);
            Blue21=C(r-2,c);
            %disp(Blue21);
            Blue22=C(r,c+2);
            %disp(Blue22);
            Blue23=C(r,c-2);
            %disp(Blue23);
            Blue24=C(r+2,c);
            %disp(Blue24);
            Bdel=B2-(0.25*(double(Blue21)+double(Blue22)+double(Blue23)+double(Blue24)));
            %disp(Bdel);
            R2=(Rbl2+((3/4)*Bdel));
            %disp(R2);
            Green21=C(r-1,c);
            %disp(Green21);
            Green22=C(r,c+1);
            %disp(Green22);
            Green23=C(r+1,c);
            %disp(Green23);
            Green24=C(r,c-1);
            %disp(Green24);
            Gbl2=0.25*(double(Green21)+double(Green22)+double(Green23)+double(Green24));
            %disp(Gbl2);
            G2=(Gbl2+((3/4)*Bdel));
            %disp(G2);
            E(r,c,1)=R2;
            E(r,c,2)=G2;
            E(r,c,3)=B2;
        elseif (mod(r,2)==1 && mod(c,2)==0) % Pattern 3
            R3=C(r,c);
            %disp(R3);
            Red31=C(r-2,c);
            %disp(Red31);
            Red32=C(r,c+2);
            %disp(Red32);
            Red33=C(r,c-2);
            %disp(Red33);
            Red34=C(r+2,c);
            %disp(Red34);
            Rdel=R3-(0.25*(double(Red31)+double(Red32)+double(Red33)+double(Red34)));
            %disp(Rdel);
            Blue31=C(r-1,c-1);
            %disp(Blue31);
            Blue32=C(r-1,c+1);
            %disp(Blue32);
            Blue33=C(r+1,c+1);
            %disp(Blue33);
            Blue34=C(r+1,c-1);
            %disp(Blue34);
            Bbl3=0.25*(double(Blue31)+double(Blue32)+double(Blue33)+double(Blue34));
            %disp(Bbl3);
            B3=(Bbl3+((3/4)*Rdel));
            %disp(B3);
            Green31=C(r-1,c);
            %disp(Green31);
            Green32=C(r,c+1);
            %disp(Green32);
            Green33=C(r+1,c);
            %disp(Green33);
            Green34=C(r,c-1);
            %disp(Green34);
            Gbl3=0.25*(double(Green31)+double(Green32)+double(Green33)+double(Green34));
            %disp(Gbl3);
            G3=(Gbl3+((0.5)*Rdel));
            %disp(G3);
            E(r,c,1)=R3;
            E(r,c,2)=G3;
            E(r,c,3)=B3;
        else (mod(r,2)==1 && mod(c,2)==1);  % Pattern 4
            G4=C(r,c);
            %disp(G4);
            Green41=C(r-2,c);
            %disp(Green41);
            Green42=C(r-1,c+1);
            %disp(Green42);
            Green43=C(r,c+2);
            %disp(Green43);
            Green44=C(r+1,c+1);
            %disp(Green44);
            Green45=C(r+2,c);
            %disp(Green45);
            Green46=C(r+1,c-1);
            %disp(Green46);
            Green47=C(r,c-2);
            %disp(Green47);
            Green48=C(r-1,c-1);
            %disp(Green48);
            Gdel2=G4-(0.125*(double(Green41)+double(Green42)+double(Green43)+double(Green44)+double(Green45)+double(Green46)+double(Green47)+double(Green48)));
            %disp(Gdel2);
            Blue41=C(r-1,c);
            %disp(Blue41);
            Blue42=C(r+1,c);
            %disp(Blue42);
            Bbl4=0.5*(double(Blue41)+double(Blue42));
            %disp(Bbl4);
            B4=(Bbl4+((5/8)*Gdel2));
            %disp(B4);
            Red41=C(r,c-1);
            %disp(Red41);
            Red42=C(r,c+1);
            %disp(Red42);
            Rbl4=0.5*(double(Red41)+double(Red42));
            %disp(Rbl4);
            R4=(Rbl4+((5/8)*Gdel2));
            %disp(R4);
            E(r,c,1)=R4;
            E(r,c,2)=G4;
            E(r,c,3)=B4;
        end
    end
end
subplot(1,2,1)
imshow(C);
subplot(1,2,2)
imshow(uint8(E));
count = writeraw(E,'Dog_MHC.raw');