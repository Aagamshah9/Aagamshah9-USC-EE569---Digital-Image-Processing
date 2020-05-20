function [mbvq,v]=MBVQ(R,G,B,H_Red,H_Green,H_Blue)
if( (R+G)>255)
           if((G+B> 255))
               if((R+G+B)> 510)
                   mbvq= 'CMYW';
               else
                   mbvq= 'MYGC';
               end
            else
                   mbvq= 'RGMY';
           end
       else
           if(~((G+B)>255))
               if(~((R+G+B)>255))
                   mbvq= 'KRGB';
               else
                   mbvq= 'RGBM';
               end
           else
               mbvq= 'CMGB';
           end
end

vertex=getNearestVertex(mbvq,H_Red/255,H_Green/255,H_Blue/255);
       if isequal(vertex,'white')
           v(1,1,1:3)=255;       
       elseif isequal(vertex,'cyan')
           v(1,1,1)=0;v(1,1,2)=255;v(1,1,3)=255; %v(1,1,1)=255-Red;v(1,1,2)=0;v(1,1,3)=0;
       elseif isequal(vertex,'magenta')
           v(1,1,1)=255;v(1,1,2)=0;v(1,1,3)=255; %v(1,1,1)=0;v(1,1,2)=255-Green;v(1,1,3)=0;
       elseif isequal(vertex,'yellow')
           v(1,1,1)=255;v(1,1,2)=255;v(1,1,3)=0; %v(1,1,1)=0;v(1,1,2)=0;v(1,1,3)=255-Blue;
       elseif isequal(vertex,'red')  
           v(1,1,1)=255;v(1,1,2)=0;v(1,1,3)=0;
       elseif isequal(vertex,'green')
           v(1,1,1)=0;v(1,1,2)=255;v(1,1,3)=0;
       elseif isequal(vertex,'blue')
           v(1,1,1)=0;v(1,1,2)=0;v(1,1,3)=255;
       else
           v(1,1,1)=0;v(1,1,2)=0;v(1,1,3)=0; %black
       end
end