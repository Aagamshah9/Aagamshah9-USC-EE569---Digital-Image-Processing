function K=extended_boundary_r_map(r_map,row,col,n)
mid_image=double(r_map);
I=mid_image;
K=zeros(row+(n-1),col+(n-1));
K(((n-1)*0.5)+1:((n-1)*0.5)+row,((n-1)*0.5)+1:((n-1)*0.5)+col)=I;
for i = 1:((n-1)*0.5)                                                      % Row Extension 
    K(i,((n-1)*0.5)+1:((n-1)*0.5)+col)=I(((n-1)*0.5)+2-i,1:col);           % Upper rows
    K(((n-1)*0.5)+row+i,((n-1)*0.5)+1:((n-1)*0.5)+col)=I(row-i,1:col);     % Lower rows
end
for j = 1:((n-1)*0.5)                                                      % Column Extension
    K(:,j)=K(:,(n-1)+2-j);                                                 % Leftmost Columns
    K(:,((n-1)*0.5)+col+j)=K(:,col+((n-1)*0.5)-j);                         % Rightmost Columns
end
end