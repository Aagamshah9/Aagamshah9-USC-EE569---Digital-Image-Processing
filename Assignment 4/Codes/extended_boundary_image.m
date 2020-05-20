function J = extended_boundary_image(Input_image,N)
mid_image = double(Input_image);
I = mid_image;
I = double(mid_image - (sum(sum(mid_image))/(N*N)));
J = zeros(N+4,N+4);
J(3:N+2,3:N+2)=I;        % Center of an image
J(2,3:N+2)=I(2,:);     % 2nd Upper Row
J(1,3:N+2)=I(3,:);     % 1st Upper Row %
J(N+3,3:N+2)=I(N-1,:); % 2nd Lower Row
J(N+4,3:N+2)=I(N-2,:); % 1st Lower Row %
J(:,2)=J(:,4);         % 2nd Leftmost Column
J(:,1)=J(:,5);         % 1st Leftmost Column %
J(:,N+3)=J(:,N+1);     % 2nd Rightmost Column 
J(:,N+4)=J(:,N);       % 1st Rightmost Column %
end