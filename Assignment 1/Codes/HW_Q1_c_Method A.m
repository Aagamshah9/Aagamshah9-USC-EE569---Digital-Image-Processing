Q = readraw1('Toy.raw',560,400);
%imshow(Q);

W = zeros(560,400,3);

S = zeros(256,1);
T = zeros(256,1);
X = zeros(256,1);

U = zeros(256,1);
V = zeros(256,1);
Y = zeros(256,1);

r=560;
c=400;

R=Q(:,:,1);
G=Q(:,:,2);
B=Q(:,:,3);

for i=1:r
    for j=1:c
        A=R(i,j)+1;
        S(A,1)=S(A,1)+1;
        C=G(i,j)+1;
        T(C,1)=T(C,1)+1;
        D=B(i,j)+1;
        X(D,1)=X(D,1)+1;
    end
end

S=S/224000;
%disp(S);

T=T/224000;
%disp(T);

X=X/224000;
%disp(X);

for k=2:256
      U(k,1) = S(k,1);
      U(1,1) = S(1,1);
      U(k,1) = U(k,1)+U(k-1,1);
      V(k,1) = T(k,1);
      V(1,1) = T(1,1);
      V(k,1) = V(k,1)+V(k-1,1);
      Y(k,1) = X(k,1);
      Y(1,1) = X(1,1);
      Y(k,1) = Y(k,1)+Y(k-1,1);
end

U = round(U*255);
V = round(V*255);
Y = round(Y*255);

for i = 1:r
    for j = 1:c
        x=R(i,j);
        R(i,j)=U(x+1,1);
        y=G(i,j);
        G(i,j)=V(y+1,1);
        z=B(i,j);
        B(i,j)=Y(z+1,1);
    end
end

W(:,:,1)=R;
W(:,:,2)=G;
W(:,:,3)=B;
figure(1)
subplot(3,4,1)
bar(S);
subplot(3,4,2)
plot(S);
subplot(3,4,3)
bar(T);
subplot(3,4,4)
plot(T);
subplot(3,4,5)
bar(X);
subplot(3,4,6)
plot(X);
subplot(3,4,7)
bar(U);
subplot(3,4,8)
plot(U);
subplot(3,4,9)
bar(V);
subplot(3,4,10)
plot(V);
subplot(3,4,11)
bar(Y);
subplot(3,4,12)
plot(Y);
figure(2)
subplot(1,2,1)
imshow(Q);
subplot(1,2,2)
imshow(uint8(W));
count = writeraw(W,'HEMethodA.raw');