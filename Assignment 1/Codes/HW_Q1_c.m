M = readraw1('Toy.raw',560,400);
imshow(M);
N=zeros(256,1);
O=zeros(256,1);
P=zeros(256,1);
R=M(:,:,1);
%disp(R);
G=M(:,:,2);
%disp(G);
B=M(:,:,3);
%disp(B);
r=560;
c=400;
for i=1:r
    for j=1:c
        A=R(i,j)+1;
        N(A,1)=N(A,1)+1;
        C=G(i,j)+1;
        O(C,1)=O(C,1)+1;
        D=B(i,j)+1;
        P(D,1)=P(D,1)+1;
    end
end
subplot(3,2,1)
bar(N);
subplot(3,2,2)
plot(N);
subplot(3,2,3)
bar(O);
subplot(3,2,4)
plot(O);
subplot(3,2,5)
bar(P);
subplot(3,2,6)
plot(P);