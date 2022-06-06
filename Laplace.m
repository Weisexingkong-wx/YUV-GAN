%Laplace 
 N1 = 140; 
 A = zeros(1,N1); 
 X = zeros(1,N1);
 tic
 for L=1: N1
 I=imread([int2str(L),'.png']); 
 I=double(I); 
 [M N]=size(I); 
 FI=0; 
 for x=2:M-1 
     for y=2:N-1 
         IXXIYY = -4*I(x,y)+I(x,y+1)+I(x,y-1)+I(x+1,y)+I(x-1,y); 
             FI=FI+IXXIYY*IXXIYY;        %取各像素点梯度的平方和作为清晰度值    
     end 
 end  
A(1,L) = FI; 
 end 
time=toc
CalAve = sum(A,2)/N1;
 for W = 1:N1 
     C = max(A); 
     D = min(A); 
     E = C-D; 
     R = (A(1,W) - D)/(E); 
     X(1,W) = R; 
 end 
 
x1=[-20 -10 0 10 20 ]; 
y1 = [X(1,1) X(1,2) X(1,3) X(1,4) X(1,5)];
[p,S]=polyfit(x1,y1,2); 
Y=polyconf(p,x1,y1); 
plot(x1,y1,'m'); 
hold off;