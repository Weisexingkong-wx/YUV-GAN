%Variance
 N1 = 64; 
 A = zeros(1,N1); 
 X = zeros(1,N1); 
tic
 for L=1: N1
 I=imread([int2str(L),'.png']);  
 I=double(I); 
 [M N]=size(I);  
 gama = 0;   %gama图像平均灰度值
 %求gama
 for x=1:M 
     for y=1:N 
         gama = gama + I(x,y); 
     end 
 end 
 gama = gama/(M*N); 
  
 FI=0; 
 for x=1:M 
     for y=1:N 
         FI=FI+(I(x,y)-gama)*(I(x,y)-gama); 
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
 plot(x1,y1,'b');
 hold on;