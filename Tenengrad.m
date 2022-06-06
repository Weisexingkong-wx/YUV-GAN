
%Tenengrad
 N1 =140; 
 A = zeros(1,N1); 
 X = zeros(1,N1); 
 tic
 for L=1: N1 
 I=imread([int2str(L),'.png']); 
 I=double(I); dengw
 [M N]=size(I); 
 %利用sobel算子gx,gy与图像做卷积，提取图像水平方向和垂直方向的梯度值
GX = 0;   %图像水平方向梯度值
GY = 0;   %图像垂直方向梯度值
FI = 0;   %变量，暂时存储图像清晰度值
T  = 0;   %设置的阈值
 for x=2:M-1 
     for y=2:N-1 
         GX = I(x-1,y+1)+2*I(x,y+1)+I(x+1,y+1)-I(x-1,y-1)-2*I(x,y-1)-I(x+1,y-1); 
         GY = I(x+1,y-1)+2*I(x+1,y)+I(x+1,y+1)-I(x-1,y-1)-2*I(x-1,y)-I(x-1,y+1); 
         SXY= sqrt(GX*GX+GY*GY); %某一点的梯度值
         %某一像素点梯度值大于设定的阈值，将该像素点考虑，消除噪声影响
         if SXY>T 
           FI = FI + SXY*SXY;    %Tenengrad值定义
         end 
     end 
 end 
 A(1,L) = FI; 
 end 
 time=toc
 
% X = zeros(1,N1); 
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
plot(x1,y1,'g'); 

