
%Tenengrad
 N1 =140; 
 A = zeros(1,N1); 
 X = zeros(1,N1); 
 tic
 for L=1: N1 
 I=imread([int2str(L),'.png']); 
 I=double(I); dengw
 [M N]=size(I); 
 %����sobel����gx,gy��ͼ�����������ȡͼ��ˮƽ����ʹ�ֱ������ݶ�ֵ
GX = 0;   %ͼ��ˮƽ�����ݶ�ֵ
GY = 0;   %ͼ��ֱ�����ݶ�ֵ
FI = 0;   %��������ʱ�洢ͼ��������ֵ
T  = 0;   %���õ���ֵ
 for x=2:M-1 
     for y=2:N-1 
         GX = I(x-1,y+1)+2*I(x,y+1)+I(x+1,y+1)-I(x-1,y-1)-2*I(x,y-1)-I(x+1,y-1); 
         GY = I(x+1,y-1)+2*I(x+1,y)+I(x+1,y+1)-I(x-1,y-1)-2*I(x-1,y)-I(x-1,y+1); 
         SXY= sqrt(GX*GX+GY*GY); %ĳһ����ݶ�ֵ
         %ĳһ���ص��ݶ�ֵ�����趨����ֵ���������ص㿼�ǣ���������Ӱ��
         if SXY>T 
           FI = FI + SXY*SXY;    %Tenengradֵ����
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

