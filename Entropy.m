
%entropy
N1 = 140;           %N1为要处理的图片张数
A = zeros(1,N1);   %zeros()定义指定行列的零矩阵；A矩阵用来存储每一幅图像的清晰度原值
X = zeros(1,N1);   %X用来存储做归一化处理后的函数值
CalAve = 0; 
%处理图片
tic
for L=1: N1        
 I=imread([int2str(L),'.png']); %读取图片，将值转换为字符串接受向量和矩阵输入
 I=rgb2gray(I);
 I=double(I); 
 A(1,L)=entr(I);    %调用求熵值函数
 CalAve = CalAve + entr(I);
end
time=toc
CalAve = CalAve/N1;
%对原始数据做归一化处理，线性函数归一化公式
 for W = 1:N1 
   C = max(A); 
   D = min(A); 
   E = C-D; 
   R = (A(1,W) - D)/(E); 
   X(1,W) = R; 
  end 
x1=[-20 -10 0 10 20 ]; 
y1 = [X(1,1) X(1,2) X(1,3) X(1,4) X(1,5)]; 
[p,S]=polyfit(x1,y1,2);   %polyfit(x,y,n)曲线拟合函数，已知离散点坐标拟合曲线；x,y横纵坐标，n为拟合阶数，一阶直线拟合，二阶抛物线拟合 ，返回幂次从高到低的多项式系数向量P，矩阵S用于生成预测值的误差估计
Y=polyconf(p,x1,y1); %置信区间
plot(x1,y1,'r');     %画出拟合曲线，红线red
title('基于信息熵的评价函数');
xlabel('成像面位置');
ylabel('归一化后的图像清晰度评价值');
