
%entropy
N1 = 140;           %N1ΪҪ������ͼƬ����
A = zeros(1,N1);   %zeros()����ָ�����е������A���������洢ÿһ��ͼ���������ԭֵ
X = zeros(1,N1);   %X�����洢����һ��������ĺ���ֵ
CalAve = 0; 
%����ͼƬ
tic
for L=1: N1        
 I=imread([int2str(L),'.png']); %��ȡͼƬ����ֵת��Ϊ�ַ������������;�������
 I=rgb2gray(I);
 I=double(I); 
 A(1,L)=entr(I);    %��������ֵ����
 CalAve = CalAve + entr(I);
end
time=toc
CalAve = CalAve/N1;
%��ԭʼ��������һ�����������Ժ�����һ����ʽ
 for W = 1:N1 
   C = max(A); 
   D = min(A); 
   E = C-D; 
   R = (A(1,W) - D)/(E); 
   X(1,W) = R; 
  end 
x1=[-20 -10 0 10 20 ]; 
y1 = [X(1,1) X(1,2) X(1,3) X(1,4) X(1,5)]; 
[p,S]=polyfit(x1,y1,2);   %polyfit(x,y,n)������Ϻ�������֪��ɢ������������ߣ�x,y�������꣬nΪ��Ͻ�����һ��ֱ����ϣ�������������� �������ݴδӸߵ��͵Ķ���ʽϵ������P������S��������Ԥ��ֵ��������
Y=polyconf(p,x1,y1); %��������
plot(x1,y1,'r');     %����������ߣ�����red
title('������Ϣ�ص����ۺ���');
xlabel('������λ��');
ylabel('��һ�����ͼ������������ֵ');