
function[H_img]= entr(I)   
[C,R]=size(I); %��ͼ��Ĺ��
Img_size=C*R; %ͼ�����ص���ܸ���
L=256; %ͼ��ĻҶȼ�0-255
H_img=0;  %ͼ����
nk=zeros(L,1); %�洢ͼ��Ҷȳ��ִ���
for i=1:C
for j=1:R
Img_level=I(i,j)+1; %��ȡͼ��ĻҶȼ�
nk(Img_level)=nk(Img_level)+1; %ͳ��ÿ���Ҷȼ����صĵ���
end
end
for k=1:L
Ps(k)=nk(k)/Img_size; %����ÿһ���Ҷȼ����ص���ռ�ĸ���
if Ps(k)~=0 %ȥ������Ϊ0�����ص�
H_img=-Ps(k)*log2(Ps(k))+H_img; %����ֵ�Ĺ�ʽ
end
end
end

