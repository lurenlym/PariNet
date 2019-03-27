function [res] = chi_square(X,Y)
%CHI_SQU �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
EX = sum(sum(X))/size(X,2);
EY = sum(sum(Y))/size(Y,2);
res = (sum(((X-EX)/EX).^2)+sum(((Y-EY)/EY).^2))^(1/2);
end


