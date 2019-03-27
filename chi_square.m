function [res] = chi_square(X,Y)
%CHI_SQU 此处显示有关此函数的摘要
%   此处显示详细说明
EX = sum(sum(X))/size(X,2);
EY = sum(sum(Y))/size(Y,2);
res = (sum(((X-EX)/EX).^2)+sum(((Y-EY)/EY).^2))^(1/2);
end


