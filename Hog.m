close all;
clear;
%im = imread('C:\Users\lyming\Desktop\anomalydata\anomalydata\clip1\KM1983+822M.jpg');
im = imread('C:\Users\lyming\Desktop\anomalydata\anomalydata\zoo1\animal1111.jpg');
resmat = zeros(size(im));
Middlex = 399;
sizex = 128;
sizey = 128;
Hog_CellSizex = 32; 
Hog_CellSizey = 32; 
stepsizex = 32;
stepsizey = 32;
figure;
imshow(im);

for x = Middlex+sizex/2:stepsizex:750-sizex
    for y = 1+sizey/2:stepsizey:1160-sizey
        
%         if(y<150||(y>365&&y<524)||(y>744&&y<902)||(y>1120))
%             continue
%         end
    y1 = y;
    x1 = Middlex - (x-Middlex);
    
    im2 = im(y1-sizey/2:y1+sizey/2,x1-sizex/2:x1+sizex/2);
    im2 = flip(im2,2);
    [featureVector1,hogVisualization1] = extractHOGFeatures(im2,'CellSize',[Hog_CellSizex Hog_CellSizey]);
%     if(y1>830 && x1 <350)
%         figure;
%         imshow(im2);
%         hold on;
%         plot(hogVisualization1);
%     end
    y2=y;
    x2=x;
    im3 = im(y2-sizey/2:y2+sizey/2,x2-sizex/2:x2+sizex/2);
    
    
    [featureVector2,hogVisualization2] = extractHOGFeatures(im3,'CellSize',[Hog_CellSizex Hog_CellSizey]);
%     if(y1>830 && x1 < 350)
%        figure;
%        imshow(im3);
%        hold on;
%        plot(hogVisualization2);
%     end
    D = distMATChiSquare(featureVector1,featureVector2);
    if D >3
        D
%        close all;
%        figure;
%        imshow(im2);
%        hold on;
%        plot(hogVisualization1);
%       
%        figure;
%        imshow(im3);
%        hold on;
%        plot(hogVisualization2);
       
       rectangle('Position', [x1-sizex/2,y1-sizey/2,sizex,sizey],'EdgeColor','r','LineWidth', 4)
       rectangle('Position', [x2-sizex/2,y2-sizey/2,sizex,sizey],'EdgeColor','r','LineWidth', 4)
       hold on;
       a=0;
    end
    end
end
%resmat = mat2gray(imfilter(resmat, fspecial('gaussian', [4, 4], 1)));
%resmat(resmat<0.6)=0;
%imshow(resmat)