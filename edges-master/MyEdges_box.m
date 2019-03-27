close all;
clear;
root_dir = '/home/liuyuming/SWINT/数据/武汉180605/L/';
save_root =  '/home/liuyuming/SWINT/数据/武汉180605/L_edgeboximg/';
edgbox_root =  '/home/liuyuming/SWINT/数据/武汉180605/L_edgeboxlabel/';
fileList=dir(root_dir);
filenumber = length(fileList);
%% load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
% opts.alpha = .65;     % step size of sliding window search
% opts.beta  = .75     % nms threshold for object proposals
% opts.minScore = .01;  % min score of boxes to detect
% opts.maxBoxes = 1e4;  % max number of boxes to detect
opts.alpha = .85;     % step size of sliding window search
opts.beta  = .75;  % nms threshold for object proposals
opts.minScore = .1;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect

%% detect Edge Box bounding box proposals (see edgeBoxes.m)
for k = 1:filenumber
    if strcmp(fileList(k).name,'.')==1||strcmp(fileList(k).name,'..')==1
        continue;
    end
    full_name=[root_dir,fileList(k).name];
    I = imread(full_name);
    labelfile = [edgbox_root,fileList(k).name];
    %fid=fopen(labelfile,'w');
    %I=I(1:1090,:);
    sumcol = sum(I,1);
    Mid = int32(sum(find(sumcol==295800),2)/size(find(sumcol==295800),2));
    I1 = zeros([size(I,1),size(I,2),3]);
    I1=im2uint8(I1);
    I1(:,:,1)=I;
    I1(:,:,2)=I;
    I1(:,:,3)=I;
    tic, bbs=edgeBoxes(I1,model,opts); toc
    imshow(I1)
    res=[];
    for i =1:size(bbs,1)
        x = bbs(i,1);
        y = bbs(i,2);
        w = bbs(i,3);
        h = bbs(i,4);
        score = bbs(i,5);
%         if x <127 || x+w>670 || y<1 || w<10 || w>200|| h<10 ||h>200  || w*h<1200
%             continue;
%         end
        res(end+1,:)=bbs(i,:);
        rectangle('position',bbs(i,1:4),'edgecolor','r');
    end
    pick=NMS(res,0.75);
    for i =1:size(pick,1)
        x = res(pick(i),1);
        y = res(pick(i),2);
        w = res(pick(i),3);
        h = res(pick(i),4);
        score = res(pick(i),5);
        
%         if x <100 || x>750 || y<1 || w<10 || w>200|| h<10 ||h>200 || score < 0.1
%             continue;
%         end
%         rectangle('position',res(pick(i),1:4),'edgecolor','r');
% 
% %         img1 = I(y:y+h,x:x+w);
%         if Mid>=x
%             img2 = I(y:y+h,Mid - (x - Mid) - w:Mid - (x - Mid));
%         else
%             img2 = I(y:y+h,Mid + (Mid - x) - w:Mid + (Mid - x));
%         end
%         img2 = fliplr(img2);
%         [feature1,vision] = extractHOGFeatures(double(img1),'CellSize',[8 8]);
%         [feature2,vision] = extractHOGFeatures(double(img2),'CellSize',[8 8]);
%         hogchidis = distMATChiSquare(feature1,feature2)/w/h
%         if(hogchidis>0.02)
%            rectangle('position',res(pick(i),1:4),'edgecolor','r');
%         end
    end
   
    save_name=[save_root,'\',fileList(k).name];
%     saveas(gcf,save_name)
end

