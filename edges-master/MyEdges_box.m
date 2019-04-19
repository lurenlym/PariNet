close all;
clear;
rdir = 'E:\liuyuming\SiameseNet\DATA\WH180605\';
LR = 'L';
root_dir = [rdir,LR,'\','VAL\'];
if exist(root_dir,'dir')==0
   mkdir(root_dir);
end
save_root =  [rdir,LR,'\','edgeboximg\'];
if exist(save_root,'dir')==0
   mkdir(save_root);
end
edgbox_root =  [rdir,LR,'\','edgeboxlabel\'];
if exist(edgbox_root,'dir')==0
   mkdir(edgbox_root);
end
regionraw_root =  [rdir,LR,'\','regionraw\'];
if exist(regionraw_root,'dir')==0
   mkdir(regionraw_root);
end
regionture_root =  [rdir,LR,'\','regionture\'];
if exist(regionture_root,'dir')==0
   mkdir(regionture_root);
end
regionfalse_root =  [rdir,LR,'\','regionfalse\'];
if exist(regionfalse_root,'dir')==0
   mkdir(regionfalse_root);
end
imshow_root =  [rdir,LR,'\','imshow_root\'];
if exist(imshow_root,'dir')==0
   mkdir(imshow_root);
end
regionlist = [rdir,LR,'\','regionlistval.txt'];

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
opts.alpha = .7;     % step size of sliding window search
opts.beta  = 0.5;  % nms threshold for object proposals
opts.minScore = .09;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to d30petect
regionlistfid=fopen(regionlist,'w');
sumcount = 0;
%% detect Edge Box bounding box proposals (see edgeBoxes.m)
for k = 1:filenumber
    if strcmp(fileList(k).name,'.')==1||strcmp(fileList(k).name,'..')==1
        continue;
    end
    imgname = fileList(k).name;
    full_name=[root_dir,imgname]
    I = imread(full_name);
    
    labelfile = [edgbox_root,imgname(1:end-3),'txt'];
    fid=fopen(labelfile,'w');
    %I=I(1:1090,:);
    sumcol = sum(I,1);
    Mid = int32(sum(find(sumcol>290000),2)/size(find(sumcol>290000),2));
    I1 = zeros([size(I,1),size(I,2),3]);
    I1=im2uint8(I1);
    I1(:,:,1)=I;
    I1(:,:,2)=I;
    I1(:,:,3)=I;
    tic, bbs=edgeBoxes(I1,model,opts); toc
%     index = NMS(bbs,0.95);
%     bbs = bbs(index,:);
    imshow(I1)
    res=[];
    region_count=0;
    
    for i =1:size(bbs,1)
        x = int32(bbs(i,1));
        y = int32(bbs(i,2));
        w = int32(bbs(i,3));
        h = int32(bbs(i,4));
        score = bbs(i,5);
        if x <155 || x+w>645 || w<10 || w>200|| h<10 ||h>200  %|| w*h<1200 %|| score<0.1
            continue;
        end
        res(end+1,:)=bbs(i,:);
        rectangle('position',bbs(i,1:4),'edgecolor','r');

        fprintf(fid,'%d %d %d %d\n',[x,y,w,h]);
        img1 = I(y:y+h,x:x+w);
           if Mid>=x
            imgTure = I(y:y+h,Mid - (x - Mid) - w:Mid - (x - Mid));
        else
            imgTure = I(y:y+h,Mid + (Mid - x) - w:Mid + (Mid - x));
           end
        [sizex,sizey]=size(I);
        FalseX = int32((sizey-w-1)*rand(1))+1;
        FalseY = int32((sizex-h-1)*rand(1))+1;
        imgFalse = I(FalseY:FalseY+h,FalseX:FalseX+w);

        imgTure = fliplr(imgTure);
        regionname = [imgname(1:end-4),'_',num2str(region_count),'.','jpg'];
        imwrite(img1,[regionraw_root,regionname]);
        imwrite(imgTure,[regionture_root,regionname]);
        imwrite(imgFalse,[regionfalse_root,regionname]);
        imwrite([img1,imgTure,imgFalse],[imshow_root,regionname]);
        fprintf(regionlistfid,'%s\n',regionname);
        region_count = region_count+1;
        sumcount=sumcount+1;
    end
    fclose(fid);
%     pick=NMS(res,0.75);
%     for i =1:size(pick,1)
%         x = res(pick(i),1);
%         y = res(pick(i),2);
%         w = res(pick(i),3);
%         h = res(pick(i),4);
%         score = res(pick(i),5);
%         
% %         if x <100 || x>750 || y<1 || w<10 || w>200|| h<10 ||h>200 || score < 0.01
% %             continue;
% %         end
% %         rectangle('position',res(pick(i),1:4),'edgecolor','r');
% % 
% % %         img1 = I(y:y+h,x:x+w);
% %         if Mid>=x
% %             img2 = I(y:y+h,Mid - (x - Mid) - w:Mid - (x - Mid));
% %         else
% %             img2 = I(y:y+h,Mid + (Mid - x) - w:Mid + (Mid - x));
% %         end
% %         img2 = fliplr(img2);
% %         [feature1,vision] = extractHOGFeatures(double(img1),'CellSize',[8 8]);
% %         [feature2,vision] = extractHOGFeatures(double(img2),'CellSize',[8 8]);
% %         hogchidis = distMATChiSq     uare(feature1,feature2)/w/h
% %         if(hogchidis>0.02)
% %            rectangle('position',res(pick(i),1:4),'edgecolor','r');
% %         end
%     end
%    
    save_name=[save_root,fileList(k).name(1:end-3),'jpg'];
    saveas(gcf,save_name)
end
sumcount
fclose(regionlistfid);

