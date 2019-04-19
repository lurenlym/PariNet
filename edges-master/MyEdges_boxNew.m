close all;
clear;
rdir = 'E:\liuyuming\SiameseNet\DATA\WH180605\';
LR = 'R';
root_dir = [rdir,LR,'\','VAL1\'];
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
regionlist = [rdir,LR,'\','regionlistVal1.txt'];
mergeimg = [rdir,LR,'\','MergeImg\'];
if exist(mergeimg,'dir')==0
   mkdir(mergeimg);
end
imshow_root =  [rdir,LR,'\','imshow_root\'];
if exist(imshow_root,'dir')==0
   mkdir(imshow_root);
end
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
opts.alpha = .75;     % step size of sliding window search
opts.beta  = 0.5;  % nms threshold for object proposals
opts.minScore = .1;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to d30petect
regionlistfid=fopen(regionlist,'w');
%% detect Edge Box bounding box proposals (see edgeBoxes.m)
for k = 4:filenumber
    if strcmp(fileList(k).name,'.')==1||strcmp(fileList(k).name,'..')==1
        continue;
    end
    imgnamepre = fileList(k-1).name;
    imgname = fileList(k).name;
    full_name=[root_dir,imgname]
    full_namepre=[root_dir,imgnamepre]
    Ipre = imread(full_namepre);
    I = imread(full_name);
    sumcol = sum(I,1);
    Mid = int32(sum(find(sumcol==295800),2)/size(find(sumcol==295800),2));
    I=[Ipre;I];
    labelfile = [edgbox_root,imgnamepre(1:end-3),'txt'];
    fid=fopen(labelfile,'w');
    %I=I(1:1090,:);
   
    I1 = zeros([size(I,1),size(I,2),3]);
    I1=im2uint8(I1);
    I1(:,:,1)=I;
    I1(:,:,2)=I;
    I1(:,:,3)=I;
    tic, bbs=edgeBoxes(I1,model,opts); toc
%     index = NMS(bbs,0.95);
%     bbs = bbs(index,:);
    imshow(I1)
    imwrite(I1,[mergeimg,imgnamepre]);
    res=[];
    region_count=0;
    for i =1:size(bbs,1)
        x = int32(bbs(i,1));
        y = int32(bbs(i,2));
        w = int32(bbs(i,3));
        h = int32(bbs(i,4));
        score = bbs(i,5);
        if x <155 || x+w>645 || w<10 || w>200|| h<10 ||h>200  || w*h<1200 || score<0.1 || y+h>2100||y<200
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
        diffrawandflase=0;
%         while(diffrawandflase<5)
%             if Mid>=x
%                 FalseX = int32((670-x-w-w)*rand(1))+x+w;
%                 imgFalse = I(y:y+h,FalseX:FalseX+w);
%             else
%                 FalseX = int32((x-w-155)*rand(1))+155;
%                 if(y>1160/2)
%                     FalseY = int32((y-h)*rand(1))+1;
%                     imgFalse = I(FalseY:FalseY+h,FalseX:FalseX+w);
%                 else
%                     FalseY = int32((1160/2-h-h)*rand(1)+1160/2+h);
%                     imgFalse = I(FalseY:FalseY+h,FalseX:FalseX+w);
%                 end
%             end
                [sizex,sizey]=size(I);
                FalseX = int32((sizey-w-1)*rand(1))+1;
                FalseY = int32((sizex-h-1)*rand(1))+1;
                imgFalse = I(FalseY:FalseY+h,FalseX:FalseX+w);
%             imgsize=size(img1);
%             diffrawandflase=sum(sum(pdist2(im2double(img1),im2double(imgFalse),'euclidean')))/(imgsize(1)*imgsize(2));
%             if(diffrawandflase<5)
%                 pause;
%             end
%         end
        
        imgTure = fliplr(imgTure);
        regionname = [imgnamepre(1:end-4),'_',num2str(region_count),'.','jpg'];
        imwrite(img1,[regionraw_root,regionname]);
        imwrite(imgTure,[regionture_root,regionname]);
        imwrite(imgFalse,[regionfalse_root,regionname]);
        imwrite([img1,imgTure,imgFalse],[imshow_root,regionname]);
        fprintf(regionlistfid,'%s\n',regionname);
        region_count = region_count+1;
    end
    fclose(fid);

    save_name=[save_root,fileList(k).name(1:end-3),'jpg'];
    saveas(gcf,save_name)
end
fclose(regionlistfid);

