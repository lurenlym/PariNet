function feat = compute_dalaltriggs(im)
% featIm = compute_dalaltriggs(impatch)
% let [imy,imx] = size(impatch)
% Then featIm will be (imy/8-1) by (imx/8-1) by (4*9)
% where there are 9 orientation bins, and 4 normalizations 
% for every 8x8 pixel-block


%Compute the gradient in each plane, and take the best one
[imy,imx,imz] = size(im);
%[imy,imx] = size(im);
%Stick on extra zeros at the end if we need to
if round(imy/18) ~= imy/18,
    extray = ceil(imy/18)*18 - imy;
    im(end+1:end+extray,:,:) = 0;
    imy = size(im,1);
end
if round(imx/18) ~=imx/18,
    extrax = ceil(imx/18)*18 - imx;
    im(:,end+1:end+extrax,:) = 0;    
    imx = size(im,2);
end

im = double(im);
n = (imy-2)*(imx-2);

%Pick the strongest gradient across color channels
dy = im(3:end,2:end-1,:) - im(1:end-2,2:end-1,:); dy = reshape(dy,n,3); 
dx = im(2:end-1,3:end,:) - im(2:end-1,1:end-2,:); dx = reshape(dx,n,3);
len = dx.^2 + dy.^2;
[len,I] = max(len,[],2);
len = sqrt(len);
I = sub2ind([n 3],[1:n]',I);
dy = dy(I); dx = dx(I);

%Snap to an orientation
[uu,vv] = pol2cart([0:pi/9:pi-.01],1);
v = dy./(len+eps); u = dx./(len+eps);
[dummy,I] = max(abs(u(:)*uu + v(:)*vv),[],2);

%Bin spatially
sbin = 18;
ssiz = [imy imx]/sbin;
feat = zeros(prod(ssiz), 9);
for i = 1:9,
    %Generate sparse map
    tmp = reshape(len.*(I == i),imy-2,imx-2);
    tmp = padarray(tmp,[1 1]);
    feat(:,i) = sum(im2col(tmp,[sbin sbin],'distinct'))';
end

indMask = reshape(1:prod(ssiz),ssiz);
indMask = im2col(indMask,[2 2])';
n = size(indMask,1);
feat = reshape(feat(indMask,:),n,4*9);

%Normalize the way he does
nn = sqrt(sum(feat.^2,2)) + eps;
feat = feat./repmat(nn,1,4*9);
feat = min(feat,.2); %Clip to .2
feat = reshape(feat,[ssiz-1 (4*9)]);