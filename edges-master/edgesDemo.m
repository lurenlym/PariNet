% Demo for Structured Edge Detector (please see readme.txt first).

%% set opts for training (see edgesTrain.m)
opts=edgesTrain();                % default options (good settings)
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm='modelBsds';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
opts.useParfor=0;                 % parallelize if sufficient memory

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
tic, model=edgesTrain(opts); toc; % will load model if already trained

%% set detection parameters (can set after training)
model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpn=0
model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=0;                 % set to true to enable nms

%% evaluate edge detector on BSDS500 (see edgesEval.m)
if(0), edgesEval( model, 'show',1, 'name','' ); end

%% detect edge and visualize results

I = imread('C:\Users\lyming\Desktop\anomalydata\anomalydata\zoo\KM1106+186M.jpg');
I1 = zeros([size(I,1),size(I,2),3]);
I1=im2uint8(I1);
I1(:,:,1)=I;
I1(:,:,2)=I;
I1(:,:,3)=I;

I = imread('peppers.png');
%I = imread('a.jpg');
tic, E=edgesDetect(I1,model); toc
E
figure(1); im(I1); figure(2); im(1-E);
