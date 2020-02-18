clear;
clc;

fc = 1;

basePathRGB = '/home/vahid/RL_project_1/part3/data/test1/rgb/';
basePathDPH = '/home/vahid/RL_project_1/part3/data/test1/depth/';

classifier = load('/home/vahid/caffe/matlab/demo/classifiers/mcsvm_rgb_9.175000e+01_fc6.mat');

sizeW        = 64;
step         = 8;
testNum      = 126;
pyramidSize  = 1;
pyramidScale = 0.75;

fileID = fopen( sprintf( '%s', 'results/part3_1.txt' ), 'wt+' );

fprintf( fileID, '%s\n', 'Id,bb_tl_x,bb_tl_y,bb_br_x,bb_br_y' );

for i = 0 : testNum
    
    img   = imread( sprintf( '%s%s%d%s', basePathRGB, 'image_', i, '.png') );
    %depth = imread( sprintf( '%s%s%d%s', basePathDPH, 'depth_', i, '.png') );
   
    probList   = [];
    lblList    = [];
    windowInfo = [];
    maxProbs   = [];
    
    for ii = 1 : pyramidSize
        
        if ii ~= 1
            img   = imresize( img  , pyramidScale, 'bilinear' );
            %depth = imresize( depth, pyramidScale, 'bilinear' );
        end
        
        for k = 1 : step : size( img, 1 ) - sizeW

            for l = 1 : step : size( img, 2 ) - sizeW

                curImg   = imcrop( img  , [l k sizeW sizeW] );
                %curDepth = imcrop( depth, [l k sizeW sizeW] );

                [scores, ~] = classification_demo(curImg, 1, fc);
%                 [surf  , ~] = get_surf_rgb(curImg);
% 
%                 [surfD  , ~] = get_surf_rgb(curDepth);
% 
%                 [Nx,Ny,Nz] = surfnorm(curDepth);
% 
%                 clear RGNImg;
% 
%                 RGBImg(:,:,1)= Nx;
%                 RGBImg(:,:,2)= Ny;
%                 RGBImg(:,:,3)= Nz;
% 
%                 [scoresD, ~] = classification_demo(RGBImg, 1, fc);

                features = [scores'];% scoresD' surf(:)' surfD(:)'];

                [predictedLabels probs] = predict( classifier.classifier, features );
                
                
                probList   = [probList probs'];
                lblList    = [lblList predictedLabels];
                
                windowTmp = [l k pyramidScale sizeW];
                windowInfo = [windowInfo windowTmp'];
                
                maxProbs = [maxProbs max(probs')];

            end

        end
    end
    
    txtFileName = sprintf( '%s%s%d%s', '/home/vahid/RL_project_1/part3/data/test1/', 'boxes_', i, '.txt' );
    fid = fopen( txtFileName );
     
    tline = fgets(fid);
    count = 1;
    while ischar(tline)
        disp(tline);
        splitedStr = strsplit( tline, ',' );
        objNum = str2num( cell2mat( splitedStr(1, 2) ) );   
    
        res = find( lblList == objNum );
        
        tmpMax = [;];
       
        for jj = 1 : size(res,2)
            tmpMax(1,jj)= maxProbs( res(1,jj) );
            tmpMax(2,jj)= res(1,jj);
        end
        
         if isempty(res) == 1
            fprintf( fileID, '%d%s%d%s%d%s%d%s%d%s%d\n', i, '_', count, ',',...
            0, ',', 0, ',', 0 + sizeW, ',', 0 + sizeW  );
         else
            [maxx maxI] = max(tmpMax(1,:));
            
            fprintf( fileID, '%d%s%d%s%d%s%d%s%d%s%d\n', i, '_', count, ',',...
            windowInfo(1, maxI), ',', windowInfo(2, maxI), ',', windowInfo(1, maxI) + sizeW, ',', windowInfo(2, maxI) + sizeW  );
         end
   
        tline = fgets(fid);
        count = count + 1;
    end
    
end