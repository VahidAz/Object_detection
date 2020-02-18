clear;
clc;

basePathRGB = '/home/vahid/RL_project_1/part2/data/test1/rgb/';
%basePathDPH = '/home/vahid/RL_project_1/part2/data/test1/depth/';

classifier = load('/home/vahid/caffe/matlab/demo/classifiers/mcsvm_rgb_all.mat');

sizeW   = 128;
step    = 16;
testNum = 113;
fc      = 1;

fileID = fopen( sprintf( '%s', 'results/part2.txt' ), 'wt+' );

fprintf( fileID, '%s\n', 'Id,Binary' );

for mm = 0 : testNum
    
    img   = imread( sprintf( '%s%s%d%s', basePathRGB, 'image_', mm, '.png') );
    %depth = imread( sprintf( '%s%s%d%s', basePathDPH, 'depth_', i, '.png') );
   
    probList = [];
    lblList  = [];
        
    for k = 1 : step : size( img, 1 ) - sizeW
            
        for l = 1 : step : size( img, 2 ) - sizeW
                    
            im   = imcrop( img  , [l k sizeW sizeW] );
            classification_demo1;
            %curDepth = imcrop( depth, [l k sizeW sizeW] );
                
            %[scores, ~] = classification_demo(curImg, 1, fc);
            %[surf  , ~] = get_surf_rgb(curImg);
                
            %[surfD  , ~] = get_surf_rgb(curDepth);
                
%             [Nx,Ny,Nz] = surfnorm(curDepth);
%         
%             clear RGNImg;
%                 
%             RGBImg(:,:,1)= Nx;
%             RGBImg(:,:,2)= Ny;
%             RGBImg(:,:,3)= Nz;
%     
%             [scoresD, ~] = classification_demo(RGBImg, 1, fc);
               
            features = [scores'];% scoresD' surf(:)' surfD(:)'];
                
            [predictedLabels probs] = predict( classifier.classifier, features );
            
            max(probs)
                
            %probList = [probList probs'];
            lblList  = [lblList predictedLabels];
                
        end
       
    end
     
    final = zeros(12,1);
    
    for ii = 1 : size(final,1)
        res = find( lblList == ii, 1);
        
        if isempty(res) == 0
            final( ii, 1) = 1;
        end
    end
    
    for ii = 1 : size(final,1)
        fprintf( fileID, '%d%s%d%s%d\n', mm, '_', ii, ',', final(ii,1) );
    end
    
end
