clear;
clc;

basePathRGB = '/home/vahid/RL_project_1/part2/data/test1/rgb/';

classifier = load('/home/vahid/caffe/matlab/demo/classifiers/mcsvm_rgb_all.mat');

sizeW   = 128;
step    = 16;
testNum = 113;
fc      = 1;

fileID = fopen( sprintf( '%s', 'results/part2.txt' ), 'wt+' );

fprintf( fileID, '%s\n', 'Id,Binary' );

for mm = 0 : testNum
    
    img   = imread( sprintf( '%s%s%d%s', basePathRGB, 'image_', mm, '.png') );
   
    probList = [];
    lblList  = [];
        
    for k = 1 : step : size( img, 1 ) - sizeW
            
        for l = 1 : step : size( img, 2 ) - sizeW
                    
            im   = imcrop( img  , [l k sizeW sizeW] );
            
            classification_demo1;
               
            features = [scores'];
                
            [predictedLabels probs] = predict( classifier.classifier, features );
            
            if max(probs) > -0.01
               lblList  = [lblList predictedLabels];
            else
                predictedLabels = -1;
                
                lblList  = [lblList predictedLabels];
            end
                
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
