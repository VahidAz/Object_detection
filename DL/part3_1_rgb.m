clear;
clc;

fc = 1;

basePathRGB = '/home/vahid/RL_project_1/part3/data/test1/rgb/';

classifier = load('/home/vahid/caffe/matlab/demo/classifiers/mcsvm_rgb_all.mat');

sizeW        = 128;
step         = 16;
testNum      = 126;
pyramidSize  = 1;
pyramidScale = 0.75;

fileID = fopen( sprintf( '%s', 'results/part3_1.txt' ), 'wt+' );

fprintf( fileID, '%s\n', 'Id,bb_tl_x,bb_tl_y,bb_br_x,bb_br_y' );

for mm = 0 : testNum
    
    img   = imread( sprintf( '%s%s%d%s', basePathRGB, 'image_', mm, '.png') );
   
    probList   = [];
    lblList    = [];
    windowInfo = [];
    maxProbs   = [];
    
    for ii = 1 : pyramidSize
        
        if ii ~= 1
            img   = imresize( img  , pyramidScale, 'bilinear' );
        end
        
        for k = 1 : step : size( img, 1 ) - sizeW

            for l = 1 : step : size( img, 2 ) - sizeW

                im   = imcrop( img  , [l k sizeW sizeW] );
               
                classification_demo1;

                features = [scores'];

                [predictedLabels probs] = predict( classifier.classifier, features );
                
                if max(probs) > -0.01
                    lblList  = [lblList predictedLabels];
                    probList   = [probList max(probs)];
                else
                    predictedLabels = -1;
                    probs = -1000;
                    
                    probList   = [probList probs];
                    lblList  = [lblList predictedLabels];
                end
                
                windowTmp  = [l k pyramidScale sizeW];
                windowInfo = [windowInfo windowTmp'];
                
            end

        end
    end
    
    txtFileName = sprintf( '%s%s%d%s', '/home/vahid/RL_project_1/part3/data/test1/', 'boxes_', mm, '.txt' );
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
            tmpMax(1,jj)= probList( res(1,jj) );
            tmpMax(2,jj)= res(1,jj);
        end
        
         if isempty(res) == 1
            fprintf( fileID, '%d%s%d%s%d%s%d%s%d%s%d\n', mm, '_', count, ',',...
            0, ',', 0, ',', 0 + sizeW, ',', 0 + sizeW  );
         else
            [maxx maxI] = max(tmpMax(1,:));
            maxI = tmpMax(2,maxI);
            
            fprintf( fileID, '%d%s%d%s%d%s%d%s%d%s%d\n', mm, '_', count, ',',...
            windowInfo(1, maxI), ',', windowInfo(2, maxI), ',', windowInfo(1, maxI) + sizeW, ',', windowInfo(2, maxI) + sizeW  );
         end
   
        tline = fgets(fid);
        count = count + 1;
    end
    
end