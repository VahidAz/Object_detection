%% Classifying Images in Corresponding Classes

clear;
clc;

basePath = '/home/vahid/RL_project_1/part3/data/';

trainPath = sprintf( '%s%s', basePath, 'train/' );

boxesPrefix    = 'boxes_';
boxesExtension = '.txt';
depthPrefix    = 'depth_';
depthExtension = '.png';
imagePrefix    = 'image_';
imageExtension = '.png';

% Reading Train Data
startIndex = 0;
endIndex   = 1022;

objectsList;
counter = zeros( size(objsList, 1), 1 );

for i = startIndex : endIndex
    
    txtFileName = sprintf( '%s%s%d%s', trainPath, boxesPrefix, i, boxesExtension );
    fid = fopen( txtFileName );
    if fid == -1
        fprintf( '%s%s', 'Could not open >>> ', txtFileName );
        return;
    end
    
    imageFileName = sprintf( '%s%s%d%s', trainPath, imagePrefix, i, imageExtension );
    img = imread( imageFileName );
    
    depthFileName = sprintf( '%s%s%d%s', trainPath, depthPrefix, i, depthExtension );
    dpth = imread( depthFileName );
    
    tline = fgets(fid);
    while ischar(tline)
        disp(tline);
        splitedStr = strsplit( tline, ',' );
        
        lable = str2num( cell2mat( splitedStr(1, 1) ) );
        xMin  = str2num( cell2mat( splitedStr(1, 2) ) );
        yMin  = str2num( cell2mat( splitedStr(1, 3) ) );
        xMax  = str2num( cell2mat( splitedStr(1, 4) ) ) - xMin;
        yMax  = str2num( cell2mat( splitedStr(1, 5) ) ) - yMin;
        
        curImg  = imcrop(img,  [xMin yMin xMax yMax]);
        dpthImg = imcrop(dpth, [xMin yMin xMax yMax]);
        
        counter(lable, 1) = counter(lable, 1) + 1;
              
        imwrite( curImg , sprintf( '%s%s%d%s%d%s', basePath, 'train_class/rgb_class/'  , lable, '/', counter(lable, 1), '.png') );
        imwrite( dpthImg, sprintf( '%s%s%d%s%d%s', basePath, 'train_class/depth_class/', lable, '/', counter(lable, 1), '.png') );
  
        tline = fgets(fid);
    end
    
    fclose(fid);

end

clearvars;