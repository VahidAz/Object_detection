clear;
clc;

basePath = '/home/vahid/RL_project_1/part1/data/';

testPath = sprintf( '%s%s', basePath, 'test1/' );

boxesPrefix    = 'boxes_';
boxesExtension = '.txt';
depthPrefix    = 'depth_';
depthExtension = '.png';
imagePrefix    = 'image_';
imageExtension = '.png';

startIndex = 0;
endIndex   = 113;

counter = 0;

for i = startIndex : endIndex
    
    txtFileName = sprintf( '%s%s%d%s', testPath, boxesPrefix, i, boxesExtension );
    fid = fopen( txtFileName );
    if fid == -1
        fprintf( '%s%s', 'Could not open >>> ', txtFileName );
        return;
    end
    
    imageFileName = sprintf( '%s%s%d%s', testPath, imagePrefix, i, imageExtension );
    img = imread( imageFileName );
    
    depthFileName = sprintf( '%s%s%d%s', testPath, depthPrefix, i, depthExtension );
    dpth = imread( depthFileName );
    
    tline = fgets(fid);
    while ischar(tline)
        disp(tline)
        splitedStr = strsplit( tline, ',' );
        
        lable = cell2mat( splitedStr(1, 1) ) ;
        xMin  = str2num( cell2mat( splitedStr(1, 2) ) );
        yMin  = str2num( cell2mat( splitedStr(1, 3) ) );
        xMax  = str2num( cell2mat( splitedStr(1, 4) ) ) - xMin;
        yMax  = str2num( cell2mat( splitedStr(1, 5) ) ) - yMin;
                
        curImg  = imcrop(img,  [xMin yMin xMax yMax]);
        dpthImg = imcrop(dpth, [xMin yMin xMax yMax]);
        
        counter = counter + 1;
        
        testID{counter, 1} = lable;
        
        imwrite( curImg , sprintf( '%s%s%d%s', basePath, 'extracted_test/rgb/'  , counter, '.png' ) );
        imwrite( dpthImg, sprintf( '%s%s%d%s', basePath, 'extracted_test/depth/', counter, '.png' ) );
        
        tline = fgets(fid);
    end
    
    fclose(fid);

end

save( 'testID', 'testID' );

clearvars;