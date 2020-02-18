clear;
clc;

basePath    = '/home/vahid/RL_project_1/part3/data/train_class/';

imgSetsRGB = [ imageSet( fullfile( basePath, 'rgb_class/', '1'  ) ), ...
               imageSet( fullfile( basePath, 'rgb_class/', '2'  ) ), ...
               imageSet( fullfile( basePath, 'rgb_class/', '3'  ) ), ...
               imageSet( fullfile( basePath, 'rgb_class/', '4'  ) ), ...
               imageSet( fullfile( basePath, 'rgb_class/', '5'  ) ), ...
               imageSet( fullfile( basePath, 'rgb_class/', '6'  ) ), ...
               imageSet( fullfile( basePath, 'rgb_class/', '7'  ) ), ...
               imageSet( fullfile( basePath, 'rgb_class/', '8'  ) ), ...
               imageSet( fullfile( basePath, 'rgb_class/', '9'  ) ), ...
               imageSet( fullfile( basePath, 'rgb_class/', '10' ) ), ...
               imageSet( fullfile( basePath, 'rgb_class/', '11' ) ), ...
               imageSet( fullfile( basePath, 'rgb_class/', '12' ) ), ...
             ];
         
for i = 1 : numel( imgSetsRGB )
    
    numImages = imgSetsRGB( i ).Count;
    
    for j = 1 : numImages
        location   = imgSetsRGB(i).ImageLocation(j);
        slashPlace = find( location{1,1} == '/' );
        dotPlace   = find( location{1,1} == '.' );
        classNum   = str2double( location{1,1}( slashPlace(end-1) + 1 : slashPlace(end) - 1 ) );
        imgNum     = str2double( location{1,1}( slashPlace(end) + 1 : dotPlace(1) - 1 ) );
        depth = imread( sprintf( '%s%s%d%s%d%s', basePath, 'depth_class/', classNum, '/', imgNum, '.png' ) );
        img = read( imgSetsRGB(i), j );
        clear temp;
        temp(:,:,1) = img(:,:,1);
        temp(:,:,2) = img(:,:,2);
        temp(:,:,3) = img(:,:,3);
        temp(:,:,4) = depth;
        imwrite( temp, sprintf( '%s%s%d%s%d%s', basePath, 'tiff_class/', classNum, '/', imgNum, '.tiff' ) );
    end
    
end

clearvars;