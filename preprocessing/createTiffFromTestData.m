clear;
clc;

basePath = '/home/vahid/RL_project_1/data/extracted_test/';

imgSetsRGB = imageSet( fullfile( basePath, 'rgb/' ) );
         
for j = 1 : imgSetsRGB.Count
    location   = imgSetsRGB.ImageLocation(j);
    slashPlace = find( location{1,1} == '/' );
    dotPlace   = find( location{1,1} == '.' );
    num   = str2double( location{1,1}( slashPlace(end) + 1 : dotPlace(1) - 1 ) );
    depth = imread( sprintf( '%s%s%d%s', basePath, 'depth/', num, '.png' ) );
    img = read( imgSetsRGB, j );
    clear temp;
    temp(:,:,1) = img(:,:,1);
    temp(:,:,2) = img(:,:,2);
    temp(:,:,3) = img(:,:,3);
    temp(:,:,4) = depth;
    imwrite( temp, sprintf( '%s%s%d%s', basePath, 'tiff/', num, '.tiff' ) );
end
    
clearvars;