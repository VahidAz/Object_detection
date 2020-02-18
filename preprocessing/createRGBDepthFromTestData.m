clear;
clc;

basePath = '/home/vahid/RL_project_1/data/extracted_test/';

imgSetsDepth = imageSet( fullfile( basePath, 'depth/' ) );
         
for i = 1 : imgSetsDepth.Count
    
    depth = read( imgSetsDepth, i );
        
    [Nx,Ny,Nz] = surfnorm(depth);
        
    clear RGBImg;
        
    RGBImg(:,:,1)= Nx;
    RGBImg(:,:,2)= Ny;
    RGBImg(:,:,3)= Nz;
        
    location   = imgSetsDepth.ImageLocation(i);
    slashPlace = find( location{1,1} == '/' );
    dotPlace   = find( location{1,1} == '.' );
    num   = str2double( location{1,1}( slashPlace(end) + 1 : dotPlace(1) - 1 ) );
       
    imwrite( RGBImg, sprintf( '%s%s%d%s', basePath, 'rgbdepth/', num, '.png' ) );
end

clearvars;