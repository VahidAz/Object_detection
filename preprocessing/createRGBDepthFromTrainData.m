clear;
clc;

basePath    = '/home/vahid/RL_project_1/part3/data/train_class/';

imgSetsDepth = [ 
               imageSet( fullfile( basePath, 'depth_class/', '1'  ) ), ...
               imageSet( fullfile( basePath, 'depth_class/', '2'  ) ), ...
               imageSet( fullfile( basePath, 'depth_class/', '3'  ) ), ...
               imageSet( fullfile( basePath, 'depth_class/', '4'  ) ), ...
               imageSet( fullfile( basePath, 'depth_class/', '5'  ) ), ...
               imageSet( fullfile( basePath, 'depth_class/', '6'  ) ), ...
               imageSet( fullfile( basePath, 'depth_class/', '7'  ) ), ...
               imageSet( fullfile( basePath, 'depth_class/', '8'  ) ), ...
               imageSet( fullfile( basePath, 'depth_class/', '9'  ) ), ...
               imageSet( fullfile( basePath, 'depth_class/', '10' ) ), ...
               imageSet( fullfile( basePath, 'depth_class/', '11' ) ), ...
               imageSet( fullfile( basePath, 'depth_class/', '12' ) ), ...
             ];
         
for i = 1 : numel( imgSetsDepth )
    
    numImages = imgSetsDepth( i ).Count;
    
    for j = 1 : numImages
        depth = read( imgSetsDepth(i), j );
        
        [Nx,Ny,Nz] = surfnorm(depth);
        
        clear RGBImg;
        
        RGBImg(:,:,1)= Nx;
        RGBImg(:,:,2)= Ny;
        RGBImg(:,:,3)= Nz;
        
        location   = imgSetsDepth(i).ImageLocation(j);
        slashPlace = find( location{1,1} == '/' );
        dotPlace   = find( location{1,1} == '.' );
        classNum   = str2double( location{1,1}( slashPlace(end-1) + 1 : slashPlace(end) - 1 ) );
        imgNum     = str2double( location{1,1}( slashPlace(end) + 1 : dotPlace(1) - 1 ) );
       
        imwrite( RGBImg, sprintf( '%s%s%d%s%d%s', basePath, 'rgbdepth_class/', classNum, '/', imgNum, '.png' ) );
    end
    
end

clearvars;  