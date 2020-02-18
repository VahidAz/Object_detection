function [] = createTrainingData_rgbdimg( percent, fc )

basePathRGB = '/home/vahid/RL_project_1/data/train_class/rgb_class/';
basePathDTH = '/home/vahid/RL_project_1/data/train_class/rgbdepth_class/';

imgSetsRGB = imageSet( basePathRGB, 'recursive' );

%[trainingSetsRGB, validationSetsRGB] = partition( imgSetsRGB, percent, 'randomize' );
trainingSetsRGB = imgSetsRGB;

location   = trainingSetsRGB(1).ImageLocation(1);
slashPlace = find( location{1,1} == '/' );
dotPlace   = find( location{1,1} == '.' );
classNum   = str2double( location{1,1}( slashPlace(end-1) + 1 : slashPlace(end) - 1 ) );
imgNum     = str2double( location{1,1}( slashPlace(end) + 1 : dotPlace(1) - 1 ) );

depth = imread( sprintf( '%s%d%s%d%s', basePathDTH, classNum, '/', imgNum, '.png' ) );

[scoresD, ~] = classification_demo(depth, 1, fc);

%I = imresize( depth, [256 256] );
I = single(vl_imdown(rgb2gray(depth))) ;
binSize = 8 ;
magnif = 3 ;
Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;
[fD, dD] = vl_dsift(Is, 'size', binSize) ;


img = read( trainingSetsRGB(1), 1 );
[scores, ~] = classification_demo(img, 1, fc);

%I = imresize( img, [256 256] );
I = single(vl_imdown(rgb2gray(img))) ;
binSize = 8 ;
magnif = 3 ;
Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;
[f, d] = vl_dsift(Is, 'size', binSize) ;

featureSize = length( scores ) + length( scoresD ) + length( dD(:) ) + length( d(:) );


trainingFeatures = [];
trainingLabels   = [];


for i = 1 : numel( trainingSetsRGB )
    
    numImages = trainingSetsRGB( i ).Count;
    features  = zeros( numImages, featureSize );
    
    for j = 1 : numImages

        img = read( trainingSetsRGB(i), j );
        [scores, ~] = classification_demo(img, 1, fc);
       
        I = single(vl_imdown(rgb2gray(img))) ;
        binSize = 8 ;
        magnif = 3 ;
        Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;
        [f, d] = vl_dsift(Is, 'size', binSize);
        
        location   = trainingSetsRGB(i).ImageLocation(j);
        slashPlace = find( location{1,1} == '/' );
        dotPlace   = find( location{1,1} == '.' );
        classNum   = str2double( location{1,1}( slashPlace(end-1) + 1 : slashPlace(end) - 1 ) );
        imgNum     = str2double( location{1,1}( slashPlace(end) + 1 : dotPlace(1) - 1 ) );
        
        depth = imread( sprintf( '%s%d%s%d%s', basePathDTH, classNum, '/', imgNum, '.png' ) );
        
        [scoresD, ~] = classification_demo(depth, 1, fc);
        
        I = single(vl_imdown(rgb2gray(depth))) ;
        binSize = 8 ;
        magnif = 3 ;
        Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;
        [fD, dD] = vl_dsift(Is, 'size', binSize);
        
        features(j, :) = [scores' scoresD' single(d(:)') single(dD(:)')];
    end

    labels = repmat( str2num(trainingSetsRGB(i).Description), numImages, 1 );
    
    trainingFeatures = [trainingFeatures; features];
    trainingLabels   = [trainingLabels;   labels  ];
end


% validationFeatures = [];
% validationLabels   = [];
% 
% 
% for i = 1 : numel( validationSetsRGB )
%     
%     numImages = validationSetsRGB( i ).Count;
%     features  = zeros( numImages, featureSize );
% 
%     for j = 1 : numImages
% 
%         img = read( validationSetsRGB(i), j );
%         [scores, ~] = classification_demo(img, 0, fc);
%         
%         location   = validationSetsRGB(i).ImageLocation(j);
%         slashPlace = find( location{1,1} == '/' );
%         dotPlace   = find( location{1,1} == '.' );
%         classNum   = str2double( location{1,1}( slashPlace(end-1) + 1 : slashPlace(end) - 1 ) );
%         imgNum     = str2double( location{1,1}( slashPlace(end) + 1 : dotPlace(1) - 1 ) );
%         
%         depth = imread( sprintf( '%s%d%s%d%s', basePathDTH, classNum, '/', imgNum, '.png' ) );
%         
%         [scoresD, ~] = classification_demo(depth, 0, fc);
%         
%         features(j, :) = [scores' scoresD'];
%     end
% 
%     labels = repmat( str2num(validationSetsRGB(i).Description), numImages, 1 );
%     
%     validationFeatures = [validationFeatures; features];
%     validationLabels   = [validationLabels;   labels  ];
% end
% 

load('testID.mat');

testFeatures = [];

for i = 1 : size( testID, 1 )
    
    img   = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/rgb/'     , i, '.png' ) );
    depth = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/rgbdepth/', i, '.png' ) );
    
    
    [scores, ~]  = classification_demo(img  , 0, fc);
    [scoresD, ~] = classification_demo(depth, 0, fc);
        
    testFeatures(i, :) = [scores' scoresD'];
end


save( sprintf( '%s%d%s%d%s', 'trainingFeatures_rgbdimg_'  , fc, '_', percent, '.mat' ) , 'trainingFeatures'  , '-v7.3' );
save( sprintf( '%s%d%s%d%s', 'trainingLables_rgbdimg_'    , fc, '_', percent, '.mat' ) , 'trainingLabels'    , '-v7.3' );
%save( sprintf( '%s%d%s%d%s', 'validationFeatures_rgbdimg_', fc, '_', percent, '.mat' ) , 'validationFeatures', '-v7.3' );
%save( sprintf( '%s%d%s%d%s', 'validationLabels_rgbdimg_'  , fc, '_', percent, '.mat' ) , 'validationLabels'  , '-v7.3' );
save( sprintf( '%s%d%s%d%s', 'testFeatures_rgbdimg_'      , fc, '_', percent, '.mat' ) , 'testFeatures'      , '-v7.3' );

clearvars;

end