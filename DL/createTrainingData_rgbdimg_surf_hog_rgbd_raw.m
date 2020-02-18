function [] = createTrainingData_rgbdimg_surf_hog_rgbd_raw( percent, fc )

basePathRGB    = '/home/vahid/RL_project_1/data/train_class/rgb_class/';
basePathDTH    = '/home/vahid/RL_project_1/data/train_class/rgbdepth_class/';
basePathDTHRaw = '/home/vahid/RL_project_1/data/train_class/depth_class/';

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

depth = imread( sprintf( '%s%d%s%d%s', basePathDTHRaw, classNum, '/', imgNum, '.png' ) );
[surfD  , ~] = get_surf_rgb(depth);


img = read( trainingSetsRGB(1), 1 );
[scores, ~] = classification_demo(img, 1, fc);
[surf  , ~] = get_surf_rgb(img);


featureSize = length( scores ) + length( scoresD ) + length( surfD(:) ) + length( surf(:) );


trainingFeatures = [];
trainingLabels   = [];


for i = 1 : numel( trainingSetsRGB )
    
    numImages = trainingSetsRGB( i ).Count;
    features  = zeros( numImages, featureSize );
    
    for j = 1 : numImages

        img = read( trainingSetsRGB(i), j );
        [scores, ~] = classification_demo(img, 1, fc);
        [surf  , ~] = get_surf_rgb(img);
%         img = imresize( img, [150 150] );
% 
%         hog_4x4 = extractHOGFeatures( img, 'CellSize', [4 4] );
        
        location   = trainingSetsRGB(i).ImageLocation(j);
        slashPlace = find( location{1,1} == '/' );
        dotPlace   = find( location{1,1} == '.' );
        classNum   = str2double( location{1,1}( slashPlace(end-1) + 1 : slashPlace(end) - 1 ) );
        imgNum     = str2double( location{1,1}( slashPlace(end) + 1 : dotPlace(1) - 1 ) );
        
        depth = imread( sprintf( '%s%d%s%d%s', basePathDTH, classNum, '/', imgNum, '.png' ) );
        
        [scoresD, ~] = classification_demo(depth, 1, fc);
        
        depth = imread( sprintf( '%s%d%s%d%s', basePathDTHRaw, classNum, '/', imgNum, '.png' ) );
        [surfD  , ~] = get_surf_rgb(depth);
        
%         depth = mat2gray( depth );
%         depth = imresize( depth, [150 150] );
% 
%         hog_4x4D = extractHOGFeatures( depth, 'CellSize', [4 4] );
        
        features(j, :) = [scores' scoresD' surf(:)' surfD(:)'];
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
%         [scores, ~] = classification_demo(img, 1, fc);
%         [surf  , ~] = get_surf_rgb(img);
%         img = imresize( img, [150 150] );
% 
%         hog_4x4 = extractHOGFeatures( img, 'CellSize', [4 4] );
%         
%         location   = validationSetsRGB(i).ImageLocation(j);
%         slashPlace = find( location{1,1} == '/' );
%         dotPlace   = find( location{1,1} == '.' );
%         classNum   = str2double( location{1,1}( slashPlace(end-1) + 1 : slashPlace(end) - 1 ) );
%         imgNum     = str2double( location{1,1}( slashPlace(end) + 1 : dotPlace(1) - 1 ) );
%         
%         depth = imread( sprintf( '%s%d%s%d%s', basePathDTH, classNum, '/', imgNum, '.png' ) );
%         
%         [scoresD, ~] = classification_demo(depth, 1, fc);
%         
%         depth = imread( sprintf( '%s%d%s%d%s', basePathDTHRaw, classNum, '/', imgNum, '.png' ) );
%         [surfD  , ~] = get_surf_rgb(depth);
%         
%         depth = mat2gray( depth );
%         depth = imresize( depth, [150 150] );
% 
%         hog_4x4D = extractHOGFeatures( depth, 'CellSize', [4 4] );
%         
%         features(j, :) = [scores' scoresD' surf(:)' surfD(:)' hog_4x4 hog_4x4D];
%     end
% 
%     labels = repmat( str2num(validationSetsRGB(i).Description), numImages, 1 );
%     
%     validationFeatures = [validationFeatures; features];
%     validationLabels   = [validationLabels;   labels  ];
% end


load('testID.mat');

testFeatures = [];

for i = 1 : size( testID, 1 )
    
    img   = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/rgb/'     , i, '.png' ) );
    depth = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/rgbdepth/', i, '.png' ) );
    
    
    [scores, ~]  = classification_demo(img  , 1, fc);
    [surf  , ~] = get_surf_rgb(img);
%     img = imresize( img, [150 150] );
% 
%     hog_4x4 = extractHOGFeatures( img, 'CellSize', [4 4] );
        
    [scoresD, ~] = classification_demo(depth, 1, fc);
    
    depth = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/depth/', i, '.png' ) );
    [surfD  , ~] = get_surf_rgb(depth);
    
%     depth = mat2gray( depth );
%     depth = imresize( depth, [150 150] );
% 
%     hog_4x4D = extractHOGFeatures( depth, 'CellSize', [4 4] );
        
    testFeatures(i, :) = [scores' scoresD' surf(:)' surfD(:)'];% hog_4x4 hog_4x4D];
end


save( sprintf( '%s%d%s', 'trainingFeatures_rgbdimg_surf_rgbd_raw_'  , fc, '.mat' ) , 'trainingFeatures'   , '-v7.3' );
save( sprintf( '%s%d%s', 'trainingLables_rgbdimg_surf_rgbd_raw_'    , fc, '.mat' ) , 'trainingLabels'     , '-v7.3' );
save( sprintf( '%s%d%s', 'validationFeatures_rgbdimg_surf_rgbd_raw_', fc, '.mat' ) , 'validationFeatures' , '-v7.3' );
save( sprintf( '%s%d%s', 'validationLabels_rgbdimg_surf_rgbd_raw_'  , fc, '.mat' ) , 'validationLabels'   , '-v7.3' );
save( sprintf( '%s%d%s', 'testFeatures_rgbdimg_surf_rgbd_raw_'      , fc, '.mat' ) , 'testFeatures'       , '-v7.3' );

clearvars;

end