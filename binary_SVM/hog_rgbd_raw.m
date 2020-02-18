clear;
clc;

basePathRGB = '/home/vahid/RL_project_1/data/train_class/rgb_class/';
basePathDTH = '/home/vahid/RL_project_1/data/train_class/depth_class/';

imgSetsRGB = imageSet( basePathRGB, 'recursive' );

[trainingSetsRGB, validationSetsRGB] = partition( imgSetsRGB, 0.8, 'randomize' );

location   = trainingSetsRGB(1).ImageLocation(1);
slashPlace = find( location{1,1} == '/' );
dotPlace   = find( location{1,1} == '.' );
classNum   = str2double( location{1,1}( slashPlace(end-1) + 1 : slashPlace(end) - 1 ) );
imgNum     = str2double( location{1,1}( slashPlace(end) + 1 : dotPlace(1) - 1 ) );
depth = imread( sprintf( '%s%d%s%d%s', basePathDTH, classNum, '/', imgNum, '.png' ) );

depth = mat2gray( depth );
depth = imresize( depth, [150 150] );

[hog_2x2D, vis2x2D] = extractHOGFeatures( depth, 'CellSize', [2 2] );
[hog_4x4D, vis4x4D] = extractHOGFeatures( depth, 'CellSize', [4 4] );
[hog_8x8D, vis8x8D] = extractHOGFeatures( depth, 'CellSize', [8 8] );
        
img = read( trainingSetsRGB(1), 1 );
img = rgb2gray( img );
img = imresize( img, [150 150] );

[hog_2x2, vis2x2] = extractHOGFeatures( img, 'CellSize', [2 2] );
[hog_4x4, vis4x4] = extractHOGFeatures( img, 'CellSize', [4 4] );
[hog_8x8, vis8x8] = extractHOGFeatures( img, 'CellSize', [8 8] );

cellSize = [4 4];
hogFeatureSize = length( hog_4x4 ) + length( hog_4x4D );


trainingFeatures = [];
trainingLabels   = [];

for i = 1 : numel( trainingSetsRGB )
    
    numImages = trainingSetsRGB( i ).Count;
    features  = zeros( numImages, hogFeatureSize, 'single' );

    for j = 1 : numImages

        img = read( trainingSetsRGB(i), j );
        img = rgb2gray( img );
        img = imresize( img, [150 150] );
        
        featTemp = extractHOGFeatures( img, 'CellSize', cellSize );
        
        location   = trainingSetsRGB(i).ImageLocation(j);
        slashPlace = find( location{1,1} == '/' );
        dotPlace   = find( location{1,1} == '.' );
        classNum   = str2double( location{1,1}( slashPlace(end-1) + 1 : slashPlace(end) - 1 ) );
        imgNum     = str2double( location{1,1}( slashPlace(end) + 1 : dotPlace(1) - 1 ) );
        depth = imread( sprintf( '%s%d%s%d%s', basePathDTH, classNum, '/', imgNum, '.png' ) );
        
        depth = mat2gray( depth );
        depth = imresize( depth, [150 150] );
        
        depthTemp = extractHOGFeatures( depth, 'CellSize', cellSize );
        
        features(j, :) = [featTemp depthTemp];

    end

    labels = repmat( str2num(trainingSetsRGB(i).Description), numImages, 1 );
    
    trainingFeatures = [trainingFeatures; features];
    trainingLabels   = [trainingLabels;   labels  ];

end


classifier = fitcecoc(trainingFeatures, trainingLabels);


validationFeatures = [];
validationLabels   = [];

for i = 1 : numel( validationSetsRGB )
    
    numImages = validationSetsRGB( i ).Count;
    features  = zeros( numImages, hogFeatureSize, 'single' );

    for j = 1 : numImages

        img = read( validationSetsRGB(i), j );
        img = rgb2gray( img );
        img = imresize( img, [150 150] );
        
        featTemp = extractHOGFeatures( img, 'CellSize', cellSize );
        
        location   = validationSetsRGB(i).ImageLocation(j);
        slashPlace = find( location{1,1} == '/' );
        dotPlace   = find( location{1,1} == '.' );
        classNum   = str2double( location{1,1}( slashPlace(end-1) + 1 : slashPlace(end) - 1 ) );
        imgNum     = str2double( location{1,1}( slashPlace(end) + 1 : dotPlace(1) - 1 ) );
        depth = imread( sprintf( '%s%d%s%d%s', basePathDTH, classNum, '/', imgNum, '.png' ) );
        
        depth = mat2gray(depth);
        depth = imresize( depth, [150 150] );
        
        depthTemp = extractHOGFeatures( depth, 'CellSize', cellSize );
        
        features(j, :) = [featTemp depthTemp];
    end

    labels = repmat( str2num(validationSetsRGB(i).Description), numImages, 1 );
    
    validationFeatures = [validationFeatures; features];
    validationLabels   = [validationLabels;   labels  ];
end

predictedLabels = predict(classifier, validationFeatures);

confMat = confusionmat(validationLabels, predictedLabels);

percent = mean(diag(confMat));

save( sprintf( '%s%d', 'classifiers/hog_rgbd_raw_', percent ), 'classifier' );


fileID = fopen( sprintf( '%s%d', 'results/hog_rgbd_raw_res_', percent ), 'wt+' );

fprintf( fileID, '%s\n', 'Id, Category' );

load('testID.mat');

for i = 1 : size( testID, 1 )
    
    img   = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/rgb/'  , i, '.png' ) );
    depth = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/depth/', i, '.png' ) );
    
    imshow( img );
    
    img = rgb2gray( img );
    img = imresize( img, [150 150] );
    
    depth = mat2gray( depth );
    depth = imresize( depth, [150 150] );
    
    feature   = zeros( 1, hogFeatureSize, 'single' );
    featTemp  = extractHOGFeatures( img, 'CellSize', cellSize );
    depthTemp = extractHOGFeatures( depth, 'CellSize', cellSize );
        
    feature(1, :) = [featTemp depthTemp];

    predictedLabels = predict(classifier, feature)
  
    fprintf(fileID,'%s%s%d\n', testID{i,1}, ',', predictedLabels );
end

fclose(fileID);
