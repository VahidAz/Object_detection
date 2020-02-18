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

[feat1 met1] = get_surf_rgb( depth );
        
img = read( trainingSetsRGB(1), 1 );
[feat met] = get_surf_rgb( img );

hogFeatureSize = length( feat(:) ) + length( feat1(:) );


trainingFeatures = [];
trainingLabels   = [];

for i = 1 : numel( trainingSetsRGB )
    
    numImages = trainingSetsRGB( i ).Count;
    features  = zeros( numImages, hogFeatureSize );

    for j = 1 : numImages

        img = read( trainingSetsRGB(i), j );
        [feat met] = get_surf_rgb( img );
        
        location   = trainingSetsRGB(i).ImageLocation(j);
        slashPlace = find( location{1,1} == '/' );
        dotPlace   = find( location{1,1} == '.' );
        classNum   = str2double( location{1,1}( slashPlace(end-1) + 1 : slashPlace(end) - 1 ) );
        imgNum     = str2double( location{1,1}( slashPlace(end) + 1 : dotPlace(1) - 1 ) );
        depth = imread( sprintf( '%s%d%s%d%s', basePathDTH, classNum, '/', imgNum, '.png' ) );
        

        [feat1 met1] = get_surf_rgb( depth );
        
        features(j, :) = [feat(:)' feat1(:)'];
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
    features  = zeros( numImages, hogFeatureSize );

    for j = 1 : numImages

        img = read( validationSetsRGB(i), j );
        [feat met] = get_surf_rgb( img );
            
        location   = validationSetsRGB(i).ImageLocation(j);
        slashPlace = find( location{1,1} == '/' );
        dotPlace   = find( location{1,1} == '.' );
        classNum   = str2double( location{1,1}( slashPlace(end-1) + 1 : slashPlace(end) - 1 ) );
        imgNum     = str2double( location{1,1}( slashPlace(end) + 1 : dotPlace(1) - 1 ) );
        depth = imread( sprintf( '%s%d%s%d%s', basePathDTH, classNum, '/', imgNum, '.png' ) );
        
        [feat1 met1] = get_surf_rgb( depth );
        
        features(j, :) = [feat(:)' feat1(:)'];
    end

    labels = repmat( str2num(validationSetsRGB(i).Description), numImages, 1 );
    
    validationFeatures = [validationFeatures; features];
    validationLabels   = [validationLabels;   labels  ];
end

predictedLabels = predict(classifier, validationFeatures);

confMat = confusionmat(validationLabels, predictedLabels);

percent = mean(diag(confMat));

save(  sprintf('%s%d', 'classifiers/surf_rgbd_raw_', percent), 'classifier' );


fileID = fopen( sprintf('%s%d', 'results/surf_rgbd_res_raw_', percent), 'wt+' );

fprintf( fileID, '%s\n', 'Id, Category' );

load('testID.mat');

for i = 1 : size( testID, 1 )
    
    img   = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/rgb/'  , i, '.png' ) );
    depth = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/depth/', i, '.png' ) );
    
    imshow(img);
    
    [feat met] = get_surf_rgb( img );
    
    [feat1 met1] = get_surf_rgb( depth );
    
    feature  = zeros( 1, hogFeatureSize );
        
    feature(1, :) = [feat(:)' feat1(:)'];

    predictedLabels = predict(classifier, feature)
  
    fprintf(fileID,'%s%s%d\n', testID{i,1}, ',', predictedLabels );
end

fclose(fileID);
