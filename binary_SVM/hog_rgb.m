clear;
clc;

basePath = '/home/vahid/RL_project_1/data/train_class/rgb_class/';

imgSets = imageSet( basePath, 'recursive' );

[trainingSets, validationSets] = partition( imgSets, 0.8, 'randomize' );
      
trainingFeatures = [];
trainingLabels   = [];

img = read( trainingSets(3), 4 );
img = imresize( img, [150 150] );
img = rgb2gray( img );

[hog_2x2, vis2x2] = extractHOGFeatures( img, 'CellSize', [2 2] );
[hog_4x4, vis4x4] = extractHOGFeatures( img, 'CellSize', [4 4] );
[hog_8x8, vis8x8] = extractHOGFeatures( img, 'CellSize', [8 8] );

cellSize = [4 4];
hogFeatureSize = length( hog_4x4 );


for i = 1 : numel( trainingSets )
    
    numImages = trainingSets( i ).Count;
    features  = zeros( numImages, hogFeatureSize, 'single' );

    for j = 1 : numImages

        img = read( trainingSets(i), j );
        img = imresize( img, [150 150] );
        img = rgb2gray( img );
        
        features(j, :) = extractHOGFeatures( img, 'CellSize', cellSize );
    end

    labels = repmat( str2num(trainingSets(i).Description), numImages, 1 );
    
    trainingFeatures = [trainingFeatures; features];
    trainingLabels   = [trainingLabels;   labels  ];
end


classifier = fitcecoc(trainingFeatures, trainingLabels);


validationFeatures = [];
validationLabels   = [];

for i = 1 : numel( validationSets )
    
    numImages = validationSets( i ).Count;
    features  = zeros( numImages, hogFeatureSize, 'single' );

    for j = 1 : numImages

        img = read( validationSets(i), j );
        img = imresize( img, [150 150] );
        img = rgb2gray( img );

        features(j, :) = extractHOGFeatures( img, 'CellSize', cellSize );
    end

    labels = repmat( str2num(validationSets(i).Description), numImages, 1 );
    
    validationFeatures = [validationFeatures; features];
    validationLabels   = [validationLabels;   labels  ];
end

predictedLabels = predict(classifier, validationFeatures);

confMat = confusionmat(validationLabels, predictedLabels);

percent = mean(diag(confMat));


save( sprintf('%s%d', 'classifiers/hog_rgb_', percent), 'classifier' );
    

fileID = fopen( sprintf('%s%d', 'results/hog_res_', percent), 'wt+' );

fprintf( fileID, '%s\n', 'Id, Category' );

load('testID.mat');

for i = 1 : size( testID, 1 )
    
    img = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/rgb/', i, '.png' ) );
    
    imshow(img);
    
    img = imresize( img, [150 150] );
    img = rgb2gray( img );
    
    feature  = zeros( 1, hogFeatureSize, 'single' );
    feature(1, :) = extractHOGFeatures( img, 'CellSize', cellSize );

    predictedLabels = predict(classifier, feature)
  
    fprintf(fileID,'%s%s%d\n', testID{i,1}, ',', predictedLabels );
end

fclose(fileID);

%% hog with SVM only RGB
%% 72     4*4  rgbgray  resize   100*100
%% 73.58  4*4  resize   rgbgray  100*100
%% 76.16  4*4  resize   rgbgray  150*150 <--- 3
%% 74.58  4*4  resize   rgbgray  200*200
%% 69.58  2*2  resize   rgbgray  100*100
%% 73.58  8*8  resize   rgbgray  100*100