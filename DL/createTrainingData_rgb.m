function [] = createTrainingData_rgb( percent, fc)

basePathRGB = '/home/vahid/RL_project_1/data/train_class/rgb_class/';

imgSetsRGB = imageSet( basePathRGB, 'recursive' );

[trainingSetsRGB, validationSetsRGB] = partition( imgSetsRGB, percent, 'randomize' );


img = read( trainingSetsRGB(1), 1 );
[scores, ~] = classification_demo(img, 1, fc);

featureSize = length( scores );


trainingFeatures = [];
trainingLabels   = [];


for i = 1 : numel( trainingSetsRGB )
    
    numImages = trainingSetsRGB( i ).Count;
    features  = zeros( numImages, featureSize );
    
    for j = 1 : numImages

        img = read( trainingSetsRGB(i), j );
        [scores, ~] = classification_demo(img, 1, fc);
        
        features(j, :) = [scores'];
    end

    labels = repmat( str2num(trainingSetsRGB(i).Description), numImages, 1 );
    
    trainingFeatures = [trainingFeatures; features];
    trainingLabels   = [trainingLabels;   labels  ];
end


validationFeatures = [];
validationLabels   = [];


for i = 1 : numel( validationSetsRGB )
    
    numImages = validationSetsRGB( i ).Count;
    features  = zeros( numImages, featureSize );

    for j = 1 : numImages

        img = read( validationSetsRGB(i), j );
        [scores, ~] = classification_demo(img, 1, fc);
        
        features(j, :) = [scores'];
    end

    labels = repmat( str2num(validationSetsRGB(i).Description), numImages, 1 );
    
    validationFeatures = [validationFeatures; features];
    validationLabels   = [validationLabels;   labels  ];
end


load('testID.mat');

testFeatures = [];

for i = 1 : size( testID, 1 )
    
    img = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/rgb/', i, '.png' ) );
    
    
    [scores, ~] = classification_demo(img, 1, fc);
        
    testFeatures(i, :) = scores';
end


save( sprintf( '%s%d%d','trainingFeatures_rgb_'  , fc, percent ) , 'trainingFeatures'   );
save( sprintf( '%s%d%d','trainingLables_rgb_'    , fc, percent ) , 'trainingLabels'     );
save( sprintf( '%s%d%d','validationFeatures_rgb_', fc, percent ) , 'validationFeatures' );
save( sprintf( '%s%d%d','validationLabels_rgb_'  , fc, percent ) , 'validationLabels'   );
save( sprintf( '%s%d%d','validationLabels_rgb_'  , fc, percent ) , 'validationLabels'   );
save( sprintf( '%s%d%d','testFeatures_rgb_'      , fc, percent ) , 'testFeatures'       );

clearvars;

end
