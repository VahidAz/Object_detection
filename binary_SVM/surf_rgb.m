clear;
clc;

basePath = '/home/vahid/RL_project_1/data/train_class/rgb_class/';

imgSets = imageSet( basePath, 'recursive' );

[trainingSets, validationSets] = partition( imgSets, 0.8, 'randomize' );
      
trainingFeatures = [];
trainingLabels   = [];

img = read( trainingSets(1), 4 );
[feat met] = get_surf_rgb( img );

hogFeatureSize = length( feat(:) );


for i = 1 : numel( trainingSets )
    
    numImages = trainingSets( i ).Count;
    features  = zeros( numImages, hogFeatureSize, 'single' );

    for j = 1 : numImages

        img = read( trainingSets(i), j );
        [feat met] = get_surf_rgb( img );
        
        features(j, :) = feat(:);

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
        [feat met] = get_surf_rgb( img );
        
        features(j, :) = feat(:);
    end

    labels = repmat( str2num(validationSets(i).Description), numImages, 1 );
    
    validationFeatures = [validationFeatures; features];
    validationLabels   = [validationLabels;   labels  ];
end


predictedLabels = predict(classifier, validationFeatures);

confMat = confusionmat(validationLabels, predictedLabels);

percent = mean(diag(confMat));


save( sprintf('%s%d', 'classifiers/surf_rgb_', percent), 'classifier' );
    

fileID = fopen( sprintf('%s%d', 'results/surf_res_', percent), 'wt+' );

fprintf( fileID, '%s\n', 'Id, Category' );

load('testID.mat');

for i = 1 : size( testID, 1 )
    
    img = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/rgb/', i, '.png' ) );
    
    imshow(img);
    
    [feat met] = get_surf_rgb( img );
        
    feature  = zeros( 1, hogFeatureSize, 'single' );
    feature(1, :) = feat(:);

    predictedLabels = predict(classifier, feature)
  
    fprintf(fileID,'%s%s%d\n', testID{i,1}, ',', predictedLabels );
end

fclose(fileID);
