clear;
clc;

basePath = '/home/vahid/RL_project_1/data/train_class/';

imgSets = [ imageSet( fullfile( basePath, 'rgb_class/', '1'  ) ), ...
            imageSet( fullfile( basePath, 'rgb_class/', '2'  ) ), ...
            imageSet( fullfile( basePath, 'rgb_class/', '3'  ) ), ...
            imageSet( fullfile( basePath, 'rgb_class/', '4'  ) ), ...
            imageSet( fullfile( basePath, 'rgb_class/', '5'  ) ), ...
            imageSet( fullfile( basePath, 'rgb_class/', '6'  ) ), ...
            imageSet( fullfile( basePath, 'rgb_class/', '7'  ) ), ...
            imageSet( fullfile( basePath, 'rgb_class/', '8'  ) ), ...
            imageSet( fullfile( basePath, 'rgb_class/', '9'  ) ), ...
            imageSet( fullfile( basePath, 'rgb_class/', '10' ) ), ...
            imageSet( fullfile( basePath, 'rgb_class/', '11' ) ), ...
            imageSet( fullfile( basePath, 'rgb_class/', '12' ) ), ...
          ];
      
%minSetCount = min( [imgSets.Count] );

%imgSets = partition( imgSets, minSetCount, 'randomize' );

[imgSets.Count];

[trainingSets, validationSets] = partition( imgSets, 0.8, 'randomize' );

bag = bagOfFeatures( trainingSets,'VocabularySize', 8000, 'GridStep', [8 8] );

categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);

confMatrix = evaluate(categoryClassifier, trainingSets);

confMatrix = evaluate(categoryClassifier, validationSets);

mean(diag(confMatrix));

%save('78','bag');
%save('categoryClassifier','categoryClassifier');
%load('80.mat');
%load('testID.mat');

fileID = fopen( 'bof_result.txt', 'w' );
fprintf( fileID, '%s\n', 'Id, Category' );

for i = 1 : size( testID, 1 )
    
    img = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/rgb/', i, '.png' ) );
    
    imshow(img);

    [labelIdx, scores] = predict(categoryClassifier, img);

    categoryClassifier.Labels(labelIdx)
    
    fprintf(fileID,'%s%s%d\n', testID{i,1}, ',', labelIdx );
end

fclose(fileID);