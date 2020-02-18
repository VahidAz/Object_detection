function [] = linear_SVM(type)

    basePath = '/home/vahid/RL_project_1/data/train_class/';
    class    = 'rgb_class/';
    
    imgSets = [ imageSet( fullfile( basePath, class, '1'  ) ), ...
                imageSet( fullfile( basePath, class, '2'  ) ), ...
                imageSet( fullfile( basePath, class, '3'  ) ), ...
                imageSet( fullfile( basePath, class, '4'  ) ), ...
                imageSet( fullfile( basePath, class, '5'  ) ), ...
                imageSet( fullfile( basePath, class, '6'  ) ), ...
                imageSet( fullfile( basePath, class, '7'  ) ), ...
                imageSet( fullfile( basePath, class, '8'  ) ), ...
                imageSet( fullfile( basePath, class, '9'  ) ), ...
                imageSet( fullfile( basePath, class, '10' ) ), ...
                imageSet( fullfile( basePath, class, '11' ) ), ...
                imageSet( fullfile( basePath, class, '12' ) ), ...
              ];
          
    %minSetCount = min( [imgSets.Count] );

    %imgSets = partition( imgSets, minSetCount, 'randomize' );

    [imgSets.Count];

    [trainingSets, validationSets] = partition( imgSets, 0.8, 'randomize' );

    if strcmp( type, 'surf_rgb' ) == 1
        bag = bagOfFeatures( trainingSets, 'VocabularySize', 7000, 'CustomExtractor', @surf_rgb ); 
    elseif strcmp( type, 'mser_rgb' ) == 1
        bag = bagOfFeatures( trainingSets, 'VocabularySize', 7000, 'CustomExtractor', @mser_rgb );
    elseif strcmp( type, 'hog_rgb' ) == 1
        bag = bagOfFeatures( trainingSets, 'VocabularySize', 7000, 'CustomExtractor', @hog_rgb );
    end

    categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);
    
    confMatrix = evaluate(categoryClassifier, trainingSets);
    
    confMatrix = evaluate(categoryClassifier, validationSets);
    
    percent = mean(diag(confMatrix));
    
    save( sprintf('%s%s%s%d', 'bags/bag_'                , type, '_', percent), 'bag' );
    save( sprintf('%s%s%s%d', 'cClassifiers/cClassifier_', type, '_', percent), 'categoryClassifier' );
   
    load('testID.mat');
    
    fileID = fopen( sprintf('%s%s%s%d', 'results/res_', type, '_', percent), 'wt+' );
    fprintf( fileID, '%s\n', 'Id, Category' );
    
    for i = 1 : size( testID, 1 )
        
        img = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/rgb/', i, '.png' ) );
        
        imshow(img);
    
        [labelIdx, scores] = predict(categoryClassifier, img);
    
        categoryClassifier.Labels(labelIdx)
        
        fprintf(fileID,'%s%s%d\n', testID{i,1}, ',', labelIdx );
    end
    
    fclose(fileID);
end