basePathRGB = '/home/vahid/RL_project_1/data/train_class/rgb_class/';
basePathDTH = '/home/vahid/RL_project_1/data/train_class/rgbdepth_class/';
basePathRaw = '/home/vahid/RL_project_1/data/train_class/depth_class/';

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

% depth = imread( sprintf( '%s%d%s%d%s', basePathRaw, classNum, '/', imgNum, '.png' ) );
% [surfD  , ~] = get_surf_rgb(depth);

% depth = mat2gray( depth );
% depth = imresize( depth, [150 150] );
% 
% [hog_4x4D, vis4x4D] = extractHOGFeatures( depth, 'CellSize', [4 4] );


img = read( trainingSetsRGB(1), 1 );
[scores, ~] = classification_demo(img, 1, fc);
% [surf  , ~] = get_surf_rgb(img);
% img = imresize( img, [150 150] );
% 
% [hog_4x4, vis4x4] = extractHOGFeatures( img, 'CellSize', [4 4] );


featureSize = length( scores ) + length( scoresD );% + length( hog_4x4 ) + length( hog_4x4D ) + length( surfD(:) ) + length( surf(:) );


for i = 1 : numel( trainingSetsRGB )
    
    numImages = trainingSetsRGB( i ).Count;
    features  = zeros( numImages, featureSize );
    
    for j = 1 : numImages

        img = read( trainingSetsRGB(i), j );
        [scores, ~] = classification_demo(img, 1, fc);
%         [surf  , ~] = get_surf_rgb(img);
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
        
%         depth = imread( sprintf( '%s%d%s%d%s', basePathRaw, classNum, '/', imgNum, '.png' ) );
%         [surfD  , ~] = get_surf_rgb(depth);
%         
%         depth = mat2gray( depth );
%         depth = imresize( depth, [150 150] );
% 
%         hog_4x4D = extractHOGFeatures( depth, 'CellSize', [4 4] );
        
        features(j, :) = [scores' scoresD'];% surf(:)' surfD(:)' hog_4x4 hog_4x4D];  
    end

    labels = repmat( str2num(trainingSetsRGB(i).Description), numImages, 1 );
    
    save( sprintf( '%s%s%d%s', 'features', '_', i, '.mat'), 'features', '-v7.3');
    save( sprintf( '%s%s%d%s', 'labels'  , '_', i, '.mat'), 'labels'  , '-v7.3');
    
    SVMModel = fitcsvm(features,labels);
    
    save( sprintf( '%s%s%d', 'SVMModel'  , '_', i), 'SVMModel' , '-v7.3' );
 
end
%%% done to here

for i = 1 : numel( trainingSetsRGB )
    
    currentSVM = load( sprintf( '%s%s%d%s', 'SVMModel', '_', i, '.mat') );
    
    falsePositiveF = [];
    falsePositiveL = [];
    
    for j = 1 : numel( trainingSetsRGB )
        
        if i == j 
            continue;
        end
        
        currentFeats = load( sprintf( '%s%s%d%s', 'features', '_', j, '.mat') );
        currentlbls  = load( sprintf( '%s%s%d%s', 'labels'  , '_', j, '.mat') );
        
        count = 0;
        
        for k = 1 : size( currentFeats.features, 1 )
            
            [~, score] = predict( currentSVM.SVMModel, currentFeats.features(k,:) );
            
            count = count + 1;
            
            if score > 0 
                
                falsePositiveF(count,:) = currentFeats.features(k,:);
                falsePositiveL(count,:) = currentlbls.labels(k,:);
                
            end
           
        end
        
    end
    
    save( sprintf( '%s%s%d%s', 'falsePositiveF', '_', i, '.mat'), 'falsePositiveF', '-v7.3' );
    
end

%done until here

for i = 1 : numel( trainingSetsRGB )
    
    realFeatures  = load( sprintf( '%s%s%d%s', 'features'      , '_', i, '.mat') );
    realLbls      = load( sprintf( '%s%s%d%s', 'labels'        , '_', i, '.mat') );
    falseFeatures = load( sprintf( '%s%s%d%s', 'falsePositiveF', '_', i, '.mat') );
    
    trainingFeatures = [realFeatures.features falseFeatures];
    trainingLables{1, 1:length(realLbls) }   = 'true'; 
    trainingLables{length(realLbls)+1, 1:length(falseFeatures) }   = 'false'; 
    
    SVMModel = fitcsvm( trainingFeatures, trainingLabels );
    
    save( sprintf( '%s%s%d%s', 'SVMModelFinal'  , '_', i, '.mat'), 'SVMModel', '-v7.3' );
end


% validationResult   = [];
% validationLabels   = [];
% 
% for i = 1 : numel( validationSetsRGB )
%     
%     numImages = validationSetsRGB( i ).Count;
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
%         [surfD  , ~] = get_surf_rgb(depth);
%         depth = imresize( depth, [150 150] );
% 
%         hog_4x4D = extractHOGFeatures( depth, 'CellSize', [4 4] );
%         
%         features = [scores' scoresD' surf(:)' surfD(:)' hog_4x4 hog_4x4D];
%         
%         [~ score1] = SVMModelF_1(features);
%         [~ score2] = SVMModelF_2(features);
%         [~ score3] = SVMModelF_3(features);
%         [~ score4] = SVMModelF_4(features);
%         [~ score5] = SVMModelF_5(features);
%         [~ score6] = SVMModelF_6(features);
%         [~ score7] = SVMModelF_7(features);
%         [~ score8] = SVMModelF_8(features);
%         [~ score9] = SVMModelF_9(features);
%         [~ score10] = SVMModelF_10(features);
%         [~ score11] = SVMModelF_11(features);
%         [~ score12] = SVMModelF_12(features);
%         
%         result = [ score1 score2 score3 score4 score5 score6 score7 score8 score9 score10 score11 score12];
%     end
% 
%     labels = repmat( str2num(validationSetsRGB(i).Description), numImages, 1 );
%     
%     validationResults = [validationResults; features];
%     validationLabels  = [validationLabels;   labels  ];
%     
% en


load('testID');

SVMModelF_1  = load('SVMModel_1.mat');
SVMModelF_2  = load('SVMModel_2.mat');
SVMModelF_3  = load('SVMModel_3.mat');
SVMModelF_4  = load('SVMModel_4.mat');
SVMModelF_5  = load('SVMModel_5.mat');
SVMModelF_6  = load('SVMModel_6.mat');
SVMModelF_7  = load('SVMModel_7.mat');
SVMModelF_8  = load('SVMModel_8.mat');
SVMModelF_9  = load('SVMModel_9.mat');
SVMModelF_10 = load('SVMModel_10.mat');
SVMModelF_11 = load('SVMModel_11.mat');
SVMModelF_12 = load('SVMModel_12.mat');

for i = 1 : size( testID, 1 )
    
    img   = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/rgb/'  , i, '.png' ) );
    depth = imread( sprintf( '%s%d%s', '/home/vahid/RL_project_1/data/extracted_test/rgbdepth/', i, '.png' ) );
    
    [scores , ~] = classification_demo(img, 1, fc);
    [scoresD, ~] = classification_demo(depth, 1, fc);
    
    features  = zeros( 1, featureSize );
    features(1, :) = [scores' scoresD'];
    
        [~, score1] = predict( SVMModelF_1.SVMModel, features);
        [~, score2] = predict( SVMModelF_2.SVMModel, features);
        [~, score3] = predict( SVMModelF_3.SVMModel, features);
        [~, score4] = predict( SVMModelF_4.SVMModel, features);
        [~, score5] = predict( SVMModelF_5.SVMModel, features);
        [~, score6] = predict( SVMModelF_6.SVMModel, features);
        [~, score7] = predict( SVMModelF_7.SVMModel, features);
        [~, score8] = predict( SVMModelF_8.SVMModel, features);
        [~, score9] = predict( SVMModelF_9.SVMModel, features);
        [~, score10] = predict( SVMModelF_10.SVMModel, features);
        [~, score11] = predict( SVMModelF_11.SVMModel, features);
        [~, score12] = predict( SVMModelF_12.SVMModel, features);
        
        total = [score1 score2 score3 score4 score5 score6 score7 score8 score9 score10 score11 score12];
        maxtotal= max(total);
  
    %fprintf(fileID,'%s%s%d\n', testID{i,1}, ',', predictedLabels );
end
