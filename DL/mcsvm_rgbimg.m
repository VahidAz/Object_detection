clear
clc

trainingFeatures  = load('trainingFeatures_rgbdimg_hog_surf_rgbdimg_1_8.000000e-01.mat');
trainingLabels    = load('trainingLables_rgbdimg_hog_surf_rgbdimg_1_8.000000e-01.mat');

validationFeatures  = load('validationFeatures_rgbdimg_hog_surf_rgbdimg_1_8.000000e-01.mat');
validationLabels    = load('validationLabels_rgbdimg_hog_surf_rgbdimg_1_8.000000e-01.mat');

Options = statset('UseParallel', 1);
classifier = fitcecoc(train.trainingFeatures, lbl.trainingLabels, 'Coding', 'binarycomplete', 'CrossVal', 'on', 'Options', Options);

predictedLabels = predict(classifier, validationFeatures);

confMat = confusionmat(validationLabels, predictedLabels);

percent = mean(diag(confMat));


save( sprintf('%s%d%s', 'classifiers/mcsvm_rgbdimg_surf_1_', percent, '.mat'), 'classifier', '-v7.3' );
    

fileID = fopen( sprintf('%s%d', 'results/mcsvm_res_rgbdimg_surf_1_ternarycomplete_', percent), 'wt+' );

fprintf( fileID, '%s\n', 'Id,Category' );

testFeatures    = load('testFeatures_rgbdimg_hog_surf_rgbdimg_1_8.000000e-01.mat');

load('testID');

for i = 1 : size( testID, 1 )
    predictedLabels = predict(classifier, testFeatures(i,:) )
  
    fprintf(fileID,'%s%s%d\n', testID{i,1}, ',', predictedLabels );
end

fclose(fileID);