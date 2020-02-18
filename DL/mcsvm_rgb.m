clear
clc

trainingFeatures  = load('trainingFeatures_rgb.mat');
trainingLabels    = load('trainingLables_rgb.mat');

validationFeatures  = load('validationFeatures_rgb.mat');
validationLabels    = load('validationLabels_rgb.mat');

classifier = fitcecoc(trainingFeatures, trainingLabels);

predictedLabels = predict(classifier, validationFeatures);

confMat = confusionmat(validationLabels.validationLabels, predictedLabels);

percent = mean(diag(confMat));


save( sprintf('%s%d', 'classifiers/mcsvm_rgb_', percent), 'classifier' );
    

fileID = fopen( sprintf('%s%d', 'results/mcsvm_res_rgb_all_data', percent), 'wt+' );

fprintf( fileID, '%s\n', 'Id, Category' );

testFeatures    = load('testFeatures_rgb.mat');
load('testID');

for i = 1 : size( testID, 1 )
    predictedLabels = predict(classifier, testFeatures(i,:) )
  
    fprintf(fileID,'%s%s%d\n', testID{i,1}, ',', predictedLabels );
end

fclose(fileID);