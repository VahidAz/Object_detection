basePathRGB = '/home/vahid/RL_project_1/data/train_class/rgb_class/';

imgSetsRGB = imageSet( basePathRGB, 'recursive' );

[trainingSetsRGB, validationSetsRGB] = partition( imgSetsRGB, 0.8, 'randomize' );

counter = 0;

fileID = fopen( 'train.txt', 'wt+' );
    
for i = 1 : numel( trainingSetsRGB )
    
    numImages = trainingSetsRGB( i ).Count;
    
    for j = 1 : numImages

        img = read( trainingSetsRGB(i), j );
        counter = counter + 1;
        
        imwrite( img , sprintf( '%s%d%s', 'train/', counter, '.png') );
        
        fprintf(fileID,'%s%d%s%d\n', '/home/vahid/caffe/matlab/demo/train/', counter, '.png ',  str2num( trainingSetsRGB(i).Description ) );
    end
end

fclose(fileID);

fileID = fopen( 'val.txt', 'wt+' );

counter = 0;

for i = 1 : numel( validationSetsRGB )
    
    numImages = validationSetsRGB( i ).Count;
    
    for j = 1 : numImages

        img = read( validationSetsRGB(i), j );
        counter = counter + 1;
        
        imwrite( img , sprintf( '%s%d%s', 'val/', counter, '.png') );
        
        fprintf(fileID,'%s%d%s%d\n', '/home/vahid/caffe/matlab/demo/val/', counter, '.png ',  str2num( validationSetsRGB(i).Description ) );
   
    end
end

fclose(fileID);