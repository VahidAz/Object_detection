function [features, featureMetrics] = get_surf_rgb(I)

    [height, width, numChannels] = size(I);
    
    if numChannels > 1
        grayImage = rgb2gray(I);
    elseif numChannels == 1
        grayImage = mat2gray(I);
    else
        grayImage = I;
    end
    
    %imshow(I);
    grayImage = imresize( grayImage, [150 150] );
    [height, width, numChannels] = size(grayImage);

    % Define a regular grid over I.
    gridStep = 8; % in pixels
    gridX = 1 : gridStep : width;
    gridY = 1 : gridStep : height;

    [x,y] = meshgrid(gridX, gridY);

    gridLocations = [x(:) y(:)];

    multiscaleGridPoints = [SURFPoints(gridLocations, 'Scale', 1.6); 
                            SURFPoints(gridLocations, 'Scale', 3.2);
                            SURFPoints(gridLocations, 'Scale', 4.8);
                            SURFPoints(gridLocations, 'Scale', 6.4)];
                    

    features = extractFeatures(grayImage, multiscaleGridPoints,'Upright',true);

    featureMetrics = var(features,[],2);

end
