function [features, featureMetrics] = mser_rgb(I)

    [height, width, numChannels] = size(I);
    
    if numChannels > 1
        grayImage = rgb2gray(I);
    else
        grayImage = I;
    end

    % Define a regular grid over I.
    gridStep = 8; % in pixels
    gridX = 1 : gridStep : width;
    gridY = 1 : gridStep : height;

    [x,y] = meshgrid(gridX, gridY);

    gridLocations = [x(:) y(:)];

    multiscaleGridPoints = detectMSERFeatures(grayImage);

    features = extractFeatures( grayImage, multiscaleGridPoints, 'Upright', true );

    featureMetrics = var(features,[],2);

end
