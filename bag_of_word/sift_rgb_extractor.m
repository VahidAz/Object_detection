function [features,featureMetrics] = sift_rgb_extractor(I)

I = imresize(I, [100 100]);

    [height,width,numChannels] = size(I);
    
    if numChannels > 1
        grayImage = single(rgb2gray(I)) ;;
    else
        grayImage = I;
    end

    [features,d] = vl_sift(grayImage) ;
    length(features)

    featureMetrics = var(features,[],2);

end