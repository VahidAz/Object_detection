function [features, featureMetrics] = extractorFcn(I)

	fc = 1;

        img(:,:,1) = I(:,:,1);
        img(:,:,2) = I(:,:,2);
        img(:,:,3) = I(:,:,3);
                
        depth = I(:,:,4); 

                
       [scores, ~] = classification_demo(img, 1, fc);
       [surf  , ~] = get_surf_rgb(img);


       [Nx,Ny,Nz] = surfnorm(depth);
        
       RGBImg(:,:,1)= Nx;
       RGBImg(:,:,2)= Ny;
       RGBImg(:,:,3)= Nz;
    
       [scoresD, ~] = classification_demo(RGBImg, 1, fc);
       [surfD  , ~] = get_surf_rgb(depth);

       features = [scores' scoresD' surf(:)' surfD(:)'];
                
       featureMetrics = var(features,[],2);

 end
