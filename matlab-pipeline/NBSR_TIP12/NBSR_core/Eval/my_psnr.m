function [pv, mse] = my_psnr(img1,img2)
% PSNR Evaluation
%Image value should be in range [0 255]
%2010-1-18 15:28:32
%Haichao Zhang 
err =double( img1)-double(img2);
mse = sum(sum(err.^2))/ prod(size(img1));

%if(max(img1(:))>100)
    pv = 20*log10(255/sqrt(mse));
% else
%     pv = 20*log10(1/sqrt(mse));
% end
