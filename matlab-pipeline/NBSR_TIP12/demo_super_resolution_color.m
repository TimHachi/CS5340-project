%==========================================================================
% * demo super-resolution code for the method described in TIP12 paper *
%
%   Haichao Zhang, Yanning Zhang, Haisen Li and Thomas S. Huang, 
%   Generative Bayesian Image Super-Resolution with Natural Image Prior, 
%   IEEE Trans. on Image Processing (IEEE TIP), Vol. 21, No. 9, pp.4054-4067, 2012

% Haichao Zhang
% 2011-8-6
%==========================================================================


 addpath(genpath('.'))

  
 % load learned image prior
 mrf = learned_models.cvpr_pw_mrf;  % pairwise MRF 
 % mrf = learned_models.cvpr_3x3_foe; % 3x3 FoE
  
  sigma = 2; % choose noise std

  
  img_names = {'1', '2'};
  idx = 1;
  
  img_clean = double(imread(sprintf('images/%s.bmp', img_names{idx})));
  
  %img_clean = imresize(img_clean, sc);


    
 
  zoom = 4;
  
  if(zoom==2)
      k = fspecial('gaussian', 7,1);
  elseif(zoom==3)
      k = fspecial('gaussian', 7,2); 
      H = fspecial('gaussian', 7,2); 
  elseif(zoom==4)
      k = fspecial('gaussian', 7,2);
  end
  
  img_sz = size(img_clean);
  nb = img_sz(3);
  img_sz = img_sz(1:2); % remove band number
  n_img_sz = zoom*floor(img_sz/zoom);
  
  img_clean = imresize(img_clean, n_img_sz);  
  
  img_blurred = [];
  for b =1:nb
    img_blurred = cat(3, img_blurred, obs_for_SR(img_clean(:,:,b), k, 1, zoom)); % blur image
  end
%    img_blurred = conv2(img_clean, k, 'valid'); % blur image
  
  
 
  img_blurred_rgb = img_blurred + sigma * randn(size(img_blurred)); % add noise
  % keep only part of the ground truth image (same size as blurred image)
  
  
  %% transform color space
  img_ycbcr = rgb2ycbcr(uint8(img_blurred_rgb)); 
  img_blurred = double(img_ycbcr(:,:,1));
  
  img_clean_ycbcr = rgb2ycbcr(uint8(img_clean)); 
  
    BI = imresize(img_ycbcr, size(img_clean(:,:,1))); 
  NN = imresize(img_ycbcr, size(img_clean(:,:,1)), 'nearest'); 
  
  
  BI_rgb = imresize(img_blurred_rgb, size(img_clean(:,:,1))); 
   
  NN_rgb = imresize(img_blurred_rgb, size(img_clean(:,:,1)), 'nearest'); 
   imwrite(uint8(img_blurred_rgb), ['LR.bmp'])
  imwrite(uint8(img_clean), ['GT.bmp'])
  

  
  b = (size(k,1)-1)/2; 

max_iters = 200;
  [img_deblurred, sigma_est, psnr, ssim] = NBSR(zoom,mrf, img_blurred, k,  1, double(img_clean_ycbcr(:,:,1)), max_iters);
  
  img_deblurred_rgb = ycbcr2rgb(uint8(cat(3, img_deblurred, BI(:,:,2:3))));

  
  %% Eval
    [pv, mse] = my_psnr(double(img_deblurred),double(img_clean_ycbcr(:,:,1)));
    ssim = ssim_index(double(img_deblurred), double(img_clean_ycbcr(:,:,1)))
  
  
 imwrite(uint8(img_deblurred_rgb), ['HR_PSNR' num2str(pv) '_SSIM' num2str(ssim) '.bmp'])
