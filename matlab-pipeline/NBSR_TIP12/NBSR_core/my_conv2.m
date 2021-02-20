function y = my_conv2(img,k, flag)
% Haichao Zhang
% 2011-8-8 11:53:02
if(~exist('flag', 'var'))
    flag = 1;
end
    
if(flag==1) % convolution
    y = real(ifft2(fft2(img).* (psf2otf(k, size(img)))));
elseif(flag == 2) % correlation
    y = real(ifft2(fft2(img).* conj(psf2otf(k, size(img)))));
end