function y = obs_for_SR(x, k,  sigma, zoom)
% forward observation model
% Haichao Zhang
% 2011-8-6 19:27:
r = zoom;

%y = conv2(x, k/sigma,'valid');

y = real(ifft2(fft2(x).* psf2otf(k/sigma, size(x))));

y = y(1:r:end, 1:r:end);
