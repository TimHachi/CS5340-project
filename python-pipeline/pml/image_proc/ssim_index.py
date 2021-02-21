import math
import numpy as np
from scipy import signal

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def mean2(x):
    y = np.sum(x) / np.size(x)
    return y


def ssim_index(img1, img2, K=[], window=None, L=None):
#Input : (1) img1: the first image being compared
#        (2) img2: the second image being compared
#        (3) K: constants in the SSIM index formula (see the above
#            reference). defualt value: K = [0.01 0.03]
#        (4) window: local window for statistics (see the above
#            reference). default widnow is Gaussian given by
#            window = fspecial('gaussian', 11, 1.5);
#        (5) L: dynamic range of the images. default: L = 255
#
#Output: (1) mssim: the mean SSIM index value between 2 images.
#            If one of the images being compared is regarded as 
#            perfect quality, then mssim can be considered as the
#            quality measure of the other image.
#            If img1 = img2, then mssim = 1.
#        (2) ssim_map: the SSIM index map of the test image. The map
#            has a smaller size than the input images. The actual size:
#            size(img1) - size(window) + 1.
#
#Default Usage:
#   Given 2 test images img1 and img2, whose dynamic range is 0-255
#   [mssim ssim_map] = ssim_index(img1, img2);
#
#Advanced Usage:
#   User defined parameters. For example
#   K = [0.05 0.05];
#   window = ones(8);
#   L = 100;
#   [mssim ssim_map] = ssim_index(img1, img2, K, window, L);
#
#See the results:
#   mssim                        #Gives the mssim value
#   imshow(max(0, ssim_map).^4)  #Shows the SSIM index map
    if img1.shape != img2.shape:
        return -math.inf, -math.inf

    M, N = img1.shape

    if len(K) == 0:
        if M < 11 and N < 11:
            return -math.inf, -math.inf

        window = matlab_style_gauss2D((11,11), 1.5)
        K = [0.01, 0.03]
        L = 255

    elif len(K) == 2:
        if K[0] < 0 or K[1] < 0:
                return -math.inf, -math.inf

        if window is None and L is None:
            if M < 11 and N < 11:
                return -math.inf, -math.inf
                window = matlab_style_gauss2D((11,11), 1.5)
                L = 255

        elif L is None and window is not None:
            H, W = window.shape
            if ((H*W) < 4 or (H > M) or (W > N)):
                return -math.inf, -math.inf
            L = 255

        elif L is not None and window is not None:
            H, W = window.shape
            if ((H*W) < 4 or (H > M) or (W > N)):
                return -math.inf, -math.inf

    else:
        return -math.inf, -math.inf

    C1 = (K[0] * L)^2
    C2 = (K[1] * L)^2
    window = window/sum((window).sum(axis=0))
    img1 = img1.astype(np.double)
    img2 = img2.astype(np.double)

    mu1 = signal.convolve2d(window, img1, mode='valid')
    mu2 = signal.convolve2d(window, img2, mode='valid')
    mu1_sq = np.multiply(mu1, mu1)
    mu2_sq = np.multiply(mu2, mu2)
    mu1_mu2 = np.multiply(mu1, mu2)
    sigma1_sq = signal.convolve2d(window, np.multiply(img1, img1), mode='valid') - mu1_sq
    sigma2_sq = signal.convolve2d(window, np.multiply(img2, img2), mode='valid') - mu2_sq
    sigma12 = signal.convolve2d(window, np.multiply(img1, img2), mode='valid') - mu1_mu2

    if (C1 > 0 and C2 > 0):
        mat_1 = np.multiply((2*mu1_mu2 + C1), (2*sigma12 + C2))
        mat_2 = np.multiply((mu1_sq + mu2_sq + C1), (sigma1_sq + sigma2_sq + C2))
        ssim_map = np.divide(mat_1, mat_2)
    else:
        numerator1 = 2*mu1_mu2 + C1
        numerator2 = 2*sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        ssim_map = np.ones(mu1.shape)
        index = (np.multiply(denominator1, denominator2) > 0)
        ssim_map[index] = np.divide(np.multiply(numerator1[index], numerator2[index]), np.multiply(denominator1[index], denominator2[index]))
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = np.divide(numerator1[index], denominator1[index])

    mssim = mean2(ssim_map)

    return mssim, ssim_map