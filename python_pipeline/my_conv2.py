from pypher.pypher import psf2otf
import numpy as np

# psf2otf
# Converts point-spread function to optical transfer function.
#     Compute the Fast Fourier Transform (FFT) of the point-spread
#     function (PSF) array and creates the optical transfer function (OTF)
#     array that is not influenced by the PSF off-centering.
#     By default, the OTF array is the same size as the PSF array.
#     To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
#     post-pads the PSF array (down or to the right) with zeros to match
#     dimensions specified in OUTSIZE, then circularly shifts the values of
#     the PSF array up (or to the left) until the central pixel reaches (1,1)
#     position.

def my_conv2(img: np.ndarray, k: np.ndarray, flag: int):
    if flag is None:
        flag = 1

    if flag == 1: # convolution
        y = np.real(np.fft.ifft2(np.multiply(np.fft.fft2(img), psf2otf(k, img.shape))))
    elif flag == 2: # correlation
        y = np.real(np.fft.ifft2(np.multiply(np.fft.fft2(img), np.conj(psf2otf(k, img.shape)))))
    return y