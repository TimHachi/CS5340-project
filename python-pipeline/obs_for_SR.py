from pypher import psf2otf
import numpy as np

def obs_for_SR(x, k, sigma, zoom):
    y = np.real(np.fft.ifft2(np.multiply(x, psf2otf(k/sigma, x.shape))))
    y = y[1:end:zoom, 1:end:zoom]

    return y