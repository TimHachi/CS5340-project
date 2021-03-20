from pypher.pypher import psf2otf
import numpy as np

def obs_for_SR(x, k, sigma, zoom):
    y = np.real(np.fft.ifft2(np.multiply(np.fft.fft2(x), psf2otf(k/sigma, x.shape))))
    y = y[0::zoom, 0::zoom]

    return y