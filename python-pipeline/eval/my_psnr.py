import numpy as np
from functools import reduce
from operator import mul
import math

def my_psnr(img1, img2):
    # Image value should be in range [0 255]
    err = np.substract(img1.astype(np.double), img2.astype(np.double))
    mse = sum((np.power(err, 2)).sum(axis=0) / reduce(mul, img1.shape))

    pv = 20*math.log10(255/math.sqrt(mse))

    return pv, mse