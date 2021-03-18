import numpy as np
from functools import reduce
from operator import mul
import math

def psnr(x, y):
    # Image value should be in range [0 255]
    diff = np.substract(x.astype(np.double), y.astype(np.double))
    n = 20 * math.log10(255 / np.std(diff, axis=0))

    return n