from functools import reduce
from operator import mul
import numpy as np
from scipy.sparse import csr_matrix

def convmtxn(F, sz, valid):
#CONVMTXN   N-D convolution matrix
#   M = CONVMTXN(F, SZ[, VALID]) returns the convolution matrix for the
#   matrix F.  SZ gives the size of the array that the convolution should
#   be applied to.  The returned matrix M is sparse.
#   If X is of size SZ, then reshape(M * X(:), SZ + size(F) - 1) is the
#   same as convn(X, F).
#   The optional parameter VALID controls which rows of M (corresponding
#   to the pixels of X) should be set to zero to suppress the convolution
#   result.  VALID must have size SZ+size(F)-1.
#  

    ndims = len(sz)
    blksz = reduce(mul, F.shape)
    nblks = reduce(mul, sz)
    nelems = blksz * nblks

    # Build index array for all possible image positions
    tmp = np.zeros(np.array(list(F.shape)) + np.array(list(sz)) - 1)
    sub = np.empty([1, ndims])
    for d in range(0, ndims):
        sub[d] = np.arange(1, sz[d] + 1)
    
    tmp[sub] = 1
    imgpos = np.where(tmp)

    tmp = np.zeros(np.array(list(F.shape)) + np.array(list(sz)) - 1)
    for d in range(0, ndims):
        sub[d] = np.arange(1, F.shape[d] + 1)

    tmp[sub] = 1
    fltpos = np.where(tmp)

    # Continue from here
    rows = np.tile(imgpos.getH(), (blksz, 1)).reshape((nelems, 1)) + np.tile(fltpos - 1, (nblks, 1))
    cols = np.tile(np.arange(1, nblks + 1), (blksz, 1)).reshape((nelems, 1))
    vals = np.tile(F, (nblks, 1))

    if valid is not None:
        valid_idx = valid[rows]
        rows = rows[valid_idx]
        cols = cols[valid_idx]
        vals = vals[valid_idx]

    M_val = reduce(mul, np.array(list(F.shape)) + np.array(list(sz)) - 1))
    M = csr_matrix((vals, (rows, cols)), shape=(M_val, nblks))

    return M