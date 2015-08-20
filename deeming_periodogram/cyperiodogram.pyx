#
# Cython version of the Deeming periodogram function. 
# See deeming.py
#


import numpy as np
cimport numpy as np
cimport cython

cdef extern from 'math.h':
    double sin(double theta)
    double cos(double theta)


DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.cdivision(True)
@cython.boundscheck(False)
def periodogram(np.ndarray[DTYPE_t, ndim=1]  t not None, \
                np.ndarray[DTYPE_t, ndim=1]  m not None,\
                np.ndarray[DTYPE_t, ndim=1]  freqs not None):

    cdef DTYPE_t realpart = 0.0
    cdef DTYPE_t imagpart = 0.0
    cdef DTYPE_t pi = np.pi

    cdef int Nf = freqs.shape[0]
    cdef int Nt = t.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=1]  amps = np.zeros([Nf], dtype=DTYPE)
    cdef int i, j

    for i in range(Nf):
        for j in range(Nt):
            realpart = realpart + m[j]*cos(2.0*pi*freqs[i]*t[j])
            imagpart = imagpart + m[j]*sin(2.0*pi*freqs[i]*t[j])

        amps[i] = 2.0*(realpart**2 + imagpart**2)**0.5/Nt

        realpart = 0.0
        imagpart = 0.0

    return amps

