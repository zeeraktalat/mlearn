




import numpy as np
cimport numpy as np
import torch
cimport cython
from libcpp.map cimport map as cpp_map

"""
Uses C++ map containers for fast dict-like behavior with keys being
integers, and values float.
"""
# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t


###############################################################################
# An object to be used in Python

cdef class IntFloatDict:
    cdef cpp_map[ITYPE_t, DTYPE_t] my_map

    def __init__(self, np.ndarray[ITYPE_t, ndim=1] keys,
                       np.ndarray[DTYPE_t, ndim=1] values):
        cdef cpp_map[ITYPE_t,DTYPE_t] my_map
        cdef int i
        cdef int size = values.size
        # Should check that sizes for keys and values are equal, and
        # after should boundcheck(False)
        for i in range(size):
            my_map[keys[i]] = values[i]
        self.my_map = my_map

    def __len__(self):
        return self.my_map.size()

    def __getitem__(self, int key):
        return self.my_map[key]

    def __setitem__(self, int key, str value):
        self.cpp_map[key] = value

    # XXX: Need a __dealloc__ to clear self.cpp_map