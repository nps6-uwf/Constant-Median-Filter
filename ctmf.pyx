
# Author: Nick Sebasco
# Date: 04/12/2022
# Constant median filter implementation, based on: Median Filtering in Constant Time - Simon Perreault and Patrick Heber 
# ----------------------------------------------------------------------------------------------------------------------


import numpy as np
from numpy cimport int32_t, uint8_t


cdef uint8_t histogram_median_odd(int32_t[:] H, int count):
    ''' Compute the median from a histogram.  Only valid for count % 2 == 1.  count is the number of elements in the histogram.
    '''
    cdef uint8_t i = 0
    cdef int t = 0
    cdef double c2
    c2 = count / 2
    for i in range(256):
        t = t + H[i]
        if t > c2:
            return i


cdef uint8_t histogram_median_even(int32_t[:] H, int count):
    ''' Compute the median from a histogram.  Only valid if count % 2 == 0.  count is the number of elements in the histogram.
    '''
    cdef int t
    cdef uint8_t iprev, i
    cdef double c2
    t = 0
    iprev = 0
    c2 = count / 2

    for i in range(256):

        t = t + H[i]
        if t > c2:
            if H[i] == 1 or t - H[i] + 1 > c2:
                # output cannot be a float (8-bit int only)
                return (i + iprev)//2
            else:
                return i
            
        if H[i] >= 1:
            iprev = i


def ctmf(I, int r):
    # 1. initialize column histograms (C) and kernel histogram (K).
    # -------------------------------------------------------------
    # Image shape
    cdef int y_max = I.shape[0]
    cdef int x_max = I.shape[1]
    
    H = np.zeros((256,), dtype=np.int32)
    C = np.zeros((x_max, 256), dtype=np.int32)

    # Access pointers to underlying c arrays
    cdef:
        int32_t[:] H_cython_view = H
        int32_t *H_c_integers_array = &H_cython_view[0]

    cdef:
        int32_t[:, :] C_cython_view = C
        int32_t *C_c_integers_array = &C_cython_view[0, 0]
        int32_t[256] *C_c_2d_array = <int32_t[256] *>C_c_integers_array
        

    cdef int H_count = 0
    cdef int c, y, x, i, idx

    # 1.1. initialize column histograms
    for c in range(x_max):
        for y in range(r+1):
            idx = I[y,c]
            C_cython_view[c][idx] += 1

    # 1.2. initialize kernel histogram
    for x in range(r+1):
        for i in range(256):
            H_c_integers_array[i] += C_cython_view[x][i]
            H_count += C_cython_view[x][i]

    # Output image
    # dtype should be uint8
    O = np.empty((y_max,x_max), dtype=np.uint8)

    # The column indices (x) will have the form d*(x)+b.
    cdef int d = -1
    cdef int b = -1
    cdef int bound_start, bound_stop, drx, xmid, i1, i2, i3
    cdef uint8_t m

    # reusing the x,y ints defined above
    for y in range(y_max):
        # update index coefficients
        b = -d+b
        d *= -1

        # [x] if/else removed by using b & d
        bound_start = d*b
        bound_stop = d*(x_max-1)+b 

        for x in range(x_max):

            # when we are travelling in -x direction, make x negative.
            x = x*d+b

            if (H_count & 1):
                m =  histogram_median_odd(H_cython_view, H_count)
            else:
                m = histogram_median_even(H_cython_view, H_count)

            O[y,x] = m
            drx = d*(r+1)+x

            if d*drx < d*(bound_stop + d):
                if y-r > 0: # remove bottom pixel from the r+x column histogram 
                    idx = I[y-r-1,drx]
                    C_cython_view[drx][idx] -= 1
               
                if y+r < y_max and y != 0: # add top right pixel to the r+x column histogram
                    idx = I[y+r,drx]
                    C_cython_view[drx][idx] += 1
                
                # kernel histogram update 1:
                # add the r+x column histogram
                # --------------------------------------------
                for i in range(256):
                    H_cython_view[i] += C_cython_view[drx][i]
                    H_count += C_cython_view[drx][i]
            
            if (d*(x-r)+2*b >= d*bound_start) and (x != bound_stop):
                # kernel histogram update 2:
                # subtract the x-r-1 column histogram
                # --------------------------------------------
                for i in range(256):
                    if H_cython_view[i] >= 1:
                        H_cython_view[i] -= C_cython_view[x-d*r][i]
                        H_count -= C_cython_view[x-d*r][i]
                
            # kernel histogram update 3: changing rows
            if x == bound_stop:
                for i in range(r+1):
                    # add pixels one level up to kernel
                    # minus 1 because the index is one less than the array height.
                    xmid = x + i*(d*-1)
                    if y + r < y_max-1:
                        idx = I[y+r+1,xmid]
                        C_cython_view[xmid][idx] += 1
                        H_cython_view[idx] += 1
                        H_count += 1
                    # drop pixels one level below kernel
                    if y-r >= 0:
                        idx = I[y-r,xmid]
                        C_cython_view[xmid][idx] -= 1
                        H_cython_view[idx] -= 1
                        H_count -= 1

    return O

            