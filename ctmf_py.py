# Author: Nick Sebasco
# Date: 04/12/2022
# Constant median filter implementation
#

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from statistics import median
import sys
import colorama as crama


crama.init()


# Functions
# ------------------------
def HTML_render(G):
    '''
    A simple custom html display which was used to help visualize small test images.  This should not be expected to work good for large images.
    G is a small grayscale image.
    '''
    from webbrowser import open as w_open
    HTML = '''
    <!DOCTYPE html>
    <html>
        <head>
            <style>
                * {
                    margin: 0; padding: 0;
                }
                table {
                    width: 100vw;
                    height: 100vh;
                    margin: auto;
                    padding: 0;
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <table>
                <tbody>
                    $
                </tbody>
            </table>
        </body>
    </html>
    '''
    rows = []
    for i in range(G.shape[0]):
        cells = []
        for j in range(G.shape[1]):
            td = f"<td style='color:blue;background:rgb({G[i,j]},{G[i,j]},{G[i,j]});'>{G[i,j]}</td>"
            cells.append(td)
        tr = f"<tr>{''.join(cells)}</tr>"
        rows.append(tr)
    print("# rows:",len(rows))
    with open("render.html","w") as f:

        f.write(HTML.replace("$",''.join(rows)))
    w_open("render.html")


def histogram_median_odd(H, count,_max=256):
    ''' Compute the median from a histogram.  Only valid for count % 2 == 1.  count is the number of elements in the histogram.
    '''
    t = 0
    c2 = count/2
    for i in range(_max):
        t += H[i]
        if t > c2:
            return i


def histogram_median_even(H, count,_max=256):
    ''' Compute the median from a histogram.  Only valid if count % 2 == 0.  count is the number of elements in the histogram.
    '''
    t = 0
    iprev = 0
    c2 = count/2
    for i in range(_max):
        t += H[i]
        if t > c2:
            if H[i] == 1 or t - H[i] + 1 > c2:
                return (i + iprev)/2
            else:
                return i
            
        if H[i] >= 1:
            iprev = i


def init_kernels_dev(I, r):
    ''' [DEV] initialize the set of n column histograms (C) and the kernel histogram (H), 
    given an mxn gray scale image (I) and integer kernel radius (r).
    '''
    def get_C_ref(C):
        return [i for i, j in enumerate(C) if j == 1]

    C = []
    C_ref = [[] for _ in range(I.shape[1])]
    H = [0 for _ in range(256)]
    H_ref = []
    H_count, C_count = 0, [0 for i in range(I.shape[1])]
    depth = r+1 # r+1

    # 1) initialize column histograms
    for c in range(I.shape[1]):
        h = [0 for _ in range(256)]
        for y in range(depth):
            h[I[y,c]] += 1
            C_ref[c].append(I[y,c])
            C_count[c] += 1
        C.append(h)

    # 2) initialize kernel histogram
    for x in range(r+1):
        h = C[x]
        for i in range(256):
            H[i] += h[i]
            for _ in range(h[i]):
                H_ref.append(i)
                H_count += 1
    
    return(H, C, H_ref, get_C_ref, H_count, C_count)


def init_kernels(I, r):
    ''' initialize the set of n column histograms (C) and the kernel histogram (H), 
    given an mxn gray scale image (I) and integer kernel radius (r).
    '''

    C = []
    H = [0 for _ in range(256)]
    H_count = 0
    depth = r+1

    # 1) initialize column histograms
    for c in range(I.shape[1]):
        h = [0 for _ in range(256)]
        for y in range(depth):
            h[I[y,c]] += 1

        C.append(h)

    # 2) initialize kernel histogram
    for x in range(r+1):
        h = C[x]
        for i in range(256):
            H[i] += h[i]
            for _ in range(h[i]):
                H_count += 1
    
    return(H, C, H_count)


def ctmf(I, r):
    ''' 
    '''
    H, C, H_count = init_kernels(I, r)
    O = np.empty(shape=I.shape)
    # where the indices (x) will have the form d*(x)+b.
    d = -1
    b = -1

    for y in range(I.shape[0]):
        # update index coefficients
        b = -d+b
        d *= -1

        # [x] if/else removed by using b & d
        bound_start = d*b
        bound_stop = d*(I.shape[1]-1)+b 

        for x in range(I.shape[1]):

            # when we are travelling in -x direction, make x a negative index.
            x = x*d+b
            m =  histogram_median_odd(H, H_count) if (H_count & 1) else histogram_median_even(H, H_count)
            O[y,x] = m
            drx = d*(r+1)+x

            if d*drx < d*(bound_stop + d ):
                if y-r > 0: # remove bottom pixel from the r+x column histogram 
                    C[drx][I[y-r-1,drx]] -= 1
      
                if y+r < I.shape[0] and y != 0: # add top right pixel to the r+x column histogram
                    C[drx][I[y+r,drx]] += 1
                
                # ***kernel histogram update 1***
                # add the r+x column histogram
                # --------------------------------------------
                for i in range(256):
                    H[i] += C[drx][i]
                    H_count += C[drx][i]
            
            if (d*(x-r)+2*b >= d*bound_start) and (x != bound_stop):
                # ***kernel histogram update 2***
                # subtract the x-r-1 column histogram
                # --------------------------------------------
                for i in range(256):
                    if H[i] >= 1:
                        H[i] -= C[x-d*r][i]
                        H_count -= C[x-d*r][i]
                
            # kernel histogram update 3: changing rows
            if x == bound_stop:
                for i in range(r+1):
                    # add pixels one level up to kernel
                    # minus 1 because the index is one less than the array height.
                    xmid = x + i*(d*-1)
                    if y + r < I.shape[0]-1:
                        C[xmid][I[y+r+1,xmid]] += 1
                        H[I[y+r+1,xmid]] += 1
                        H_count += 1
                    # drop pixels one level below kernel
                    if y-r >= 0:
                        C[xmid][I[y-r,xmid]] -= 1
                        H[I[y-r,xmid]] -= 1
                        H_count -= 1

    return O


def ctmf_dev(I, r):
    '''
    04/12: Only working for r = 1
    '''
    H, C, H_ref, C_ref, H_count, C_count = init_kernels_dev(I, r)
    O = np.empty(shape=I.shape)
    #M = {i:0 for i in range(256)}
    d = -1
    b = -1

    no = 0
    for y in range(I.shape[0]):
        # update index coefficients
        # where the indices (z) will have the form d*z+b
        b = -d+b
        d *= -1

        # *** Deliberately killing loop to investigate behavior ***
        # if y == 2: break
        bound_start = d*b #0 if d == 1 else -1
        bound_stop = I.shape[1]-1 if d == 1 else -I.shape[1]

        for x in range(I.shape[1]):
            no += 1 # track total number of iterations

            # when we are travelling in -x direction, make x a negative index.
            x = x*d+b
            print("-"*20)
            print(f"iteration ({no})",(y,x),f"pixel: {I[y,x]}")
            print("-"*20)
            print("bounds:",bound_start, bound_stop, f"coefficients: {(d,b)}")
            print(f"Kernel size (initial): {sum(H)}")
            m =  histogram_median_odd(H, H_count) if (H_count & 1) else histogram_median_even(H, H_count)
            O[y,x] = m
            #M[m] += 1
            print(f"Kernel: {H_ref}",f"count: {H_count}")
            print(f"median (histogram): {m}","median (sorted):",median(H_ref))
            # update column histogram (r+x)

            # (+) [x]
            # (-) [o]
            # in the negative direction additions work, but removals are still not working.
            #
            # -1(-1(1+1)+x) < -1*(-10+1-1) -> 

            if d*(d*(r+1)+x) < d*(bound_stop - b + d + b):

                if y-r > 0: # remove bottom pixel from the r+x column histogram 

                    drx = d*r+x + d

                    print(crama.Fore.MAGENTA+"[-]"+crama.Fore.RESET,I[y-r-1,drx])
                    print(f"C (ref): {C_ref(C[drx])}")
                    #print(f"C: {C[drx]}")
                    C[drx][I[y-r-1,drx]] -= 1
                    C_count[drx] -= 1
                    
                if y+r < I.shape[0] and y != 0: # add top right pixel to the r+x column histogram
                    print(crama.Fore.CYAN+"[+]"+crama.Fore.RESET,I[y+r,d*(r+1)+x])
                    print(f"C (ref): {C_ref(C[d*(r+1)+x])}")
                    C[d*(r+1)+x][I[y+r,d*(r+1)+x]] += 1
                    C_count[d*(r+x+1)] += 1
                
                # ***kernel histogram update 1***
                # add the r+x column histogram
                # --------------------------------------------
                for i in range(256):
                    H[i] += C[d*(r+1)+x][i]
                    H_count += C[d*(r+1)+x][i]
                    
                    # -1 (1 + 1) -x - 1 for x = 8
                    # -2 -8 - 1 = -11
                    for _ in range(C[d*(r+1)+x][i]): 
                        print(crama.Fore.GREEN+f"[+]{i}"+crama.Fore.RESET +" kernel")
                        H_ref.append(i)
                        #H_count += 1

            freq = 0
            
            if (d*(x-r)+2*b >= d*bound_start) and (x != bound_stop):
                # ***kernel histogram update 2***
                # subtract the x-r-1 column histogram
                # --------------------------------------------
                
                for i in range(256):
                    if H[i] >= 1:
                        H[i] -= C[x-d*r][i]
                        H_count -= C[x-d*r][i]
                    for _ in range(C[x-d*r][i]):
                        print(crama.Fore.RED+f"[-]{i}"+crama.Fore.RESET +" kernel")
                        if i in H_ref:
                            H_ref.remove(i)
                        #H_count -= 1

                    freq += C[x-d*r][i]
                
            print("freq",freq)

            # kernel histogram update 3: changing rows
            if x == bound_stop:
                for i in range(r+1):
                    # add pixels one level up to kernel
                    # minus 1 because the index is one less than the array height.
                    if y + r < I.shape[0]-1:
                        C[x + i*(d*-1)][I[y+r+1,x + i*(d*-1)]] += 1
                        H[I[y+r+1,x + i*(d*-1)]] += 1

                        print(crama.Fore.GREEN+"[+]"+crama.Fore.RESET +" kernel")
                        H_ref.append(I[y+r+1,x + i*(d*-1)])
                        H_count += 1
                    # drop pixels one level below kernel
                    if y-r >= 0:
                        print(crama.Fore.LIGHTYELLOW_EX + f"[-] {I[y-r,x + i*(d*-1)]}" + crama.Fore.RESET)
                        C[x + i*(d*-1)][I[y-r,x + i*(d*-1)]] -= 1
                        H[I[y-r,x + i*(d*-1)]] -= 1
                        H_ref.remove(I[y-r,x + i*(d*-1)])
                        H_count -= 1


            print(f"Kernel size (final): {sum(H)}",f"H_count: {H_count}")
    #print(M)
    return O



# Tests
# ------------------------
def Test5():
    '''
    This test will explore large batches of image sizes and make sure that there are no errors for various image sizes for a variable 
    kernel.
    Result:
    > 10,000 tests run and 0 errors.
    '''
    # Image
    # ------------------------
    N = 500
    err_count = 0
    for i in range(N):
        rad1, rad2 = np.random.randint(10,50), np.random.randint(10,50)
        I = np.random.randint(0,255,(rad1,rad2))


        G = np.uint8(I)
        G1 = G.copy()


        # Other variables
        # ------------------------
        r = np.random.randint(1,6)
        k = 2*r+1
        #M1 = cv.medianBlur(G1, k)


        # Invokation
        # ------------------------
        try:
            MC = ctmf(G, r)
        except Exception:
            print("rad:", rad)
            print("image:", G)
            err_count += 1
        
    print(f"Error %: {100*err_count/ N:0.2f}")

def Test4():
    ''' Preliminary speed test.  Benchmark slow python implementation vs cv2 code.
    '''
    from time import time
    # Image
    # ------------------------
    # ... (load image code goes here)
    # simple test for 10x10 image
    rad1, rad2 = 50, 50
    I = np.random.randint(0,255,(rad1,rad2))


    G = np.uint8(1/3*I.sum(axis=2)) if I.shape[-1] == 3 else np.uint8(I)
    G1 = G.copy()


    # Other variables
    # ------------------------
    DISPLAY = True
    r = 2
    k = 2*r+1

    t2i = time()
    M1 = cv.medianBlur(G1, k)
    t2f = time()
    print(f"time (cv2): {t2f - t2i: 0.5f}")


    # Invokation
    # ------------------------
    t1i = time()
    MC = ctmf(G, r)
    t1f = time()
    print(f"time (ctmf): {t1f - t1i: 0.5f}")
    # Display
    # ------------------------
    if DISPLAY:
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(G,cmap="gray")
        axs[1].imshow(M1,cmap="gray")
        axs[2].imshow(MC,cmap="gray")
        plt.show()

def Test3():
    ''' Testing the median filter on a real image
    '''

    # Image
    # ------------------------
    # ... (load image code goes here)

    I = cv.imread("images/viper2.png")
    RESIZE = True
    pct = 0.25
    if RESIZE:
        print("Initial size:", I.shape)
        I = cv.resize(I, (int(I.shape[0]*pct), int(I.shape[1]*pct)))
        print("New size:", I.shape)

    G = np.uint8(1/3*I.sum(axis=2)) if I.shape[-1] == 3 else np.uint8(I)
    G0 = G.copy()
    G1 = G0.copy()

    

    # Other variables
    # ------------------------
    DISPLAY = True
    r = 4
    k = 2*r+1
    M1 = cv.medianBlur(G1, k)


    # Invokation
    # ------------------------
    HTML_render(G)
    MC = ctmf(G, r)


    # Display
    # ------------------------
    if DISPLAY:
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(G,cmap="gray")
        axs[0].title.set_text('Source')

        axs[1].imshow(M1,cmap="gray")
        axs[1].title.set_text('CV2')

        axs[2].imshow(MC,cmap="gray")
        axs[2].title.set_text('CTMF')
        plt.show()


def Test2():
    '''
    This test will explore large batches of image sizes and make sure that there are no errors for various image sizes for a fixed kernel size.

    Result:
    > 10,000 tests run and 0 errors.
    '''
    # Image
    # ------------------------
    N = 1_000
    err_count = 0
    for i in range(N):
        rad1, rad2 = np.random.randint(4,30), np.random.randint(4,30)
        I = np.random.randint(0,255,(rad1,rad2))


        G = np.uint8(I)
        G1 = G.copy()


        # Other variables
        # ------------------------
        r = 1
        k = 2*r+1
        M1 = cv.medianBlur(G1, k)


        # Invokation
        # ------------------------
        try:
            MC = ctmf(G, r)
        except Exception:
            print("rad:", rad)
            print("image:", G)
            err_count += 1
        
    print(f"Error %: {100*err_count/ N:0.2f}")



def Test1():
    ''' Preliminary test in which a random grayscale image is blurred.  An HTML display of the original image is opened in the default web browser and then the constant time blur is 
    compared with the source image and the cv2 median blur implementation using matplotlib.
    '''

    # Image
    # ------------------------
    # ... (load image code goes here)
    # simple test for 10x10 image
    rad1, rad2 = 10, 10
    I = np.random.randint(0,255,(rad1,rad2))


    G = np.uint8(1/3*I.sum(axis=2)) if I.shape[-1] == 3 else np.uint8(I)
    G0 = G.copy()
    G1 = G0.copy()


    # Other variables
    # ------------------------
    DISPLAY = True
    r = 2
    k = 2*r+1
    M1 = cv.medianBlur(G1, k)


    # Invokation
    # ------------------------
    HTML_render(G)
    MC = ctmf_dev(G, r)

    # Display
    # ------------------------
    if DISPLAY:
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(G,cmap="gray")
        axs[1].imshow(M1,cmap="gray")
        axs[2].imshow(MC,cmap="gray")
        plt.show()


if __name__ == "__main__": Test5()