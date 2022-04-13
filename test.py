# Author: Nick Sebasco
# Testing out the cython & python ctmf functions
# ----------------------------------------------

from ctmf import ctmf 
from ctmf_py import ctmf as py_ctmf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from timeit import timeit


def Test7():
    ''' '''
    I = cv2.imread("images/spoke6.png")

    G = np.uint8(I.sum(axis=2)/3)

    I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    pct = 1
    I = cv2.resize(I, (int(I.shape[0]*pct), int(I.shape[1]*pct)))
    H,S,V= I[:,:,0], I[:,:,1], I[:,:,2]
    print("shape:",I.shape)

    r = 25
    DISPLAY = True

    # Display
    # ------------------------
    if DISPLAY:
        fig, axs = plt.subplots(1,4)
        print("-> working on [Hue]")
        Mh = ctmf(H,r)
        print("-> working on [Sat]")
        Ms = ctmf(S,r)
        print("-> working on [Value]")
        Mv = ctmf(V,r)

        cmap = "bone"
        #axs[0].imshow(G,cmap=cmap)
        axs[0].imshow(Mh,cmap=cmap)
        axs[0].set_axis_off()
        axs[1].imshow(Ms,cmap=cmap)
        axs[1].set_axis_off()
        axs[2].imshow(Mv,cmap=cmap)
        axs[2].set_axis_off()
        axs[3].imshow(cv2.cvtColor(cv2.merge([Mh,Ms,Mv]), cv2.COLOR_HSV2RGB),cmap=cmap)

        plt.tight_layout()
        plt.show()


def Test6():
    ''' RGB filtering
    '''
    I = cv2.imread("images/viper1.png")
    G = np.uint8(I.sum(axis=2)/3)

    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    R,G,B = I[:,:,0], I[:,:,1], I[:,:,2]
    
    r=7
    DISPLAY = True
    # Display
    # ------------------------
    if DISPLAY:
        fig, axs = plt.subplots(1,4)
        
        Mh = ctmf(R,r)
        Ms = ctmf(G,r)
        Mv = ctmf(B,r)

        #axs[0].imshow(G,cmap=cmap)
        axs[0].imshow(Mh,cmap='RdGy') # 'Reds', 'OrRd'
        axs[0].set_axis_off()
        axs[1].imshow(Ms,cmap='YlGn') # 'Greens', 'Greens_r'
        axs[1].set_axis_off()
        axs[2].imshow(Mv,cmap='winter') # 'Blues'
        axs[2].set_axis_off()
        axs[3].imshow(cv2.merge([Mh,Ms,Mv]),cmap='gray')

        plt.tight_layout()
        plt.show()

def Test5():
    """ HSV filtering
    cmap values:
    'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 
    'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r',
     'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 
     'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 
     'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 
     'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 
     'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 
     'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 
     'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 
     'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', '
     terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
    """
    I = cv2.imread("images/tadpoles.png")
    G = np.uint8(I.sum(axis=2)/3)

    I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    H,S,V = I[:,:,0], I[:,:,1], I[:,:,2]
    
    r=7
    DISPLAY = True
    # Display
    # ------------------------
    if DISPLAY:
        fig, axs = plt.subplots(1,4)

        Mh = ctmf(H,r)
        Ms = ctmf(S,r)
        Mv = ctmf(V,r)
        # nipy_spectral_r, bone, twilight, PuBu
        cmap = "nipy_spectral_r"
        #axs[0].imshow(G,cmap=cmap)
        axs[0].imshow(Mh,cmap=cmap)
        axs[0].set_axis_off()
        axs[1].imshow(Ms,cmap=cmap)
        axs[1].set_axis_off()
        axs[2].imshow(Mv,cmap=cmap)
        axs[2].set_axis_off()
        axs[3].imshow(cv2.cvtColor(cv2.merge([Mh,Ms,Mv]),cv2.COLOR_HSV2RGB ),cmap=cmap)

        plt.tight_layout()
        plt.show()

def Test4():
    ''' color filtering
    '''
    I = cv2.imread("images/viper2.png")
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    r=7
    DISPLAY = True
    # Display
    # ------------------------
    if DISPLAY:
        fig, axs = plt.subplots(1,3)

        Mcv_rgb = []
        Mctmf_rgb = []
        for i in range(1,4):
            Ic = I[:,:,i-1]
            Mcv = cv2.medianBlur(Ic, 2*r+1)
            Mctmf = np.uint8(ctmf(Ic, r))

            Mctmf_rgb.append(Mctmf)
            Mcv_rgb.append(Mcv)

        axs[0].imshow(I,cmap="gray")
        axs[1].imshow(cv2.merge(Mcv_rgb),cmap="gray")
        axs[2].imshow(cv2.merge(Mctmf_rgb),cmap="gray")

        plt.tight_layout()
        plt.show()

def Test3():
    ''' Extension of Test 2
    '''
    I = cv2.imread("images/kingCobra.png")
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    r=4
    DISPLAY = True
    # Display
    # ------------------------
    if DISPLAY:
        fig, axs = plt.subplots(5,3)
        
        I1 = np.uint8(I.sum(axis=2)/3)
        Mcv = cv2.medianBlur(I1, 2*r+1)
        Mctmf = ctmf(I1, r)
        axs[1][0].imshow(I1,cmap="gray")
        axs[2][0].imshow(Mcv,cmap="gray")
        axs[2][0].imshow(Mctmf,cmap="gray")


        colors = ['Reds','Greens','Blues']
        Mcv_rgb = []
        Mctmf_rgb = []
        for i in range(1,4):
            Ic = I[:,:,i-1]
            Mcv = cv2.medianBlur(Ic, 2*r+1)
            Mctmf = np.uint8(ctmf(Ic, r))

            Mctmf_rgb.append(Mctmf)
            Mcv_rgb.append(Mcv)

            axs[i][0].imshow(Ic,cmap=colors[i-1])
            axs[i][1].imshow(Mcv,cmap=colors[i-1])
            axs[i][2].imshow(Mctmf,cmap=colors[i-1])
        

        axs[4][0].imshow(I,cmap="gray")
        axs[4][1].imshow(cv2.merge(Mcv_rgb),cmap="gray")
        axs[4][2].imshow(cv2.merge(Mctmf_rgb),cmap="gray")

        plt.tight_layout()
        plt.show()

def Test2():
    ''' Try cython median filter on several test images:
    '''
    I1 = cv2.imread("images/kingCobra.png")
    I1 = np.uint8(I1.sum(axis=2)/3)
    r=4
    Mcv = cv2.medianBlur(I1, 2*r+1)
    Mctmf = ctmf(I1, r)
    DISPLAY = True
    # Display
    # ------------------------
    if DISPLAY:
        
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(I1,cmap="gray")
        axs[1].imshow(Mcv,cmap="gray")
        axs[2].imshow(Mctmf,cmap="gray")
        plt.show()


def Test1():
    ''' visual & time comparison of 3 median filter methods.
    '''
    DISPLAY = True

    G = np.random.randint(0,255,(400,400),dtype=np.uint8)
    r = 10
    N = 3
    Mctmf = ctmf(G, r)
    Mpyctmf = py_ctmf(G, r)
    Mcv = cv2.medianBlur(G, 2*r+1)

    res_cv = timeit(stmt=lambda: cv2.medianBlur(G, 2*r+1), number = N)
    #res_py_ctmf = timeit(stmt=lambda: py_ctmf(G, r), number = N)
    res_ctmf = timeit(stmt=lambda: ctmf(G, r), number = N)

    print(f"median filter (cv2): {res_cv:0.6f} s")
    print(f"median filter (ctmf): {res_ctmf:0.6f} s")
    #print(f"median filter (ctmf_py): {res_py_ctmf:0.6f} s")

    # Display
    # ------------------------
    if DISPLAY:
        fig, axs = plt.subplots(1,4)
        axs[0].imshow(G,cmap="gray")
        axs[1].imshow(Mcv,cmap="gray")
        axs[2].imshow(Mpyctmf,cmap="gray")
        axs[3].imshow(Mctmf,cmap="gray")
        plt.show()


if __name__=="__main__": Test7()


