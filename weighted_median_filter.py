import numpy as np
import cv2
from numba import jit

def gkern(l=5):
    sig = (l-1)/6
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

@jit
def weighted_median_filter(image, k):

    idx = k//2
    data_final = []
    h,w,_ = image.shape
    print(h,w)
    data_final = np.zeros((h,w,3))

    kernel = gkern(k)

    for r in range(3):
        img = -1*np.ones((h+k-1,w+k-1))
        h_p,w_p = img.shape

        img[idx:h_p-idx,idx:w_p-idx] = image[:,:,r]

        for i in range(k//2,h_p-k//2):
            for j in range(k//2,w_p-k//2):
                print(i,j)
                p = img[i-k//2:i+k//2+1,j-k//2:j+k//2+1] * kernel
                temp = p.ravel()
                temp = [ele for ele in temp if ele > 0]
                temp.sort()
                data_final[i-k//2,j-k//2,r] = temp[len(temp) // 2]

    return data_final

img = cv2.imread('luna.jpg')
cv2.imshow('img',img)
cv2.waitKey(0)
res = weighted_median_filter(img,5)
cv2.imshow('res',res)
cv2.waitKey(0)