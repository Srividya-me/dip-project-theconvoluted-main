import numpy as np
import matplotlib.pyplot as plt
import cv2
import statistics
import argparse
import math

def weightfun(img,mag,ori,i,j,p,q,k1):
    alpha = 1./mag[p][q] 
    beta = 2.*(ori[i][j]-ori[p][q])
    return ((math.cos(beta)+1)*alpha)

def smoothchannel(img,k,mag,ori):
    fin_img = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            up = i - k // 2
            left = j - k // 2
            down = i + k // 2 + 1
            right = j + k // 2 + 1
            sum_weights = 0
            result = 0
            for s in range(up, down):
                if s < 0 or s >= img.shape[0]:
                    continue
                for t in range(left, right):
                    if t < 0 or t >= img.shape[1]:
                        continue
                    if mag[s][t] == .0:
                        continue
                    if s != i or t != j:
                        p = s
                        q = t
                        weight = weightfun(img,mag,ori,i,j,p,q,k)
                    else:
                        weight = 1.
                    result += weight * img[s][t]
                    sum_weights += weight
            if sum_weights != 0:
                fin_img[i][j] = round(result / sum_weights)
    return (fin_img).astype(np.uint8)
    
def gradient_analysis(img, k):
    fin_img = np.zeros((img.shape))
    mag = np.zeros((img.shape[0],img.shape[1]))
    ori = np.zeros((img.shape[0],img.shape[1]))
    for ii in range(3):
        temp_img = img[:,:,ii]
        mag = np.zeros((img.shape[0],img.shape[1]))
        ori = np.zeros((img.shape[0],img.shape[1]))
        for y in range(temp_img.shape[0]):
            for x in range(temp_img.shape[1]):
                if(0<x<temp_img.shape[1]-1 and 0<y<temp_img.shape[0]-1):
                    gx = int(temp_img[y][x-1])-int(temp_img[y][x+1])
                    gy = int(temp_img[y+1][x])-int(temp_img[y-1][x]) 
                    mag[y][x] = math.sqrt(gx*gx+gy*gy)
                    ori[y][x] = math.atan2(gx, gy)
        fin_img[:,:,ii] = smoothchannel(temp_img,k,mag,ori)
    return fin_img.astype(np.uint8)  

img=cv2.imread('puppy.png',1)
k = 5
fin_img = gradient_analysis(img,k)
plt.figure(figsize=[10,10])
plt.subplot(2,2,1)
plt.imshow(img)
plt.subplot(2,2,2)
plt.imshow(fin_img)
k = 10
fin_img = gradient_analysis(img,k)
plt.subplot(2,2,3)
plt.imshow(fin_img)
k = 15
fin_img = gradient_analysis(img,k)
plt.subplot(2,2,4)
plt.imshow(fin_img)