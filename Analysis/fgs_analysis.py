# -*- coding: utf-8 -*-
"""dipprj.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ua5fxGDhzNhrJB8dwdbcMG92EUX-0U9A
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def cw_1d(p,q,g,sigmac):
  return math.exp(-1*np.linalg.norm(g[p]-g[q])/sigmac)

def compute_lamb(t, T, lamb_base):
    return 1.5 * 4**(T-t) / (4 ** T - 1) * lamb_base

def fgs1d(lamb, f, g, sigma):
    w = f.shape[0]
    c = np.zeros(w-1)
    c[0] = -lamb * cw_1d(0, 1, g, sigma) / (1 + lamb * cw_1d(0, 1, g, sigma))
    int_f = np.zeros(w)
    int_f[0] = f[0] / (1 + lamb * cw_1d(0, 1, g, sigma))
    for i in range(1, w-1):
        c[i] = -lamb * cw_1d(i, i + 1, g, sigma) / (
            1 + lamb * (cw_1d(i, i - 1, g, sigma) + cw_1d(i, i + 1, g, sigma)) +
            lamb * c[i - 1] * cw_1d(i, i - 1, g, sigma))
        int_f[i] = (f[i] + int_f[i - 1] * lamb * cw_1d(i, i - 1, g, sigma)) / (
            1 + lamb * (cw_1d(i, i - 1, g, sigma) + cw_1d(i, i + 1, g, sigma)) +
            lamb * c[i - 1] * cw_1d(i, i - 1, g, sigma))
    int_f[w-1] = (f[w-1] + int_f[w-2] * lamb * cw_1d(w-1, w-2, g, sigma)) / (
            1 + lamb * (cw_1d(w-1, w-2, g, sigma)) +
            lamb * c[w-2] * cw_1d(w-1, w-2, g, sigma))
    u = np.zeros(f.shape)
    u[w - 1] = int_f[w - 1]
    for i in range(w - 2, -1, -1):
        u[i] = int_f[i] - c[i] * u[i + 1]
    return u

x1 = np.random.normal(scale=0.2, size=(300))
x = np.concatenate((x1, x1))
plt.plot(np.arange(x.shape[0]), x)

u =fgs1d(900, np.array(x), np.array(x), 0.07)
plt.plot(np.arange(u.shape[0]), u)

def fgs2d(f, T, lamb_base, sigma):
    print('origin lamb is {}'.format(lamb_base))
    print('sigma is {}'.format(sigma))
    H, W, C = f.shape
    u = f.copy()
    for t in range(1, T+1):
        lamb_t = compute_lamb(t, T, lamb_base)
        # horizontal
        for y in range(0, W):
            g = u[:, y, :]
            for c in range(C):
                f_h = u[:, y, c]
                u[:, y, c] =fgs1d(lamb_t, f_h, g, sigma)
        # vertical
        for x in range(0, H):
            g = u[x, :, :]
            for c in range(C):
                f_v = u[x, :, c]
                u[x, :, c] =fgs1d(lamb_t, f_v, g, sigma)
    return u

import random
img = cv2.cvtColor(cv2.imread('/content/0001.png'),cv2.COLOR_BGR2RGB)
plt.imshow(img)
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

noise_img = sp_noise(img,0.05)
cv2.imwrite('sp_noise.jpg', noise_img)

import time
lamb_base = 5**2
sigma = 255 *0.15
T = 1
i=1
t0 = time.time()
u1 = fgs2d(img, T, lamb_base, sigma)
t1 = time.time()
total = t1-t0

cv2.imwrite('output_%.4d.png'%(i), u1)

#show_u = u1.astype('uint8').asnumpy()
print(np.max(u1))
fig,ax=plt.subplots(1,2,figsize=(5,5))
ax[0].imshow(u1)

ax[1].imshow(noise_img)

i=5

img=cv2.imread('/content/0001.png',1)
import random
img = cv2.cvtColor(cv2.imread('/content/0001.png'),cv2.COLOR_BGR2RGB)
plt.imshow(img)
def gauss_noise(img):
  gauss = np.random.normal(0,1,img.size)
  gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
  noise = img + img * gauss
  return noise

noise_img = gauss_noise(img)
import time
lamb_base = 5**2
sigma = 255 *0.15
T = 1
i=1
t0 = time.time()
u1 = fgs2d(noise_img, T, lamb_base, sigma)
t1 = time.time()
total = t1-t0

plt.imshow(u1)

img = cv2.cvtColor(cv2.imread('/content/0001.png'),cv2.COLOR_BGR2RGB)
from skimage.util import random_noise

noisy=random_noise(img, mode='poisson')
import time
lamb_base = 5**2
sigma = 255 *0.15
T = 1
i=1
t0 = time.time()
u1 = fgs2d(noisyImage, T, lamb_base, sigma)
t1 = time.time()
total = t1-t0

plt.figure(figsize=[10,10])
plt.subplot(1,2,1)
plt.imshow(u1)
plt.subplot(1,2,2)
plt.imshow(noisyImage)

img=cv2.imread('/content/0001.png',1)
row,col,ch = img.shape
gauss = np.random.randn(row,col,ch)
gauss = gauss.reshape(row,col,ch)        
noisy = img + img * gauss
noisy=np.clip(noisy.astype(int),0,255)
import time
lamb_base = 5**2
sigma = 255 *0.15
T = 1
i=1
t0 = time.time()
u1 = fgs2d(noisyImage, T, lamb_base, sigma)
t1 = time.time()
total = t1-t0

plt.figure(figsize=[10,10])
plt.subplot(1,2,1)
plt.imshow(u1)
plt.subplot(1,2,2)
plt.imshow(noisy)

"""### Time vs Image size"""

import time
img=cv2.imread('/content/0001.png',1)
total=[]
sizes=[]
for i in range(5,15):
  img = cv2.resize(img, (2**i, 2**i))
  lamb_base = 5**2
  sigma = 255 *0.15
  T = 1
  t0 = time.time()
  u1 = fgs2d(img, T, lamb_base, sigma)
  t1 = time.time()
  total.append(t1-t0)
  sizes.append(2**i)
plt.plot(sizes,total)

plt.plot(sizes,total)

"""### Time vs T"""

import time
img=cv2.imread('/content/0001.png',1)
img = cv2.resize(img, (32,32))
total=[]
sizes=[]
for i in range(5,15):
  lamb_base = 5**2
  sigma = 255 *0.15
  t0 = time.time()
  u1 = fgs2d(img, i, lamb_base, sigma)
  t1 = time.time()
  total.append(t1-t0)
  sizes.append(i)

plt.plot(sizes,total)