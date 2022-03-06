#!/usr/bin/env python
# coding: utf-8

# In[3]:


import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import statistics
import argparse
import math
from wand.image import Image
import scipy
import time


# In[4]:


# In[5]:


def bilateral(img,k1,sig_s,sig_c):
    img=img.astype(np.float64)
    img2=np.ones((img.shape[0]-k1+1,img.shape[1]-k1+1))
    #mask=np.ones((k1,k1))/(k1**2)
    temp=math.floor((k1-1)/2)
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            window=np.ones((k1,k1))
            for k in range(k1):
                for l in range(k1):
                    li=((img[k+i][l+j]-img[i+temp][j+temp])**2)/(2*(sig_s**2))
                    li2=((k-temp)**2)/(2*(sig_c**2))
                    li3=((l-temp)**2)/(2*(sig_c**2))
                    mo=li2+li3
                    li+=mo
                    window[k][l]=math.exp(-li)
            li=np.sum(np.multiply(window,img[i:i+k1,j:j+k1]))
            li1=np.sum(window)
            img2[i][j]=np.round(li/li1)
    return img2.astype(np.uint8)
def bilateral_rgb(img,k1,sig_s,sig_c):
    img2=np.stack([ 
        bilateral(img[:,:,0], k1,sig_s,sig_c ),
        bilateral(img[:,:,1], k1,sig_s,sig_c ),
        bilateral(img[:,:,2], k1,sig_s,sig_c )], axis=2 )
    return img2


# In[6]:


#bilateral on original image
img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
k1=5
sig_s=50
sig_c=50
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken = ",(end-start))
plt.figure(figsize=[10,10])
plt.subplot(221)
plt.title('original')
plt.imshow(img)

plt.subplot(222)
plt.title('bilateral original')
plt.imshow(out_img)


# In[32]:


k1=5
sig_s=50
sig_c=50
img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
plt.subplot(151)
plt.title('T = 1')
plt.imshow(out_img)

img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
plt.figure(figsize=[10,10])
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
plt.subplot(152)
plt.title('T = 3')
plt.imshow(out_img)
img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)           
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
plt.subplot(153)
plt.title('T = 6')
plt.imshow(out_img)
img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
plt.subplot(154)
plt.title('T = 9')
plt.imshow(out_img)
img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
img=bilateral_rgb(img,k1,sig_s,sig_c)
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
plt.subplot(155)
plt.title('T = 12')
plt.imshow(out_img)


# In[30]:


#varying k1 
img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
plt.figure(figsize=[10,10])
k1=1
sig_s=50
sig_c=50
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken for k=1 is ",(end-start))
plt.subplot(151)
plt.title('kernal size = 1')
plt.imshow(out_img)
k1=3
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken for k=3 is ",(end-start))
plt.subplot(152)
plt.title('kernal size = 3')
plt.imshow(out_img)
k1=5
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken for k=5 is ",(end-start))
plt.subplot(153)
plt.title('kernal size = 5')
plt.imshow(out_img)
k1=7
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken for k=7 is ",(end-start))
plt.subplot(154)
plt.title('kernal size = 7')
plt.imshow(out_img)
k1=9
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken for k=9 is ",(end-start))
plt.subplot(155)
plt.title('kernal size = 9')
plt.imshow(out_img)


# In[12]:



x = [1,3,5,7,9]
y = [15.6,32.5,60.5,104.1,160.5]
plt.plot(x, y)
plt.xlabel('kernal size')
plt.ylabel('time taken')
plt.title('Bilateral time analysis')
plt.show()


# In[11]:


#varying sig_s
img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
plt.figure(figsize=[10,10])
k1=3
sig_s=50
sig_c=2
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken for k=1 is ",(end-start))
plt.subplot(151)
plt.title('sig_c = 2')
plt.imshow(out_img)
sig_c= 10
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken for k=3 is ",(end-start))
plt.subplot(152)
plt.title('sig_c = 10')
plt.imshow(out_img)
sig_c = 20
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken for k=5 is ",(end-start))
plt.subplot(153)
plt.title('sig_c = 20')
plt.imshow(out_img)
sig_c=30
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken for k=7 is ",(end-start))
plt.subplot(154)
plt.title('sig_c = 30')
plt.imshow(out_img)
sig_c = 40
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken for k=9 is ",(end-start))
plt.subplot(155)
plt.title('sig_c = 40')
plt.imshow(out_img)


# In[13]:


x = [2,10,20,30,40]
y = [32.8,38.5,37.3,35.4,37.1]
plt.plot(x, y)
plt.xlabel('sig_c')
plt.ylabel('time taken')
plt.title('Bilateral time analysis')
plt.show()


# In[10]:


#varying sig_s
img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
plt.figure(figsize=[10,10])
k1=3
sig_s=2
sig_c=50
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken for k=1 is ",(end-start))
plt.subplot(151)
plt.title('sig_s = 2')
plt.imshow(out_img)
sig_s = 10
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken for k=3 is ",(end-start))
plt.subplot(152)
plt.title('sig_s = 10')
plt.imshow(out_img)
sig_s = 20
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken for k=5 is ",(end-start))
plt.subplot(153)
plt.title('sig_s = 20')
plt.imshow(out_img)
sig_s=30
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken for k=7 is ",(end-start))
plt.subplot(154)
plt.title('sig_s = 30')
plt.imshow(out_img)
sig_s = 40
start = time.time()
out_img=bilateral_rgb(img,k1,sig_s,sig_c)
end = time.time()
print("time taken for k=9 is ",(end-start))
plt.subplot(155)
plt.title('sig_s = 40')
plt.imshow(out_img)


# In[28]:


x = [2,10,20,30,40]
y = [30.4,34.9,35.5,35.8,36.1]
plt.plot(x, y)
plt.xlabel('sig_s')
plt.ylabel('time taken')
plt.title('Bilateral time analysis')
plt.show()


# In[38]:


#sp noise
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


# In[54]:


# salt pepper noise adding
img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
noise_img = sp_noise(img,0.05)
k1=5
sig_s=50
sig_c=50
start = time.time()
noise_out_img=bilateral_rgb(noise_img,k1,sig_s,sig_c)
end = time.time()
print("time taken = ",(end-start))
plt.figure(figsize=[10,10])

plt.subplot(121)
plt.title('sp noise img')
plt.imshow(noise_img)

plt.subplot(122)
plt.title('bilateral noise img')
plt.imshow(noise_out_img)


# In[29]:


#varying prob in sp noise

i=0
j=5
plt.figure(figsize=[100,100])
for prob in range(0,5,1):
    i=i+1
    j=j+1
    noise_img = sp_noise(img,(prob/10))
    plt.subplot(2,5,i)
  #  plt.title('prob=',(prob/10))
    plt.imshow(noise_img)
    k1=5
    sig_s=50
    sig_c=50
    noise_out_img=bilateral_rgb(noise_img,k1,sig_s,sig_c)
    plt.subplot(2,5,j)
    plt.title('prob='+str((prob/10)))
    plt.imshow(noise_out_img)
    
    
    


# In[21]:


k1=3
sig_s=50
sig_c=50


# In[22]:


#laplacian
img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
noise_img = cv2.cvtColor(cv2.imread('./content/0001_laplacian.png'),cv2.COLOR_BGR2RGB)
start = time.time()
out_img_spac=bilateral_rgb(noise_img,k1,sig_s,sig_c)
end = time.time()
print("time taken = ",(end-start)/100)
plt.figure(figsize=[10,10])
plt.subplot(121)
plt.title('laplacian_noise img')
plt.imshow(noise_img)

plt.subplot(122)
plt.title('bilateral noise img')
plt.imshow(out_img_spac)


# In[23]:


img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
noise_img = cv2.cvtColor(cv2.imread('./content/0001_impulse.png'),cv2.COLOR_BGR2RGB)
start = time.time()
out_img_spac=bilateral_rgb(noise_img,k1,sig_s,sig_c)
end = time.time()
print("time taken = ",(end-start)/100)
plt.figure(figsize=[10,10])
plt.subplot(121)
plt.title('impulse_noise img')
plt.imshow(noise_img)

plt.subplot(122)
plt.title('bilateral noise img')
plt.imshow(out_img_spac)


# In[24]:


img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
noise_img = cv2.cvtColor(cv2.imread('./content/0001_uniform.png'),cv2.COLOR_BGR2RGB)
start = time.time()
out_img_spac=bilateral_rgb(noise_img,k1,sig_s,sig_c)
end = time.time()
print("time taken = ",(end-start)/100)
plt.figure(figsize=[10,10])
plt.subplot(121)
plt.title('uniform_noise img')
plt.imshow(noise_img)

plt.subplot(122)
plt.title('bilateral noise img')
plt.imshow(out_img_spac)


# In[25]:


img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
noise_img = cv2.cvtColor(cv2.imread('./content/0001_multiplicative_gaussian.png'),cv2.COLOR_BGR2RGB)
start = time.time()
out_img_spac=bilateral_rgb(noise_img,k1,sig_s,sig_c)
end = time.time()
print("time taken = ",(end-start)/100)
plt.figure(figsize=[10,10])
plt.subplot(121)
plt.title('multiplicative gaussian_noise img')
plt.imshow(noise_img)

plt.subplot(122)
plt.title('bilateral noise img')
plt.imshow(out_img_spac)


# In[26]:


img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
noise_img = cv2.cvtColor(cv2.imread('./content/0001_gaussian.png'),cv2.COLOR_BGR2RGB)
start = time.time()
out_img_spac=bilateral_rgb(noise_img,k1,sig_s,sig_c)
end = time.time()
print("time taken = ",(end-start)/100)
plt.figure(figsize=[10,10])
plt.subplot(121)
plt.title('gaussian_noise img')
plt.imshow(noise_img)

plt.subplot(122)
plt.title('bilateral noise img')
plt.imshow(out_img_spac)


# In[27]:


img = cv2.cvtColor(cv2.imread('./content/0001.png'),cv2.COLOR_BGR2RGB)
noise_img = cv2.cvtColor(cv2.imread('./content/0001_poisson.png'),cv2.COLOR_BGR2RGB)
start = time.time()
out_img_spac=bilateral_rgb(noise_img,k1,sig_s,sig_c)
end = time.time()
print("time taken = ",(end-start)/100)
plt.figure(figsize=[10,10])
plt.subplot(121)
plt.title('poisson_noise img')
plt.imshow(noise_img)

plt.subplot(122)
plt.title('bilateral noise img')
plt.imshow(out_img_spac)


# In[ ]:




