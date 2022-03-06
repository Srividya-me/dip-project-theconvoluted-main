import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename,asksaveasfilename
from PIL import Image, ImageTk, ImageFilter, ImageEnhance, ImageOps
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
import statistics
import argparse
import math

LARGEFONT =("Verdana", 35)

img_path = None

def selected():
    global img_path, img, canvas2
    img_path = filedialog.askopenfilename(initialdir=os.getcwd())
    # print(img_path)
    img = Image.open(img_path)
    img.thumbnail((700, 700))
    img1 = ImageTk.PhotoImage(img)
    canvas2.create_image(300, 210, image=img1)
    canvas2.image=img1

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

def fgs2d(event):
    global img_path,img1,v1,imgg

    lamb_base = 10**2
    sigma = v1.get()
    T=1

    img = Image.open(img_path)
    img_path_rel = str(os.path.relpath(img_path))
    img_cv = cv2.imread(img_path_rel,1)
    #img_cv = cv2.resize(img_cv,(300,210))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)


    H, W, C = img_cv.shape
    u = img_cv.copy()
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
    
    imgg = u.astype(np.int8)
    img1 = ImageTk.PhotoImage(Image.fromarray(imgg))
    canvas2.create_image(300, 210, image=img1)
    canvas2.image = u.astype(np.uint8)
   
def gradient_analysis(event):
    global img_path,img1,v1,imgg

    #print("*",img_path)

    k = v1.get()
    
    img = Image.open(img_path)

    img_path_rel = str(os.path.relpath(img_path))

    img_cv = cv2.imread(img_path_rel,1)
    #img_cv = cv2.resize(img_cv,(300,210))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    fin_img = np.zeros((img_cv.shape))

    for ii in range(3):
        temp_img = img_cv[:,:,ii]
        mag = np.zeros((img_cv.shape[0],img_cv.shape[1]))
        ori = np.zeros((img_cv.shape[0],img_cv.shape[1]))
        for y in range(1,temp_img.shape[0]-1):
            for x in range(1,temp_img.shape[1]-1):
                print(ii,y,x)
                gx = int(temp_img[y][x-1])-int(temp_img[y][x+1])
                gy = int(temp_img[y+1][x])-int(temp_img[y-1][x]) 
                mag[y][x] = math.sqrt(gx*gx+gy*gy)
                ori[y][x] = math.atan2(gx, gy)
        fin_img[:,:,ii] = smoothchannel(temp_img,k,mag,ori)

    imgg = fin_img.astype(np.uint8)
    #print(imgg.shape)
    img1 = ImageTk.PhotoImage(Image.fromarray(imgg))
    # img1.thumbnail((350, 350))
    canvas2.create_image(300, 210, image=img1)
    canvas2.image = fin_img.astype(np.uint8)

class tkinterApp(tk.Tk):
    
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
        
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)

        # creating a container
        container = tk.Frame(self)
        container.pack(side = "top", fill = "both", expand = True)

        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)

        # initializing frames to an empty array
        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (StartPage, Page1, Page2):

            frame = F(container, self)

            # initializing frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[F] = frame

            frame.grid(row = 0, column = 0, sticky ="nsew")

        self.show_frame(StartPage)

    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

# first window frame startpage

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        # label of frame Layout 2
        label = ttk.Label(self, text ="TheConvoluted", font = LARGEFONT)
        
        # putting the grid in its place by using
        # grid
        label.grid(row = 0, column = 4, padx = 10, pady = 10)

        button1 = ttk.Button(self, text ="Gradient Analysis",
        command = lambda : controller.show_frame(Page1))
    
        # putting the button in its place by
        # using grid
        button1.grid(row = 1, column = 1, padx = 10, pady = 10)

        ## button to show frame 2 with text layout2
        button2 = ttk.Button(self, text ="2D FGS",
        command = lambda : controller.show_frame(Page2))
    
        # putting the button in its place by
        # using grid
        button2.grid(row = 2, column = 1, padx = 10, pady = 10)
        


# second window frame page1
class Page1(tk.Frame):
    
    def __init__(self, parent, controller):

        global canvas2, v1
        
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text ="Gradient Analysis", font = LARGEFONT)
        label.grid(row = 0, column = 4, padx = 10, pady = 10)

        # button to show frame 2 with text
        # layout2
        button1 = ttk.Button(self, text ="Start Page",
                            command = lambda : controller.show_frame(StartPage))
    
        # putting the button in its place
        # by using grid
        button1.place(x=300, y=595)

        canvas2 = tk.Canvas(self, width="600", height="420", relief=tk.RIDGE, bd=2)
        canvas2.place(x=15, y=150)
 
        btn1 = ttk.Button(self, text="Select Image", command=selected)
        btn1.place(x=100, y=595)

        k_val = ttk.Label(self, text="K:", font=("ariel 17 bold"), width=9, anchor='e')
        k_val.place(x=5, y=90)
        v1 = tk.IntVar()
        scale1 = ttk.Scale(self, from_=0, to=10, variable=v1, orient=tk.HORIZONTAL, command=gradient_analysis) 
        scale1.place(x=150, y=100)


# third window frame page2
class Page2(tk.Frame):
    def __init__(self, parent, controller):
        global canvas2, v1
        
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text ="2D FGS", font = LARGEFONT)
        label.grid(row = 0, column = 4, padx = 10, pady = 10)

        # button to show frame 2 with text
        # layout2
        button1 = ttk.Button(self, text ="Start Page",
                            command = lambda : controller.show_frame(StartPage))
    
        # putting the button in its place
        # by using grid
        button1.place(x=300, y=595)

        canvas2 = tk.Canvas(self, width="600", height="420", relief=tk.RIDGE, bd=2)
        canvas2.place(x=15, y=150)
 
        btn1 = ttk.Button(self, text="Select Image", command=selected)
        btn1.place(x=100, y=595)

        k_val = ttk.Label(self, text="K:", font=("ariel 17 bold"), width=9, anchor='e')
        k_val.place(x=5, y=90)
        v1 = tk.IntVar()
        scale1 = ttk.Scale(self, from_=0, to=10, variable=v1, orient=tk.HORIZONTAL, command=fgs2d) 
        scale1.place(x=150, y=100)


# Driver Code
app = tkinterApp()
app.geometry("750x750")
app.mainloop()