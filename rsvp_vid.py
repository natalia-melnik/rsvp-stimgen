
import glob
from PIL import Image
import sys
import os
import random
from random import shuffle 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import cv2
import math
import re

def openImage(i):
    return Image.open(i)

def operation():
    return sys.argv[1]

def seed(img):
    random.seed(hash(img.size))

def getPixels(img):
    w, h = img.size
    pxs = []
    for x in range(w):
        for y in range(h):
            pxs.append(img.getpixel((x, y)))
    return pxs

def scrambledIndex(pxs):
    idx = range(len(pxs))
    random.shuffle(idx)
    return idx

def scramblePixels(img):
    seed(img)
    pxs = getPixels(img)
    idx = scrambledIndex(pxs)
    out = []
    for i in idx:
        out.append(pxs[i])
    return out

def unScramblePixels(img):
    seed(img)
    pxs = getPixels(img)
    idx = scrambledIndex(pxs)
    out = range(len(pxs))
    cur = 0
    for i in idx:
        out[i] = pxs[cur]
        cur += 1
    return out

def storePixels(name, size, pxs):
    outImg = Image.new("RGB", size)
    w, h = size
    pxIter = iter(pxs)
    for x in range(w):
        for y in range(h):
            outImg.putpixel((x, y), pxIter.next())
    outImg.save(name)


def do_patches(img, savefile,h=50, w=50, lag=50, pixNumb=300):
    img2 = []
    y=0
    x=0
    for i in range(0,len(np.arange(0,pixNumb,lag))):
        for k in range(0,len(np.arange(0,pixNumb,lag))):
            y=np.arange(0,pixNumb,lag)[i]
            x=np.arange(0,pixNumb,lag)[k]

            img2.append(img[y:y+h, x:x+w])

            #cv2.imwrite('test_cr'+str(i)+'_'+str(y)+'_'+str(x)+'.jpg',img[y:y+h, x:x+w])
    '''
    shuffle(img2)
    ppatches = []
    rh = [np.arange(0,6), np.arange(6,12), np.arange(12,18),np.arange(18,18+6),np.arange(24,24+6),np.arange(30,30+6)]
    for numOfpatchesX in rh:

        plt.clf()
        ppatches.append(np.concatenate((img2[numOfpatchesX[0]],img2[numOfpatchesX[1]],img2[numOfpatchesX[2]],
                                   img2[numOfpatchesX[3]],img2[numOfpatchesX[4]],img2[numOfpatchesX[5]]),axis=1))


    figr = np.concatenate(ppatches)

    Image.fromarray(figr, 'RGB').save(savefile)
    #return figr, Image.fromarray(figr, 'RGB')
    '''
    return img2

def getAList(n2=6):
    n1=n2*n2
    N = np.arange(0,n1)
    tt= []
    for i in range(0,n1,n2):
        tt.append(N[i:i+n2])
    return tt


def do_nicePatches(img,savefile, bits=5):
    img2= do_patches(img, savefile, h=bits, w=bits, lag=bits)
    shuffle(img2)

    numberOfPatches=int(math.sqrt(len(img2)))
    ppatches = []
    rh = getAList(n2=numberOfPatches)
    rng = np.arange(0,numberOfPatches)
    b = ['img2[numOfpatchesX['+str(kk) +']]' for kk in range(0,len(rng))]
    for numOfpatchesX in rh:

        plt.clf()
        ppatches.append(np.concatenate(([eval(b[i]) for i in range(0,len(b))]),axis=1))

    figr = np.concatenate(ppatches)

    Image.fromarray(figr, 'RGB').save(savefile)
    
def do_img(i, savefold, sizeMin,scrambl1,scrambl2, scrFol1,scrFol2,output=False):
    im = Image.open(i)
    width, height = im.size
    if width<sizeMin or height<sizeMin:
        if width>height:
            wpercent = (sizeMin/float(im.size[1]))
            hsize = int((float(im.size[0])*float(wpercent)))
            im = im.resize((hsize,sizeMin), Image.ANTIALIAS)
        elif height>=width:
            wpercent = (sizeMin/float(im.size[0]))
            hsize = int((float(im.size[1])*float(wpercent)))
            im = im.resize((sizeMin,hsize), Image.ANTIALIAS)

    im_new = crop_center(im, sizeMin, sizeMin)
    if savefold == '':
        pass
        
    else: im_new.save(savefold + i.split('\\')[-1], quality=sizeMin)


        
    if output:
        return im_new
    
def prep_images(fold, savefold='batch2/', mask='\*.jpg', sizeMin=600,scrambl1=False,scrFol1='',scrambl2=False,scrFol2=''):

    if not os.path.exists(savefold):
        os.makedirs(savefold)

    all_jpg = glob.glob(fold+mask)
    
    
    for i in all_jpg:
        do_img(i,savefold, sizeMin,scrambl1,scrambl2, scrFol1,scrFol2)
        
        if scrambl1:
            img = openImage(savefold + i.split('\\')[-1])
            pxs = scramblePixels(img)
            storePixels(scrFol1+i.split('\\')[-1], img.size, pxs)
        if scrambl2:
            img = cv2.imread(savefold + i.split('\\')[-1])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            do_nicePatches(img, scrFol2+i.split('\\')[-1])
        
    
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def doVideo(fold, savefold, mask='*.jpg',frames=2, sortNaturally=False, shuffle1=False, black_first=True, blackSec=2, black_last=True, blackSecLast=1):
    img_array = []

    if black_first:
        filename = glob.glob(fold+'*.jpg')[0]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        my_dpi = size[0]
        plt.figure(figsize=(my_dpi / my_dpi, my_dpi / my_dpi), dpi=my_dpi)
        plt.subplots_adjust(0, 0, 1, 1)  # set white border size
        plt.savefig('temp-k.jpg', facecolor='black')  # save what's currently drawn
        img = cv2.imread('temp-k.jpg')
        # img = cv2.resize(img, (100, 50))
        height, width, layers = img.shape
        size = (width, height)
        for sec in range(0,blackSec*frames):
            img_array.append(img)


    if sortNaturally:
       
        for filename in  natural_sort(glob.glob(fold+'*.jpg')):
            img = cv2.imread(filename)
            #img = cv2.resize(img, (100, 50)) 
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
    else:
        if shuffle1:
            fff = glob.glob(fold+mask)
            shuffle(fff)
            fff = fff[0:361]
            
            for filename in fff:
                img = cv2.imread(filename)
                #img = cv2.resize(img, (100, 50)) 
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)

        else:
            for filename in glob.glob(fold+mask):
                img = cv2.imread(filename)
                #img = cv2.resize(img, (100, 50)) 
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)

    if black_last:
        filename = glob.glob(fold+'*.jpg')[0]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        my_dpi = size[0]
        plt.figure(figsize=(my_dpi / my_dpi, my_dpi / my_dpi), dpi=my_dpi)
        plt.subplots_adjust(0, 0, 1, 1)  # set white border size
        plt.savefig('temp-k.jpg', facecolor='black')  # save what's currently drawn
        img = cv2.imread('temp-k.jpg')
        # img = cv2.resize(img, (100, 50))
        height, width, layers = img.shape
        size = (width, height)
        for sec in range(0,blackSecLast*frames):
            img_array.append(img)

    out = cv2.VideoWriter(savefold,cv2.VideoWriter_fourcc(*'DIVX'), frames, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    


def natural_sort(l):
    '''
    sort files as human people would...
    :param l:
    :return:
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def do_Timer(fold='timer1/',savefold='timer1_vid.avi',my_dpi = 500, doVid=True, color1 ='powderblue', color2= 'darkcyan', facecol= 'lightgray', hz=2):
    '''
    # Create a timer video. 
    # fold -- folder to which the images will be saved
    # savefold -- timer file name 
    # my_dpi -- dpi of the video (500 means that the video is 500x500px)
    # doVid -- make video automatically when true
    # color scheme of the video: color1 is the empty ring, color2 is the filled ring, facecol is the background color. 
    # hz -- frames per sec (2 frames/images per sec means that it would take 180 secs (3 min) to finish the timer video). 
    '''
    import os
    if not os.path.exists(fold):
        os.makedirs(fold)

    plt.figure(figsize=(my_dpi/my_dpi, my_dpi/my_dpi), dpi=my_dpi)
    plt.subplots_adjust(0, 0, 1, 1) # set white border size
    ax = plt.subplot()
    fullang = 360
    for i in range(1, fullang):
        plt.cla() # clear what's drawn last time
        ###
        ax.invert_xaxis() # invert direction of x-axis since arc can only be drawn anti-clockwise
        ax.add_patch(Arc((.5, .5), .5, .5, -270, theta2=360, linewidth=5, color=color1)) # draw arc
        plt.axis('off') # hide number axis
        
        ###
        
        
        #ax.invert_xaxis() # invert direction of x-axis since arc can only be drawn anti-clockwise
        ax.add_patch(Arc((.5, .5), .5, .5, -270, theta2=i, linewidth=5, color=color2)) # draw arc
        plt.axis('off') # hide number axis
        plt.savefig(fold+str(i)+'.jpg', facecolor=facecol) # save what's currently drawn
    

    ###
    plt.cla()  # clear what's drawn last time
    ax.invert_xaxis()  # invert direction of x-axis since arc can only be drawn anti-clockwise
    ax.add_patch(Arc((.5, .5), .5, .5, -270, theta2=360, linewidth=5, color=color1))  # draw arc
    ax.add_patch(Arc((.5, .5), .5, .5, -270, theta2=360, linewidth=5, color=color2))  # draw arc
    plt.axis('off')  # hide number axis
    plt.savefig(fold + str(fullang+1) + '.jpg', facecolor=facecol)  # save what's currently drawn
    plt.cla()  # clear what's drawn last time
    ax.invert_xaxis()  # invert direction of x-axis since arc can only be drawn anti-clockwise
    ax.add_patch(Arc((.5, .5), .5, .5, -270, theta2=360, linewidth=5, color=color1))  # draw arc
    ax.add_patch(Arc((.5, .5), .5, .5, -180, theta2=360, linewidth=5, color=color2))  # draw arc
    ax.add_patch(Arc((.5, .5), .5, .5, -270, theta2=360, linewidth=5, color=color2))  # draw arc
    plt.axis('off')  # hide number axis
    plt.savefig(fold + str(fullang+2) + '.jpg', facecolor=facecol)  # save what's currently drawn
    plt.cla()  # clear what's drawn last time
    if doVid:
        doVideo(fold, savefold, mask='*.jpg',frames=hz,sortNaturally=True)
        
