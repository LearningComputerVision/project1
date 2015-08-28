'''
Created on Aug 9, 2015

@author: Anh
'''
import cv2
import cv2.cv as cv
import numpy as np
from scipy.ndimage import label
def segment_watershed(img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh=cv2.threshold(gray, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    '''
    foreground region
    '''
    fg=cv2.erode(thresh,None, iterations=2)
    '''
    background region
    '''
    bg=cv2.dilate(thresh, None,iterations=3)
    ret, bg=cv2.threshold(bg,1,128,1)
    '''
    add both fg and bg
    '''
    marker=cv2.add(fg, bg)
    '''
    convert into 32SC1
    '''
    marker32=np.int32(marker)
    '''
    apply watershed
    '''
    cv2.watershed(img, marker32)
    m=cv2.convertScaleAbs(marker32)
    '''
    threshold it properly to get the mask and perform bitwise_and with the input image
    '''
    ret, thresh=cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    result=cv2.bitwise_and(img, img,mask=thresh)
    return result
def segment_on_dt(a, img):
    '''
    find foreground
    '''
    fg=cv2.dilate(img,None, iterations=5)
    fg=fg-cv2.erode(fg, None)
    dt=cv2.distanceTransform(img, 2, 3)
    
    dt=((dt-dt.min())/(dt.max()-dt.min())*255).astype(np.uint8)
   
    _, dt=cv2.threshold(dt, 0, 255, cv2.THRESH_BINARY)
    lbl, ncc=label(dt)
    lbl=lbl*(255/ncc)
    '''
    Completing the markers now
    '''
    lbl[fg==255]=255
    
    lbl=lbl.astype(np.int32)
    cv2.watershed(a, lbl)
    
    lbl[lbl==-1]=0
    lbl=lbl.astype(np.uint8)
    
    return 255-lbl
def segmentwaterhsed_improved(img):
    
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw=cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    
    kernel=np.ones((3,3),dtype='uint8')
    bw=cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    
    result=segment_on_dt(img, bw)
    result[result!=255]=0
    result=cv2.dilate(result, None)
    img[result==255]=(0,0,255)
    return img
if __name__ == '__main__':
    img=cv2.imread('3.jpg')
#     result=segment_watershed(img)
    result=segmentwaterhsed_improved(img)
    cv2.imshow('demo',result)
    cv2.waitKey(0)