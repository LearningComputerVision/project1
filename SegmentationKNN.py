'''
Created on Aug 11, 2015

@author: Anh
'''
import sys
import cv2
import cv2.cv as cv
import numpy as np

def kmeans(img, nClass):
    '''
    preprocessing step
    '''
    img=cv2.GaussianBlur(img, (7,7), 0)
    '''
    consider each pixel is a point => convert image into a matrix have Hx3  repestively, [b, g, r]
    where H=NxM, N is row and M column of matrix
    '''
    vectorized=img.reshape(-1,3)
    
    vectorized=np.float32(vectorized)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    ret, label, center=cv2.kmeans(vectorized, nClass, 
                                  criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    res=center[label.flatten()]
    segmented_image=res.reshape((img.shape))
    return label.reshape((img.shape[0],img.shape[1])), segmented_image.astype(np.uint8)
        
if __name__ == '__main__':
   img=cv2.imread('peppers.jpg')
   label, result=kmeans(img,4)
   cv2.imshow('input', img)
   cv2.imshow('segmented',result)
   cv2.waitKey(0)