'''
Created on Jul 9, 2015

@author: Anh
'''
import cv2
import numpy as np
def diffImage(t0, t1, t2):
    delta1=cv2.absdiff(t1, t0)
    delta2=cv2.absdiff(t2,t1)
    return cv2.bitwise_and(delta1, delta2)

def WebcamDiffImage():
    cam=cv2.VideoCapture(0)
    if cam.isOpened():
        #read three images
        t_minus=cv2.cvtColor(cam.read()[1], cv2.COLOR_BGR2GRAY)
        t=cv2.cvtColor(cam.read()[1],cv2.COLOR_BGR2GRAY)
        t_plus=cv2.cvtColor(cam.read()[1],cv2.COLOR_BGR2GRAY)
        
        while True:
            #read next image
            cv2.imshow('Differental Image', diffImage(t_minus, t, t_plus))
            t_minus=t
            t=t_plus
            t_plus=cv2.cvtColor(cam.read()[1], cv2.COLOR_BGR2GRAY)
            
            key=cv2.waitKey(100)
            if key==27:
                break
    
if __name__ == '__main__':
    WebcamDiffImage()