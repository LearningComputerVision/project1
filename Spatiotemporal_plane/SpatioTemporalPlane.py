'''
Created on Jul 11, 2015

@author: Anh
'''
import cv2
import numpy as np

def MomentDescriptor(name, thres):
    img1=cv2.imread(name)   
#     img=img1
    img=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#   edges=cv2.Canny(img, thres, thres*2) 
    #Image to draw the contours
    drawing=np.zeros(img.shape[:2], np.uint8)
    ret,thresh = cv2.threshold(img,thres,255,0)
   
    contours, hierarchy=cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    MomentVector=[]
    
    for cnt in contours:
        M=cv2.moments(cnt) #Calculate moments
        if M['m00']!=0:
            Cx=int(M['m10']/M['m00'])
            Cy=int(M['m01']/M['m00'])
            
            Moments_Area=M['m00'] # Contour area moment
            Contours_Area=cv2.contourArea(cnt) # Contour area using in_built function
           #Draw moment
            rect = cv2.minAreaRect(cnt)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
#           cv2.drawContours(img1,contours, 0, (0,255,0),3) #draw contours in green color
            cv2.drawContours(img1,[box],0,(0,0,255),1)
            cv2.circle(img1, (Cx,Cy), 3,(0,255,0), -1)#draw centroids in red color
            MomentVector.append([M['m00'],Cx,Cy])
            cv2.imshow('winname',img1)
            cv2.waitKey(5000)
    print MomentVector
            
if __name__ == '__main__':
    MomentDescriptor('103_YT.bmp',0)