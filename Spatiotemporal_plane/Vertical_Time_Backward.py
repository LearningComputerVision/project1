'''
Created on Jul 10, 2015

@author: Anh
'''
import numpy as np
import cv2
import sys
from numpy import power
import matplotlib.pyplot as plt

#write file in python
def writeFile(filename, img):
    fi=open(filename,"wb")
    fi.write(str(img))
    fi.close()
    
def VTB(t0,t1,t2):
    cols1, rows1 =t0.shape[:2] #get width and hight
    cols2, rows2 =t1.shape[:2]
    cols3, rows3 =t2.shape[:2]
    min_cols=min(cols1, cols2, cols3)
    min_rows=min(rows1, rows2, rows3)
    img_VTB_code=np.zeros(([min_cols, min_rows]),np.uint8)
    
    for i in range(0,min_cols):
        for j in  range(0,min_rows-3):
            img1=t0[i,j:j+3]
            img2=t1[i,j:j+3]
            img3=t2[i,j:j+3]
            #calculate delta
            
            img_VTB11=[]
            img_VTB22=[]
            img_VTB_=[]
            for k in range(0,3):
                img_VTB11.append(int(img3[k])-int(img1[k]))
                
            for k in range(0,3):
                img_VTB22.append(int(img3[k])-int(img2[k]))
            img_VTB_=np.concatenate((img_VTB11,img_VTB22), axis=0)
            
            #calcalute VTB code
            VTB_code=0
            for k in range(0,6):
                if img_VTB_[k]<=0:
                    img_VTB_[k]=0
                else:
                    img_VTB_[k]=1
                VTB_code=VTB_code + img_VTB_[k]*power(2.0,k)
            #add to image VTB
            img_VTB_code[i,j]=VTB_code
            
    cv2.imwrite('a.bmp', img_VTB_code)
    return img_VTB_code
def VTBDescriptors(img_VTB):    
    bin=64
    plt.hist(img_VTB.ravel(),bin,[0,256]); plt.show()
    
def TestVTB(name1, name2, name3):
    t0=cv2.cvtColor(cv2.imread(name1),cv2.COLOR_BGR2GRAY)
    t1=cv2.imread(name2,0)
    t2=cv2.imread(name3,0)
    #VTB(t0,t1,t2)
    
    VTBDescriptors(cv2.imread('a.bmp',0) )
    
if __name__ == '__main__':
    TestVTB('S010_002_00000012_07_M.bmp','S010_002_00000013_07_M.bmp','S010_002_00000014_07_M.bmp')
    