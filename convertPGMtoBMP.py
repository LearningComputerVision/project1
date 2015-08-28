'''
Created on Jul 27, 2015

@author: Anh
'''
import cv2
import cv2.cv as cv
import numpy as np
import glob,os

def convert_toBMP(old_folder, new_folder):
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    
    list_file=[]
    list_dir=[]
    
    for root, dirname, file in os.walk(old_folder): 
        for filename in file:
            filepath=os.path.join(root,filename)
            list_file.append(filepath)#get file path
        
        for dir in dirname:
            list_dir.append(dir)
    
    
    #create subdir
    for dir in list_dir:
        subdir=os.path.join(new_folder,dir)
        
        if not os.path.exists(subdir):
            os.mkdir(subdir)
    #create list file
    for file in list_file:
        name=file
        name= name.split('\\')
        img=cv2.imread(file,0)
        cv2.imshow('winname', img)
        cv2.waitKey(40)
        name_file=name[len(name)-1].split('.')
        new_name= new_folder+'\\'+name_file[0]+'.bmp'
       # cv2.imwrite(new_name, img)
        
def resizeAllImage(old_folder, new_folder, new_heigh, new_width):
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    
    list_file=[]
    list_dir=[]
    
    for root, dirname, file in os.walk(old_folder): 
        for filename in file:
            filepath=os.path.join(root,filename)
            list_file.append(filepath)#get file path
        
        for dir in dirname:
            list_dir.append(dir)
    
    
    #create subdir
    for dir in list_dir:
        subdir=os.path.join(new_folder,dir)
        
        if not os.path.exists(subdir):
            os.mkdir(subdir)
    #create list file
    for file in list_file:
        name=file
        name= name.split('\\')
        img=cv2.imread(file,0)
        name_file=name[len(name)-1].split('.')
        new_name= new_folder+'\\'+name_file[0]+'.bmp'
        image_resized=cv2.resize(img,(new_width,new_heigh))
        cv2.imwrite(new_name, image_resized)
              
def detectHandFolder(in_folder):
    list_file=[]
    list_dir=[]
    
    for root, dirname, file in os.walk(in_folder): 
        for filename in file:
            filepath=os.path.join(root,filename)
            list_file.append(filepath)#get file path
    
    #create list file
    for file in list_file:
        HandDetection(file)

def HandDetection(file):   
    
    img=cv2.imread(file,0)
    #get size of input image
    w,h=img.shape[:2]
    #create grayscale version
    gray=np.zeros((h,w,3),np.uint8)

    #cv2.cvtColor(img, gray)
    gray=img
#     cv2.imshow('winname', gray)
#     cv2.waitKey(0)
    #create storage
    storage=cv.CreateMemStorage(0)
    #equalize histogram
#     cv.EqualizeHist(gray, gray)
    cascade_hand= cv2.CascadeClassifier('mycascade.xml')
    
    hands=cascade_hand.detectMultiScale(img, scaleFactor=1.3, 
                                        minNeighbors=2, minSize=(24,24), 
                                        flags=cv.CV_HAAR_DO_CANNY_PRUNING)
    
    
    for (x,y,w,h) in hands:
        cv2.rectangle(img, (x,y), (x+w,y+h), 255)
        
    cv2.imshow('winname', img)
    cv2.waitKey(0)
    
def calMoments(name, thres):
    img1=cv2.imread(name)   
    img=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#   edges=cv2.Canny(img, thres, thres*2) 
    #Image to draw the contours
    drawing=np.zeros(img.shape[:2], np.uint8)
    ret,thresh = cv2.threshold(img,127,255,0)
   
    contours, hierarchy=cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        M=cv2.moments(cnt) #Calculate moments
        if M['m00']!=0:
            Cx=int(M['m10']/M['m00'])
            Cy=int(M['m01']/M['m00'])
            C_x=M['m10']/M['m00']
            print C_x
            Moments_Area=M['m00'] # Contour area moment
            Contours_Area=cv2.contourArea(cnt) # Contour area using in_built function
#         #Draw moment
            rect = cv2.minAreaRect(cnt)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
#           cv2.drawContours(img1,contours, 0, (0,255,0),3) #draw contours in green color
            cv2.drawContours(img1,[box],0,(0,0,255),5)
            cv2.circle(img1, (Cx,Cy), 5,(0,255,0), -1)#draw centroids in red color
            
    cv2.imshow("Original", img1)
    cv2.waitKey(0)
    
if __name__ == '__main__':
 resizeAllImage('d:\\Database\\SL\\hands', 'D:\\Database\\SL\\pos', 128, 128)
#convert_toBMP('c:\\opencv\\build\\x86\\vc10\\bin\\Hand\\pos','D:\\Database\\SL\\neg')
#   detectHandFolder('D:\\Database\\Database SL\\Pet 2002\\shp_triesch\\Triesch_bmp')