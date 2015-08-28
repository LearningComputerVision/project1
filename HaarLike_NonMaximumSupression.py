'''
Created on Aug 11, 2015

@author: Anh
'''

import sys

import cv2

import cv2.cv as cv
import numpy as np
import glob,os

def HandDetectionImproved(img):   
    
    #get size of input image
    w,h=img.shape[:2]
    #create grayscale version
    gray=np.zeros((h,w,3),np.uint8)

    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #create storage
    storage=cv.CreateMemStorage(0)
    #equalize histogram
    gray1=cv2.equalizeHist(gray)
    gray=cv2.blur(gray1, (5, 5))
    cascade_hand= cv2.CascadeClassifier('haar_cascade.xml')
    
    hands=cascade_hand.detectMultiScale(gray, scaleFactor=1.3, 
                                        minNeighbors=2, minSize=(20,20), 
                                        flags=cv.CV_HAAR_SCALE_IMAGE)
    
    '''
    #loop over the bounding boxes for each image and draw them
    '''
    original_img=img
   
    boundingBoxes=np.array((len(hands),4),np.uint8)
#     print '@@@@@@@@@@@@@@'
#     print hands
    boundingBoxes=hands
#     print 'boundingBoxes'
#     print boundingBoxes
    for (x,y,w,h) in hands:
        cv2.rectangle(original_img, (x,y), (x+w,y+h), (255,0,255))
    '''
    perform non-maximum suppression on the bounding boxes
    '''  
    pick=non_max_suppression_slow(boundingBoxes, 0.5) 
     
    print 'non max suppression'
    print pick
    # loop over the picked bounding boxes and draw them
    for (x,y,w,h) in pick:
        cv2.rectangle(original_img, (x, y), (x+w, y+h), (255, 255, 0), 2)
    return original_img

def clipboxes(im, ds):
    '''
    #ds: Detection bounding boxes
    '''
    rds=ds
    row, col=im.shape[:2]
    idxs=rds[:len(ds)]
    if len(ds)>0:
        rds[:,0]=np.maximum(ds[:,0],1)
        rds[:,1]=np.maximum(ds[:,1],1)
        rds[:,2]=np.minimum(ds[:,2],row)
        rds[:,3]=np.minimum(ds[:,3],col)
        #remove invalid detections
        w=rds[:,3]-rds[:,1]+1
        h=rds[:,4]-rds[:,2]+1
        idxs=np.delete(idxs, np.where(w <0|h<0)[0])
        
'''
#######################################################
# Non-Maximum Suppression for Object Detection in Python
# Felzenszwald et al
#######################################################
'''
def non_max_suppression_slow(boxes, overlapThresh):
    '''
    if there are no boxes, return an empty list
    '''    
    if len(boxes)==0:
        return []
    '''
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    '''
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    '''
    initialize the list of picked indexes
    '''
    pick=[]
    '''
    grab the coordinates of the bounding boxes
    '''
    print boxes
    x1=boxes[:,0]
    y1=boxes[:,1]
    x2=x1+boxes[:,2]
    y2=y1+boxes[:,3]
    print '++++++++++++++++'
    print x1,y1,x2,y2
    '''
    compute the area of the bounding boxes and sort the bounding box
    boxes by the bottom-right y-coordinate of the bounding boxes
    '''
    area=np.abs((x2-x1+1)*(y2-y1+1))
    idxs=np.argsort(y2)
    '''
    keep looping while some indexes still remain in the indexes
    list
    '''
#     print idxs
    while len(idxs)>0:
        ''' 
        grab the last index in the indexes list, add the index
        value to the list of picked indexes, then initialize
        the suppression list (i. e. indexes that will be deteted)
        using the last index
        '''
        print 'len= ' + str(len(idxs))
        last=len(idxs)-1
        i=idxs[last]
        pick.append(i)
        suppress=[last]
#         print "%%%%%%%%%%%%%"
#         print area[:last]
        '''
        loop over all indexes in the indexes list
        '''
        
        '''
        find the largest (x, y) coordinates for the start of 
        the bounding box and the smallest (x, y) coordinate
        for the end of the bounding box
            '''
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        print '--------------'
        print xx1,yy1,xx2,yy2
        '''
        compute the width and height of the bounding box
        '''
        w=np.maximum(0, xx2-xx1+1)
        h=np.maximum(0, yy2-yy1+1)
        '''
        compute the ratio of overlap between the computed
        bounding box and the bounding box in the area list
        '''

        # compute the ratio of overlap
#         print idxs[:last]
        overlap = (w * h) / area[idxs[:last]]
        '''
        if there is sufficient overlap, supress the current bounding box
        '''
        if len(overlap)>1:
            for k in xrange(0,last):
                print 'overlap: ' +'%.2f' % overlap[k]
        

        '''
        delete all indexes from the index list that are in suppression list
        suppression list
        '''
#       idxs=np.delete(idxs, suppress)

        # delete all indexes from the index list that have
       
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
        
        print "pppppppppp"
        print idxs
        '''
        return only the bounding boxes that were picked
        '''
   
    print "*******"
    print pick
    return boxes[pick].astype("int")

def demoHaarLike(namevideo, width, height):
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    outvideo=cv2.VideoWriter(namevideo,-1,60,(width,height))
    idx=0
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
#         frame=cv2.imread('0.bmp')
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        frame=HandDetectionImproved(frame)
#       outvideo.write(frame)

        key = cv2.waitKey(1)
        
        if key==ord('s'):
            cv2.imwrite(str(idx)+'.bmp',frame)
            idx=idx+1
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")
    vc.release()
    outvideo.release()

def detectHandFolder(in_folder):
    list_file=[]
    list_dir=[]
    
    for root, dirname, file in os.walk(in_folder): 
        for filename in file:
            filepath=os.path.join(root,filename)
            list_file.append(filepath)#get file path
    
    #create list file
    for file in list_file:
#         HandDetection(file)
        img=cv2.imread(file)
        result=HandDetectionImproved(img)
        cv2.imshow(file, result)
        cv2.waitKey(0)

def HandDetection(file):   
    
    img=cv2.imread(file,0)
    #get size of input image
    w,h=img.shape[:2]
    #create grayscale version
    gray=np.zeros((h,w,3),np.uint8)
#     gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray=img
    cv2.imshow('winname', gray)
    cv2.waitKey(0)
#     gray=img
#     cv2.imshow('winname', gray)
#     cv2.waitKey(0)
    #create storage
    storage=cv.CreateMemStorage(0)
    #equalize histogram
    gray1=cv2.equalizeHist(gray)
    gray=cv2.blur(gray1, (5, 5))
    cascade_hand= cv2.CascadeClassifier('haar_cascade.xml')
    
    hands=cascade_hand.detectMultiScale(img, scaleFactor=1.3, 
                                        minNeighbors=2, minSize=(24,24), 
                                        flags=cv.CV_HAAR_DO_CANNY_PRUNING)
    
    
    for (x,y,w,h) in hands:
        cv2.rectangle(img, (x,y), (x+w,y+h), 255)
        
    cv2.imshow('winname', img)
    cv2.waitKey(0)
    

if __name__ == '__main__':
#     foldername="D:\\Database\\Database SL\\Pet 2002\\shp_triesch\\Triesch_bmp"
    foldername="D:\\Database\\Database SL\\hand_dataset\\hand_dataset\\test_dataset\\test_data\\images"
    detectHandFolder(foldername)
#     demoHaarLike(sys.argv[1],640,480)