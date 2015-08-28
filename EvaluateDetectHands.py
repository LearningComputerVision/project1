'''
Created on Aug 18, 2015

@author: Anh
'''
import cv2
import cv2.cv as cv
import numpy as np
import scipy.io as sio
import scipy
import glob,os
import sys
from types import NoneType
import time
def hist(name):
    img=cv2.imread(name)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hist=cv2.equalizeHist(gray)
    cv2.imshow('ogrinal', img)
    cv2.imshow('result', img_hist)
    cv2.waitKey(0)
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
#     print '++++++++++++++++'
#     print x1,y1,x2,y2
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
    while len(idxs)>0:
        ''' 
        grab the last index in the indexes list, add the index
        value to the list of picked indexes, then initialize
        the suppression list (i. e. indexes that will be deteted)
        using the last index
        '''
#         print 'len= ' + str(len(idxs))
        last=len(idxs)-1
        i=idxs[last]
        pick.append(i)
        suppress=[last]

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
        overlap = (w * h) / area[idxs[:last]]
        '''
        if there is sufficient overlap, supress the current bounding box
        '''
        '''
        delete all indexes from the index list that are in suppression list
        suppression list
        '''
        # delete all indexes from the index list that have
       
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
        
        '''
        return only the bounding boxes that were picked
        '''
   
    return boxes[pick].astype("int")
'''
###############################################
##Function detect hand region by local feature
###############################################
'''
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
    boundingBoxes=hands
    print 'boundingBoxes'
    print boundingBoxes
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
    return pick, original_img

'''
###########################################
##Function detect hand region by skin color
###########################################
'''
# Using CYrYk space
def Detect_YCrCb(img, lowerb, upperb):
   
    img_Ycrcb=cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    #find region with skin
    skinRegion=cv2.inRange(img_Ycrcb, lowerb, upperb)
    # Do contour detection on skin region
    contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the contour on the source image
    skin_boxes=[]
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(img, contours, i, (0, 255, 0), 3)
            rect_box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(rect_box)
            box = np.int0(box)
#             print box
            #calculate bounding box
            w=int(np.max([box[0][0],box[1][0],box[2][0],box[3][0]])-np.min([box[0][0],box[1][0],box[2][0],box[3][0]]))
            h=int(np.max([box[0][1],box[1][1],box[2][1],box[3][1]])-np.min([box[0][1],box[1][1],box[2][1],box[3][1]]))
            x_start=np.min([box[0][0],box[1][0],box[2][0],box[3][0]])
            y_start=np.min([box[0][1],box[1][1],box[2][1],box[3][1]])
           
            skin_boxes.append([int(x_start), int(y_start), w, h])
            
#             cv2.line(img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0,255,255),2)
#             cv2.line(img, (box[0][0], box[0][1]), (box[3][0], box[3][1]), (0,255,255),2)
#             cv2.line(img, (box[1][0], box[1][1]), (box[2][0], box[2][1]), (0,255,255),2)
#             cv2.line(img, (box[2][0], box[2][1]), (box[3][0], box[3][1]), (0,255,255),2)
            
            
#             cv2.drawContours(img,[box], 0,(0,0,255),2)
#     print skin_boxes        
#     cv2.imshow('test', img)
#     cv2.waitKey(0)
    return skin_boxes, img
# Using HSV space
def Detect_HSV(img, lowerb, upperb):
    img_hsv=cv2.cvtColor(img, cv.CV_BGR2HSV)
    skinMask=cv2.inRange(img_hsv, lowerb, upperb)
    img_hsv=skinMask
    skin_boxes=[]
    #using an elliptical kernel
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    skinMask=cv2.erode(skinMask, kernel, iterations=2)
    skinMask=cv2.dilate(skinMask, kernel, iterations=2)
    # blur the mask to help remove noise, then apply the
    # mask to img
    skinMask=cv2.GaussianBlur(skinMask, (3,3), 0)
    skin_hsv=cv2.bitwise_and(img, img, mask=skinMask)
    print type(skinMask)
    print type(skin_hsv)
    #detect contour
    contours, hierarchy = cv2.findContours(img_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 100:
            cv2.drawContours(img, contours, i, (0, 255, 0), 3)
            rect_box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(rect_box)
            box = np.int0(box)
#             cv2.drawContours(skin_hsv,[box],0,(0,0,255),2)
            #calculate bounding box
            w=int(np.max([box[0][0],box[1][0],box[2][0],box[3][0]])-np.min([box[0][0],box[1][0],box[2][0],box[3][0]]))
            h=int(np.max([box[0][1],box[1][1],box[2][1],box[3][1]])-np.min([box[0][1],box[1][1],box[2][1],box[3][1]]))
            x_start=np.min([box[0][0],box[1][0],box[2][0],box[3][0]])
            y_start=np.min([box[0][1],box[1][1],box[2][1],box[3][1]])
            skin_boxes.append([int(x_start), int(y_start), w, h])

#     cv2.imshow('demo', skin_hsv)
#     cv2.waitKey(0)
    return skin_boxes, img
'''
###########################################
##Function detect hand region by shape
###########################################
'''
def Shape_HullContour(img):
   #convert to binary
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (3,3), 1)
    
    shape_boxes=[]
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    #noise removal
    kernel=np.ones((3,3),np.uint8)
    opening=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
#    #sure background area
    sure_bg=cv2.dilate(opening, kernel)
    
#     #finding sure foreground area
    dist_transform=cv2.distanceTransform(opening, cv2.DIST_LABEL_PIXEL, 5)
    ret, sure_fg=cv2.threshold(dist_transform, 0.7*dist_transform.max(),255,0)
    
#     # Finding unknown region
    sure_fg=np.uint8(sure_fg)
    unknown=cv2.subtract(sure_bg, sure_fg)

    contours,hierarchy = cv2.findContours(unknown,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    max_area=-1
    min_area=-1
    for idx in range(len(contours)):
        cnt=contours[idx]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci_max=idx
        if (min_area==-1 or min_area>area):
            min_area=area
            ci_min=idx
            
    average_cnt=(max_area-min_area)/2.0
    print average_cnt
    if average_cnt<=0:
        return None
   
    for cnt in contours:
        #calculate average area
        area = cv2.contourArea(cnt)
        if (area<=average_cnt and area>min_area):
            hull=cv2.convexHull(cnt,returnPoints = False)
            col,row= hull.shape[0:]
            print col, row
            if row>=1:
#                 defects=cv2.convexityDefects(cnt, hull)
#                 if type(defects)!= NoneType:
#                     for i in defects:
# #                         print i
#                         s,e,f,d=i[0]
#                         start=tuple(cnt[s][0])
#                         end=tuple(cnt[e][0])
#                         far=tuple(cnt[f][0])
#                         cv2.line(img, start, end, [0, 255, 0], 2)
#                         cv2.circle(img, far, 5, [0,0,255],-1)
                rect = cv2.minAreaRect(cnt)
                box = cv2.cv.BoxPoints(rect)
                box = np.int0(box)
                #calculate bounding box
                w=int(np.max([box[0][0],box[1][0],box[2][0],box[3][0]])-np.min([box[0][0],box[1][0],box[2][0],box[3][0]]))
                h=int(np.max([box[0][1],box[1][1],box[2][1],box[3][1]])-np.min([box[0][1],box[1][1],box[2][1],box[3][1]]))
                x_start=np.min([box[0][0],box[1][0],box[2][0],box[3][0]])
                y_start=np.min([box[0][1],box[1][1],box[2][1],box[3][1]])
                shape_boxes.append([int(x_start), int(y_start), w, h])
#                 cv2.drawContours(img, [box], 0, (255,0,0),2)       
        
#     cv2.imshow('demo', img)
#     cv2.waitKey(0)
    return shape_boxes, img 
'''
###########################################
##Foreach the image folder of hand
###########################################
'''
def detectHandFolder(in_folder):
    
    folder_image_test=in_folder+"\\images";
    folder_image_annotation=in_folder+"\\annotations";
    list_image_file=[]
    list_annotation_file=[]
    list_name_image_file=[]
    for root, dirname, file in os.walk(folder_image_test): 
        for filename in file:
            filepath=os.path.join(root,filename)
            list_image_file.append(filepath)#get file path
            list_name_image_file.append(filename)
            
    for root, dirname, file in os.walk(folder_image_annotation): 
        for filename in file:
            filepath=os.path.join(root,filename)
            list_annotation_file.append(filepath)#get file path
    boxes=[]
    TP=[];
    NP=[];
    N=[];
    
    #create list file
    for file, annotation_file in zip(list_image_file,list_annotation_file) :
        #Step1: read image
        img=cv2.imread(file)
        print file;
        print annotation_file;
        t0=time.time()
        #Step 2: detect bounding boxes
#         boxes, result=HandDetectionImproved(img)
      
#         boxes, result=Detect_YCrCb(img, np.array((0,133,77)),np.array((255,173,127)))
        boxes, result=Detect_HSV(img, np.array((0,48,80)),np.array((20,255,255)))
#         boxes, result=Shape_HullContour(img)
        boxes=np.asarray(boxes)
        print "elapsed times: ", time.time()-t0
        cv2.imshow('demo', result)
        cv2.waitKey(0)
#         #Step 3: read information from annountation mat file
        annotation, annotation_fixed, img=loadMatFile(annotation_file, img);
# # #         #Show image result
        drawImage(img,boxes, annotation_fixed)
#         #calculate Accuracy rate
        TP_i=NP_i=0;
        N_i=len(annotation_fixed);
        if len(boxes)>0:
            TP_i, NP_i, N_i=boxoverlap(boxes,annotation_fixed,0.1);
        TP.append(TP_i);
        NP.append(NP_i);
        N.append(N_i);
#      #write Accuracy Rate to file
#     writeInformationAccuracy('result_file.txt', list_name_image_file, TP, NP, N);
'''
#####################################################
##Read mat file of the annotation of the hand images
#####################################################
'''
def loadMatFile(file_name, img):
    data_contents=sio.loadmat(file_name);
    annotation= data_contents['boxes'];
#     drawing=np.zeros((450,450,3),dtype=np.uint8);
    print '+++++++++++++++++++++++++';
    c, r=annotation.shape[:2];
    annotation_fixed=[]
    for i in range(r):
        x1=annotation[0,i]['a'][0,0]
        pt1=x1[0]
        x2=annotation[0,i]['b'][0,0]
        pt2=x2[0]
        x3=annotation[0,i]['c'][0,0]
        pt3=x3[0]
        x4=annotation[0,i]['d'][0,0]
        pt4=x4[0]

#         cv2.line(img, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])),(0,0,255), 3)
#         cv2.line(img, (int(pt2[1]), int(pt2[0])), (int(pt3[1]), int(pt3[0])),(0,0,255), 3)
#         cv2.line(img, (int(pt1[1]), int(pt1[0])), (int(pt4[1]), int(pt4[0])),(0,0,255), 3)
#         cv2.line(img, (int(pt3[1]), int(pt3[0])), (int(pt4[1]), int(pt4[0])), (0,0,255),3)
        w=int(np.max([pt1[1],pt2[1],pt3[1],pt4[1]])-np.min([pt1[1],pt2[1],pt3[1],pt4[1]]))
        h=int(np.max([pt1[0],pt2[0],pt3[0],pt4[0]])-np.min([pt1[0],pt2[0],pt3[0],pt4[0]]))
        x_start=np.min([pt1[1],pt2[1],pt3[1],pt4[1]])
        y_start=np.min([pt1[0],pt2[0],pt3[0],pt4[0]])
        annotation_fixed.append([int(x_start), int(y_start), w, h])
    print annotation_fixed

    return annotation,annotation_fixed, img;

'''
##############################################
##check box overlap
##############################################
'''
def boxoverlap(regions_a, region_b, thre):
    # (x1,y1) top-left coord, (x2,y2) bottom-right coord, (w,h) size
    TP=NP=0;
    TP_all=NP_all=0
    N=len(region_b);
    for (xb,yb,wb,hb) in region_b:
        x1=np.maximum(regions_a[:,0],xb);
        y1=np.maximum(regions_a[:,1],yb);
        x2=np.minimum((regions_a[:,2]+regions_a[:,0]),(xb+wb));
        y2=np.minimum((regions_a[:,3]+regions_a[:,1]),(yb+hb));
        print x1,y1,x2,y2
        w=x2-x1+1;
        h=y2-y1+1;
        inter=w*h;
        aarea=(regions_a[:,2]+1)*(regions_a[:,3]+1);
#         barea=np.dot((region_b[:,2]-region_b[:,0]+1),(region_b[:,3]-region_b[:,1]+1));
        barea=(wb+1)*(hb+1);
#         print 'intersection'
#         print inter
#         print aarea;
#         print barea;
        #intersection over union overlap
        o=inter/(aarea+float(barea)-inter);
        print "w and h";
        print w,h;
        print 'overlap: ';
        print o;
        #set invalid entries to 0 overlap
        o[w<=0]=0
        o[h<=0]=0
        TP=len(np.extract(o>=thre, o))
        NP=len(np.extract(o<thre, o))
        TP_all=TP_all+TP
        
        print o;
        print TP
        print NP
        print N;
    NP_all=NP-TP_all
    if NP_all<0:
        NP_all=0
    return TP_all, NP_all, N; 
'''
####################################################
## Write the accuracy rate of the test set
####################################################
'''
def writeInformationAccuracy(result_file,name, TP, NP, N):
    #open file
    file=open(result_file,"w");
    file.write("File Name \t\t\t" +"\tTP \t"+"NP \t"+"N \t"+"  ACCURACY_RATE \n" );
    Acc_Rate=0;
    Sum_Rate=0;
    for name_file,TP_i,NP_i,N_i in zip(name,TP, NP, N):
        if TP_i>N_i:
            TP_i=N_i;
        Acc_Rate=TP_i/float(N_i);
        file.write(name_file +"\t\t\t"+str(TP_i)+" \t"+str(NP_i)+" \t "+str(N_i)+" \t\t\t"+str(Acc_Rate)+"\n");
        Sum_Rate=Sum_Rate+Acc_Rate;
    file.write("\nAverage Accuraccy Rate: "+str(Sum_Rate/len(N)));
    #close file
    file.close();
    
def box_overlap(A, B):
    # (x1,y1) top-left coord, (x2,y2) bottom-right coord, (w,h) size
    SA=A[:,2]*A[:,3];
    SB=B[:,2]*B[:,3];
    A_x2=A[:,0]+A[:,2];
    A_y2=A[:,1]+A[:,3];
    B_x2=B[:,0]+B[:,2];
    B_y2=B[:,1]+B[:,3];
    S_intersect=np.max([0, (np.min([A_x2, B_x2])-np.max([A[:,0],B[:,0]])+1)])*np.max([0,(np.min([A_y2,B_y2])-np.max([A[:,1], B[:,1]])+1)])
    S_Union=SA+SB-S_intersect;
    o=float(S_intersect)/float(S_Union);
    print 'overlap between A and B: %f' % o
'''
#####################################
##convert to array points
#####################################
'''
def drawImage(img, boxes, annotation_fixed):
    print boxes
    print annotation_fixed
    for (x,y,w,h) in boxes:
        cv2.rectangle(img, (x,y), (x+w,y+h),255,2)
    for (x,y,w,h) in annotation_fixed:
        cv2.rectangle(img, (x,y), (x+w,y+h),(0,0,255),2)
#     cv2.imshow('demo', img)
#     cv2.waitKey(0)
    
def convertToPoint(boxes):
    boxes_point=[];
    c, r=boxes.shape[:2]
    boxes_region=np.empty(shape=[c,r]);
    for i in range(boxes.shape[0]):
        x1=boxes[i,0];
        y1=boxes[i,1];
        x2=x1+boxes[i,2];
        y2=y1;
        x3=x1;
        y3=y1+boxes[i,3];
        x4=x1+boxes[i,2];
        y4=y1+boxes[i,3];
        
        boxes_point.append([x1, y1]);
        boxes_point.append([x2, y2]);
        boxes_point.append([x3, y3]);
        boxes_point.append([x4, y4]);
        
        boxes_region[i,0]=x1;
        boxes_region[i,1]=y1;
        boxes_region[i,2]=x4;
        boxes_region[i,3]=y4;
        
    return boxes_point, boxes_region
    
if __name__ == '__main__':
#     hist('biensoxe.bmp')
#     a=np.array([[232, 291,  53,  53],
#                 [333, 251,  53,  53],
#                 [184,  60,  24,  24],]);
    c=np.array([[387, 263,  24,  24],
                [302, 107,  59,  59],
                [303,  29, 105, 105]]);
# Detect Hand region

#     boxes_pointa, boxes_a=convertToPoint(a);
# #     annotation, annotation_fixed, img=loadMatFile('hand_data//VOC2007_10.mat');
# #     drawImage(img,c, annotation_fixed)
# #     TP, NP, N=boxoverlap(c,annotation_fixed,0.1);
#     writeInformationAccuracy('result_file.txt', 'VOC2007_10.jpg', TP, NP, N);
    detectHandFolder('D:\\Database\\Database SL\\hand_dataset\\hand_dataset\\test_dataset\\test_data');
#     box_overlap(a,b);
   
#     x=np.array([[3, 3],[2, 2]]);
#     y=np.array([[1, 2],[1, 1]]);
#     z=x*y;
#     print z;