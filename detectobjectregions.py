'''
Created on Aug 9, 2015
@author: Anh
'''
import cv2
import cv2.cv as cv
import numpy as np

def detectObject(filename):
    img=cv.LoadImage(filename)
    '''
    #get color histogram
    '''
   
#     im32f=np.zeros((img.shape[:2]),np.uint32)
    hist_range=[[0,256],[0,256],[0,256]]
    im32f=cv.CreateImage(cv.GetSize(img), cv2.IPL_DEPTH_32F, 3)
    cv.ConvertScale(img, im32f)
    
    
    hist=cv.CreateHist([32,32,32],cv.CV_HIST_ARRAY,hist_range,3)
    '''
    #create three histogram'''
    b=cv.CreateImage(cv.GetSize(im32f), cv2.IPL_DEPTH_32F, 1)
    g=cv.CreateImage(cv.GetSize(im32f), cv2.IPL_DEPTH_32F, 1)
    r=cv.CreateImage(cv.GetSize(im32f), cv2.IPL_DEPTH_32F, 1)
    
   
    '''
    #create image backproject 32f, 8u
    '''
    backproject32f=cv.CreateImage(cv.GetSize(img),cv2.IPL_DEPTH_32F,1)
    backproject8u=cv.CreateImage(cv.GetSize(img),cv2.IPL_DEPTH_8U,1)
    '''
    #create binary
    '''
    bw=cv.CreateImage(cv.GetSize(img),cv2.IPL_DEPTH_8U,1)
    '''
    #create kernel image
    '''
    kernel=cv.CreateStructuringElementEx(3, 3, 1, 1, cv2.MORPH_ELLIPSE)
    cv.Split(im32f, b, g, r,None)

    planes=[b,g,r]
    cv.CalcHist(planes, hist)
    '''
    #find min and max histogram bin.
    '''
    minval=maxval=0.0
    min_idx=max_idx=0
    minval, maxval, min_idx, max_idx=cv.GetMinMaxHistValue(hist)
    '''
    # threshold histogram.  this sets the bin values that are below the threshold
    to zero
    '''
    cv.ThreshHist(hist, maxval/32.0)
    '''
    #backproject the thresholded histogram, backprojection should contian higher values for
    #background and lower values for the foreground
    '''
    cv.CalcBackProject(planes, backproject32f, hist)
    '''
    #convert to 8u type
    '''
    val_min=val_max=0.0
    idx_min=idx_max=0
    val_min,val_max,idx_min,idx_max=cv.MinMaxLoc(backproject32f)
    cv.ConvertScale(backproject32f, backproject8u,255.0/maxval)
    '''
    #threshold backprojected image. this gives us the background
    '''
    cv.Threshold(backproject8u, bw, 10, 255, cv2.THRESH_BINARY)
    '''
    #some morphology on background
    '''
    cv.Dilate(bw, bw,kernel,1)
    cv.MorphologyEx(bw, bw, None,kernel, cv2.MORPH_CLOSE, 2)
    '''
    #get the foreground
    '''
    cv.SubRS(bw,cv.Scalar(255,255,255),bw)
    cv.MorphologyEx(bw, bw, None, kernel,cv2.MORPH_OPEN,2)
    cv.Erode(bw, bw, kernel, 1)
    '''
    #find contours of foreground
    #Grabcut
    '''
    size=cv.GetSize(bw)
    color=np.asarray(img[:,:])
    fg=np.asarray(bw[:,:])
#     mask=cv.CreateMat(size[1], size[0], cv2.CV_8UC1)
    '''
    #Make anywhere black in the grey_image (output from MOG) as likely background
    #Make anywhere white in the grey_image (output from MOG) as definite foreground
    '''
    rect = (0,0,0,0)
   
    mat_mask=np.zeros((size[1],size[0]),dtype='uint8')
    mat_mask[:,:]=fg
    
    mat_mask[mat_mask[:,:] == 0] = 2
    mat_mask[mat_mask[:,:] == 255] = 1
    
    '''
    #Make containers 
    '''                               
    bgdModel = np.zeros((1, 13 * 5))
    fgdModel = np.zeros((1, 13 * 5))
    cv2.grabCut(color, mat_mask, rect, bgdModel, fgdModel,cv2.GC_INIT_WITH_MASK)
    '''
    #Multiple new mask by original image to get cut
    '''
    mask2 = np.where((mat_mask==0)|(mat_mask==2),0,1).astype('uint8')  
    gcfg=np.zeros((size[1],size[0]),np.uint8)
    gcfg=mask2
    
    img_cut = color*mask2[:,:,np.newaxis]

    contours, hierarchy=cv2.findContours(gcfg ,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        print cnt
        rect_box = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(rect_box)
        box = np.int0(box)
        cv2.drawContours(color,[box], 0,(0,0,255),2)
    cv2.imshow('demo', color)
    cv2.waitKey(0)

   
if __name__ == '__main__':
    detectObject('2.jpg')