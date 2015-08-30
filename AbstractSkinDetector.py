'''
Created on Aug 28, 2015

@author: Anh
'''
import cv2
import cv2.cv as cv
import numpy as np

class AbstractSkinDetector:
    '''
    classdocs
    '''
    def __init__(self, params):
        '''
        Constructor
        '''
    def visualizeHist(self, hist, bins, histwin_name):
        hist_img=np.ndarray((hist.shape[0],hist.shape[1],3),np.uint8)
        min_value=max_value=idx_min=idx_max=0
        min_value, max_value, idx_min, idx_max=cv2.minMaxLoc(hist)
        print min_value, max_value, idx_min, idx_max
        print bins
        #Do something here
        intensity=0
#         bins=[250, 250]
        for ubin in xrange(bins[0]):
            for vbin in xrange(bins[1]):
                binVal = hist[vbin,ubin]
#                 if binVal>0:
#                    print binVal
#                 print binVal*255/max_value
                intensity = cv.Round(binVal*255/max_value)
#                 print intensity
                cv2.rectangle(hist_img, (vbin, ubin),(vbin+1,ubin+1),(intensity,255,180),3,cv.CV_FILLED )
        hist_img=cv2.cvtColor(hist_img, cv2.COLOR_HSV2BGR)
#         cv2.imshow("visualize Hist",hist)
        cv2.imshow(histwin_name, hist_img)
        cv2.waitKey(0)
        return hist_img
    def getNormalizedRGB(self, rgb):
        rgb32f=np.float32(rgb)
        b,g,r=cv2.split(rgb32f)
        sum_rgb=b+g+r
        b[:,:]=0
        g=cv2.divide(g, sum_rgb)
        r=cv2.divide(r, sum_rgb)
        split_bgr=cv2.merge((b,g,r))
        return split_bgr
    def calc_2D_hist(self, img, mask, wchannels, bins, low, high):
        histSize=[bins[0],bins[1]]
        uranges=[low[0],high[0]]
        vranges=[low[1],high[1]]
        channels=[wchannels[0],wchannels[1]]
        ranges=uranges+vranges

        print ranges
        print bins
        print mask.shape
        hist=cv2.calcHist([img], channels, mask, bins, ranges)
#         cv2.imshow('hist', hist)
#         cv2.waitKey(0)
#         min_value, max_value, idx_min, idx_max=cv2.minMaxLoc(hist)
#         print min_value, max_value, idx_min, idx_max
        return hist
        
class SkinProbablilityMaps(AbstractSkinDetector):
    def __init__(self):
        '''
        Constructor
        '''
        self.range_dist=np.array((1,2),np.float32)
        self.theta_thresh=8.0
        self.hist_bins=[50,50]
        self.low_range=[0.2,0.3]
        self.high_range=[0.4,0.5]
        self.range_dist[0]=self.high_range[0]-self.low_range[0]
        self.range_dist[1]=self.high_range[1]-self.low_range[1]
        print self.range_dist
        
    def setTheta(self, t):
        self.theta_thresh=t  
        
    def calc_rg_hist(self, img, mask, bins, low, high):
        channels=[1, 2]
#         bins=[250, 250]
#         bins=[img.shape[0],img.shape[1]]
#         low=[0, 0]
#         high=[1, 1]
        print "calc_rg_hist"
        return self.calc_2D_hist(img, mask, channels, bins, low, high)
    
    def boostrap(self, rgb):
        hsv=cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        nrgb=self.getNormalizedRGB(rgb)
        #H=[0,50], S= [0.20,0.68] and V= [0.35,1.0]
        mask_hsv=cv2.inRange(hsv, np.array((0,0.2*255.0,0.35*255.0)), np.array((50.0/2.0,0.68*255.0,1.0*255.0)) );
#         #r = [0.36,0.465], g = [0.28,0.363]
        mask_nrgb=cv2.inRange(nrgb,np.array((0,0.28,0.363)),np.array((1.0,0.363,0.465)));
#         
        #rule from "Automatic Feature Construction and a Simple Rule Induction Algorithm for Skin Detection", Gomez & Morales 2002
        # r/g > 1.185, rb/(r+g+b)^2 > 0.107, rg/(r+g+b)^2 > 0.112
        output_mask=cv2.bitwise_and(mask_hsv, mask_nrgb)
        
        return output_mask
    def train(self, img_rgb, mask):
        nrgb=self.getNormalizedRGB(img_rgb)
        
        self.skin_hist=self.calc_rg_hist(nrgb, mask, self.hist_bins, self.low_range, self.high_range)
        
        non_mask=(~mask)
        
#         cv2.imshow('invert', non_mask)
#         cv2.waitKey(0)
        self.non_skin_hist = self.calc_rg_hist(nrgb,non_mask,self.hist_bins,self.low_range,self.high_range)
         
        #create a probabilty density function
        skin_pixels=cv2.countNonZero(mask)
        non_skin_pixels=cv2.countNonZero(non_mask)
        for ubin in xrange(self.hist_bins[0]):
            for vbin in xrange(self.hist_bins[1]):
                if  self.skin_hist[ubin][vbin]>0:
                    self.skin_hist[ubin][vbin]= self.skin_hist[ubin][vbin]/float(skin_pixels)
                if  self.non_skin_hist[ubin][vbin]>0:
                    self.non_skin_hist[ubin][vbin]= self.non_skin_hist[ubin][vbin]/float(non_skin_pixels)
#         self.visualizeHist( self.skin_hist, self.hist_bins, "Skin hist")
#         self.visualizeHist( self.non_skin_hist, self.hist_bins, "non Skin hist")
        return mask
    
    def predict(self, img_rgb):  
        nrgb=self.getNormalizedRGB(img_rgb)

        nrgb=nrgb.reshape(img_rgb.shape[0]*img_rgb.shape[1],3)
        result_mask=np.zeros((img_rgb.shape[0]*img_rgb.shape[1],1),dtype='uint8')
        print nrgb.shape[0]
#         self.visualizeHist( self.skin_hist, self.hist_bins, "Skin hist")
#         self.visualizeHist( self.non_skin_hist, self.hist_bins, "non Skin hist ")
#         cv2.imshow('demo', nrgb)
#         cv2.waitKey(0)
        print self.range_dist
        print self.high_range
#         self.hist_bins=[250, 250]
#         print self.hist_bins
        for i in xrange(nrgb.shape[0]):
            
            #print nrgb[1][i],nrgb[1][i],self.low_range, self.high_range  
            if nrgb[i][1] < self.low_range[0] or nrgb[i][1] > self.high_range[0] or nrgb[i][2] < self.low_range[1] or nrgb[i][2] > self.high_range[1]:
                result_mask[i] = 0;
                continue;
            gbin=cv.Round((nrgb[i][1] - self.low_range[0])/self.range_dist[0] * self.hist_bins[0])
            rbin=cv.Round((nrgb[i][2] - self.low_range[1])/self.range_dist[1] * self.hist_bins[1])
            
            skin_hist_val= self.skin_hist[gbin % 50][rbin % 50]
            if skin_hist_val>0:
                non_skin_hist_val=self.non_skin_hist[gbin%50][rbin%50]
                if non_skin_hist_val>0:
                    if (skin_hist_val/non_skin_hist_val)>self.theta_thresh:
                        result_mask[i]=255
                    else:
                        result_mask[i]=0
                else:
                    result_mask[i]=0
            else:
                result_mask[i]=0
        output_mask=result_mask
        print cv2.countNonZero(output_mask)   
#         print output_mask.shape
        output_mask = output_mask.reshape((img_rgb.shape[0],img_rgb.shape[1]))
        return output_mask   
class EllipticalBoundaryModel(AbstractSkinDetector): 
    def __init__(self):
        '''
        Constructor
        '''
        self.range_dist=np.array((1,2))
        self.hist_bins = [50,50]
        self.low_range = [0.2,0.3]
        self.high_range = [0.4,0.5]
        self.range_dist[0] = self.high_range[0] - self.low_range[0]
        self.range_dist[1] = self.high_range[1] - self.low_range[1]
    
    def setThetaThresh(self, t):
        self.theta_thresh=t
        
    def train(self):
        ustep = self.range_dist[0]/self.hist_bins[0]
        vstep = self.range_dist[1]/self.hist_bins[1]
        #calc n, X_i and mu
        mu=np.array((1,2))
        n = cv2.countNonZero(self.f_hist);
        count = 0;
        N = 0;
        X=np.array((n,2))
        f=[]
        for ubin in self.hist_bins[0]:
            for vbin in self.hist_bins[1]:
                histval = self.f_hist[ubin][vbin];
                if histval > 0:
                    sampleX = ((1,2) << self.low_range[0] + self.ustep * (ubin+.5), self.low_range[1] + vstep * (vbin+.5))
                    count=count+1
                    sampleX=X.row[count]
                    f.append(histval)
                    mu = mu+ histval * sampleX;
                    N += histval
       
        mu /= N;
        #calc psi - mean of DB
        self.psi=cv2.reduce(X, self.psi,0, cv.CV_REDUCE_AVG)
        #calc Lambda
        self.Lambda=np.zeros((2,2),np.uint)
        for i in n:
            X_m_mu = (X.row[i] - mu)
            prod = f[i] * X_m_mu.t() * X_m_mu
            self.Lambda += prod;
        self.Lambda /= N;
        linv = self.Lambda.inv();
        self.Lambda_inv.val[0] = linv[0][0]
        self.Lambda_inv.val[1] = linv[0][1]
        self.Lambda_inv.val[2] = linv[1][0]
        
    def accumTrain(self, img_rgb, mask): 
        img_cieLuv = self.getNormalizedRGB(img_rgb);
        self.f_hist += self.calc_uv_hist(img_cieLuv, mask);
        self.visualizeHist(self.f_hist,self.hist_bins,"EBM uv hist");
        self.train();
        return img_rgb, mask
    
    def getCIELuvFromBGR(self, img_rgb):
        img_rgbf=float(img_rgb)
        img_cieLuv=cv2.cvtColor(img_rgbf, cv.CV_BGR2Luv);
        return img_cieLuv; 
         
    def getSamplesFromImages(self, img_rgb, mask, img_cieLuv,samples):
        maskedrgb=np.array(img_rgb.shape[0]); 
        mask=float(maskedrgb)
        cv2.imshow("train",maskedrgb);
        cv2.waitKey(0)
        img_cieLuv = self.getNormalizedRGB(img_rgb);
        samples=np.array(cv2.countNonZero(mask),3)
        img_cieLuv_flat = img_cieLuv.reshape(img_cieLuv.channels(), img_cieLuv.rows * img_cieLuv.cols);
        mask_flat = mask.reshape(1,mask.rows*mask.cols);
        count = 0;
        for i in img_cieLuv_flat.shape[0]:
            if mask_flat[i] > 0:
                samples[count][0] = img_cieLuv_flat[i][0];
                samples[count][1] = img_cieLuv_flat[i][1];
                count=count+1
                samples[count][2] = img_cieLuv_flat[i][2];
        return img_rgb,mask,img_cieLuv,samples    
    def calc_uv_hist(self, img, mask):
        channels=[1, 2]
        return self.calc_2D_hist(img,mask,self.channels,self.hist_bins,self.low_range,self.high_range),img,mask
    def trainE(self, img_rgb, mask):
        img_cieLuv = self.getNormalizedRGB(img_rgb)
        self.f_hist = self.calc_uv_hist(img_cieLuv,mask)
        self.visualizeHist(self.f_hist,self.hist_bins,"EBM uv hist")
        self.train()
        self.initialized = True 
        return img_rgb, mask  
    def predict(self, img_rgb, output_mask):
       
        if len(output_mask)==0:
          output_mask=np.array(len(img_rgb), np.uint8)
        
        img_cieLuv = self.getNormalizedRGB(img_rgb);
        img_cieLuv_flat = img_cieLuv.reshape(img_cieLuv.channels(), img_cieLuv.rows * img_cieLuv.cols);
        img_cieLuv_flat_sub = img_cieLuv_flat - [0,self.psi(0),self.psi(1)]
        
        output_mask_Phi=np.array(output_mask.rows*output_mask.cols,1);

       #pragma omp parallel for
        for i in len(img_cieLuv_flat):
            #predict per pixel
            X=np.array((1,2)) 
            X[0] = img_cieLuv_flat_sub[i][1]; 
            X[1] = img_cieLuv_flat_sub[i][2]; #take only u,v
            prod = X * self.Lambda_inv * X.t()
            output_mask_Phi[i] = prod[0]
        output_mask_Phi = output_mask_Phi.reshape(1, output_mask.rows)
        output_mask = output_mask_Phi < self.theta_thresh
        return img_rgb,output_mask      
if __name__ == '__main__':
    
    img=cv2.imread('VOC2007_1.jpg')
    spm = SkinProbablilityMaps()
    mask1=spm.boostrap(img)
    mask1=cv2.medianBlur(mask1,3)
    cv2.imshow('boostrap', mask1)
    for i in xrange(10):
        print "step "+str(i)
        mask1=cv2.medianBlur(mask1,3)
        mask1=spm.train(img,mask1)
        mask1=spm.predict(img)
    cv2.imshow('demo', mask1)
    cv2.waitKey(0)
    cv2.imwrite('spm.bmp', mask1)
    
    