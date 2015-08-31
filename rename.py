'''
Created on Jul 17, 2015

@author: Anh
'''
import cv2;
import numpy as np
import glob
import os

def rename_file_in_dir(namedir):
    if not os.path.exists(namedir):
        os.mkdir(namedir)
    list_file=[]
    list_dir=[]
    
    for root, dirname, file in os.walk(namedir): 
        idx=1
        for filename in file:
            word = filename.split('.')
            oldfile=word[0]
            subword= oldfile.split('-')
            
            if idx<10:
                num_str='00' + str(idx)
            elif idx<100:
                num_str='0' + str(idx)
            else:
                num_str=str(idx)
            
           
            new_namefile= subword[0]+'-'+ subword[1]+'-' +num_str+'.'+word[1]
            
            filepath_old=os.path.join(root,filename)
            filepath_new=os.path.join(root,new_namefile)
            
            os.rename(filepath_old, filepath_new)
            list_file.append(filepath_new)#get file path
            idx=idx+1
            
    
            
if __name__ == '__main__':
    rename_file_in_dir('d:\\Database\\MMI\\MMI_landmark\\4\\S054-001')