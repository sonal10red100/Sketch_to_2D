# -*- coding: utf-8 -*-
"""
Created on Fri May 29 20:20:39 2020

@author: Chandraprakash Sharm
"""

import os
import glob

img_dir = "result\\app\\output\\" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
detect_files = glob.glob(data_path) #442 elements

#for images
img_dir = "result\\app\\input\\" 
out_path = os.path.join(img_dir, '*g')
input_files = glob.glob(out_path) #1000elements

#for text files
#img_dir = "result\\test\\output_txt\\*.txt"
#input_files = glob.glob(img_dir)

for i in range(0, 1033):
    flag = 0
    for k in range(0, 812):
        a = os.path.basename(str(input_files[i]))[0:os.path.basename(str(input_files[i])).find('.')]
        b = os.path.basename(str(detect_files[k]))[0:os.path.basename(str(detect_files[k])).find('.')]
        if(a == b):
                flag = 1
                break
    if(flag == 0):
        os.remove(input_files[i])

print("Process completed!!")